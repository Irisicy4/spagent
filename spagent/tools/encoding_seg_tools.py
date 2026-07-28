"""
Segmentation tools with configurable output ENCODING, for the Stage-II
segmentation-encoding experiments (judge-exp.md N1).

Two tools, both backed by the GroundingDINO (:20022) + SAM2 (:20020) servers:

  ReferringSegmentationTool  - segment the object(s) matching a referring
                               expression (text -> boxes -> masks)
  SemanticSegmentationTool   - segment ALL instances of the given classes
                               (open-vocabulary; per-class instance masks)

Encodings (paper Table 6 seg axes):
  overlay  - S1  alpha=0.5 colored masks over the original image + labels
  instance - S3  one distinct color per instance, numbered labels
  polygon  - S5  TEXT ONLY: simplified polygon coords per instance, no image
  maskonly - S6  masks on black background, original image NOT shown (control)

IMPORTANT plumbing constraints (verified against core/spagent.py):
  - text reaches the controller ONLY via result["description"]
  - images attach ONLY via result["output_path"] / result["vis_path"]
Every encoding therefore always sets "description"; image-bearing encodings
set "output_path"; polygon sets no image paths at all.
"""

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import itertools
import cv2
import numpy as np
import requests

# SAM2 is called once per instance, so a single Flask worker serialises the
# whole fleet (measured: 23.6 s/call under load vs 1.0 s on an idle replica).
# SAM2_URLS lets a run spread its calls over several replicas round-robin.
_SAM2_POOL = [u for u in os.environ.get("SAM2_URLS", "").split(",") if u.strip()]
_SAM2_CYCLE = itertools.cycle(_SAM2_POOL) if _SAM2_POOL else None

sys.path.append(str(Path(__file__).parent.parent))
from core.tool import Tool

logger = logging.getLogger(__name__)

SEG_ENCODINGS = ["overlay", "instance", "polygon", "maskonly", "pixelpoly",
                 "separate", "instancebox", "matrix", "overlayref"]

# distinct BGR colors for instances (same palette as sam2_client)
_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 0, 128), (203, 192, 255), (255, 128, 0),
]
_COLOR_NAMES = [
    "red", "green", "blue", "yellow", "magenta",
    "cyan", "orange", "purple", "pink", "azure",
]


def _gdino_detect(server_url: str, image_path: str, text_prompt: str,
                  box_threshold: float = 0.35) -> List[Dict[str, Any]]:
    """Call the GroundingDINO server directly; return detections with PIXEL xyxy."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    h, w = image.shape[:2]
    _, buf = cv2.imencode(".jpg", image)
    resp = requests.post(
        f"{server_url.rstrip('/')}/infer",
        json={
            "image": base64.b64encode(buf.tobytes()).decode(),
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": 0.25,
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    if not result.get("success"):
        raise RuntimeError(f"gdino error: {result.get('message') or result.get('error')}")
    dets = []
    for d in result.get("detections", []):
        bbox = list(map(float, d["bbox"]))
        # The :20022 server passes GroundingDINO's raw predict() boxes through
        # under an xyxy-named key, but they are NORMALIZED CXCYWH. Convert.
        if max(bbox) <= 1.5:
            cx, cy, bw, bh = bbox
            bbox = [(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h]
        bbox = [min(max(bbox[0], 0), w - 1), min(max(bbox[1], 0), h - 1),
                min(max(bbox[2], 1), w), min(max(bbox[3], 1), h)]
        dets.append({"bbox": bbox, "label": d.get("label", "object"),
                     "confidence": float(d.get("confidence") or 0.0)})
    dets.sort(key=lambda d: -d["confidence"])
    return dets


def _sam2_mask(server_url: str, image_bgr: np.ndarray, box: List[float]) -> Optional[np.ndarray]:
    """Call the SAM2 server with a box prompt; return a binary mask (H,W) or None."""
    if _SAM2_CYCLE is not None:
        server_url = next(_SAM2_CYCLE)
    _, buf = cv2.imencode(".jpg", image_bgr)
    resp = requests.post(
        f"{server_url.rstrip('/')}/infer",
        json={
            "image": base64.b64encode(buf.tobytes()).decode(),
            "box": [int(v) for v in box],
            "conf": 0.5,
        },
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    if not result.get("success") or not result.get("mask"):
        return None
    arr = cv2.imdecode(np.frombuffer(base64.b64decode(result["mask"]), np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    if arr.shape[:2] != image_bgr.shape[:2]:
        arr = cv2.resize(arr, (image_bgr.shape[1], image_bgr.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return (arr > 0).astype(np.uint8)


def _polygons_of(mask: np.ndarray, max_points: int = 24,
                 pixel: bool = False) -> List[List[List[float]]]:
    """Simplified polygon(s) of a binary mask.

    pixel=False: normalized xy, 3 decimals. pixel=True: integer pixel xy
    (matches Qwen2.5-VL's absolute-coordinate grounding pretraining).
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
        if cv2.contourArea(c) < 25:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) > max_points:
            step = len(approx) // max_points + 1
            approx = approx[::step]
        # NOTE: cast through float()/int() -- numpy>=2 reprs scalars as
        # "np.float64(0.35)", which would leak into the injected text.
        if pixel:
            polys.append([[int(x), int(y)] for x, y in approx])
        else:
            polys.append([[round(float(x) / w, 3), round(float(y) / h, 3)]
                          for x, y in approx])
    return polys


class _EncodedSegTool(Tool):
    """Shared machinery: detect -> mask -> render per the configured encoding."""

    def __init__(self, name: str, description: str, encoding: str,
                 gdino_url: str, sam2_url: str, max_instances: int = 10):
        if encoding not in SEG_ENCODINGS:
            raise ValueError(f"encoding must be one of {SEG_ENCODINGS}; got '{encoding}'")
        super().__init__(name=name, description=description)
        self.encoding = encoding
        self.gdino_url = gdino_url
        self.sam2_url = sam2_url
        self.max_instances = max_instances

    def _segment(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        dets = _gdino_detect(self.gdino_url, image_path, text_prompt)[: self.max_instances]
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        instances = []
        for d in dets:
            mask = _sam2_mask(self.sam2_url, image, d["bbox"])
            if mask is None or mask.sum() == 0:
                continue
            instances.append({"label": d["label"], "confidence": d["confidence"],
                              "bbox": d["bbox"], "mask": mask,
                              "area_frac": round(float(mask.sum()) / (h * w), 4)})
        return {"image": image, "instances": instances, "hw": (h, w)}

    def _render(self, seg: Dict[str, Any], stem: str, query: str) -> Dict[str, Any]:
        image, instances = seg["image"], seg["instances"]
        os.makedirs("outputs", exist_ok=True)
        enc = self.encoding

        if not instances:
            return {
                "success": True, "num_instances": 0,
                "description": (f"Segmentation found NO instances matching "
                                f"'{query}' in the image."),
            }

        # per-instance text lines (labels, colors where relevant, area, bbox)
        lines = []
        for i, ins in enumerate(instances):
            x1, y1, x2, y2 = [int(v) for v in ins["bbox"]]
            color_part = f", color={_COLOR_NAMES[i % len(_COLOR_NAMES)]}" if enc == "instance" else ""
            lines.append(f"  #{i + 1} {ins['label']}{color_part}: bbox=({x1},{y1},{x2},{y2}), "
                         f"area={ins['area_frac'] * 100:.1f}% of image")

        raw = {"success": True, "num_instances": len(instances),
               "labels": [i["label"] for i in instances],
               "boxes": [i["bbox"] for i in instances]}

        if enc == "overlayref":
            # Stage-I "Overlay" (instance-seg WORST, median .508), faithful to
            # /raid/icy/vision-judge-encoding encoders/instance_segmentation.py
            # draw(): the blend covers the WHOLE FRAME, not just the mask, so the
            # photo is darkened everywhere behind a mostly-black layer. Colour is
            # BY CLASS, boundaries are drawn after the blend at full opacity, and
            # there are no labels and no boxes.
            layer = np.zeros_like(image)
            class_col = {}
            for ins in instances:
                col = class_col.setdefault(
                    ins["label"], _COLORS[len(class_col) % len(_COLORS)])
                layer[ins["mask"].astype(bool)] = col
            canvas = cv2.addWeighted(layer, 0.5, image, 0.5, 0)
            for ins in instances:
                cnts, _ = cv2.findContours(ins["mask"], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 2)
            out = f"outputs/{self.name}_{self.encoding}_{stem}.png"
            cv2.imwrite(out, canvas)
            raw["output_path"] = out
            raw["description"] = (
                f"Segmentation of '{query}': {len(instances)} instance(s), shown as "
                f"colour-per-class masks blended over the image; region boundaries "
                f"are outlined in white.")
            return raw

        if enc == "matrix":
            # Stage-I reference "Text" for semantic seg: sub-sampled class-id
            # matrix -- majority vote over 20x20-pixel cells, one line per grid
            # row of space-separated integers, IDs remapped 1..N with a legend.
            h, w = seg["hw"]
            cell = 20
            rows_n, cols_n = max(1, h // cell), max(1, w // cell)
            grid = np.zeros((rows_n, cols_n), dtype=int)
            legend = []
            for i, ins in enumerate(instances):
                legend.append(f"{i + 1} = {ins['label']}")
                m = ins["mask"][:rows_n * cell, :cols_n * cell]
                # block-mean by reshape: majority vote inside each cell
                occ = m.reshape(rows_n, cell, cols_n, cell).mean(axis=(1, 3)) > 0.5
                grid[(grid == 0) & occ] = i + 1
            body = "\n".join(" ".join(str(int(v)) for v in row) for row in grid)
            raw["description"] = (
                f"Segmentation of '{query}' as a sub-sampled class-id matrix "
                f"({rows_n}x{cols_n} cells, each cell = {cell}x{cell} image pixels; "
                f"0 = background).\nLegend: " + "; ".join(legend) + "\n" + body)
            return raw  # text-only

        if enc in ("polygon", "pixelpoly"):
            h, w = seg["hw"]
            pixel = enc == "pixelpoly"
            poly_lines = []
            for i, ins in enumerate(instances):
                polys = _polygons_of(ins["mask"], pixel=pixel)
                poly_lines.append(f"  #{i + 1} {ins['label']} "
                                  f"(area {ins['area_frac'] * 100:.1f}%): polygons={polys}")
            if pixel:
                head = (f"Segmentation of '{query}': {len(instances)} instance(s). "
                        f"The image is {w}x{h} pixels (width x height). Mask outlines "
                        f"as (x,y) PIXEL-coordinate polygons (origin top-left):\n")
            else:
                head = (f"Segmentation of '{query}': {len(instances)} instance(s). "
                        f"Mask outlines as normalized (x,y) polygons "
                        f"(origin top-left, values 0-1):\n")
            raw["description"] = (
                head + "\n".join(poly_lines) +
                "\nNo visualization image is provided; use the polygon coordinates.")
            return raw  # text-only: NO output_path / vis_path

        if enc == "overlay":
            canvas = image.copy()
            for ins in instances:
                m = ins["mask"].astype(bool)
                col = np.zeros_like(canvas); col[m] = _COLORS[0]
                canvas[m] = cv2.addWeighted(canvas[m], 0.5, col[m], 0.5, 0)
            head = f"Segmented {len(instances)} instance(s) matching '{query}' (red overlay"
        elif enc == "instance":
            canvas = image.copy()
            for i, ins in enumerate(instances):
                m = ins["mask"].astype(bool)
                col = np.zeros_like(canvas); col[m] = _COLORS[i % len(_COLORS)]
                canvas[m] = cv2.addWeighted(canvas[m], 0.5, col[m], 0.5, 0)
            head = (f"Segmented {len(instances)} instance(s) matching '{query}' "
                    f"(one distinct color per instance")
        elif enc == "separate":
            # Stage-I reference "Separate": masks composited OPAQUELY in solid
            # green on a pure black canvas -- no photo, no boxes, no labels.
            canvas = np.zeros_like(image)
            for ins in instances:
                canvas[ins["mask"].astype(bool)] = (0, 255, 0)
            head = (f"Segmentation of '{query}' shown as solid green regions on a "
                    f"black canvas; the original image is provided separately "
                    f"({len(instances)} instance(s)")
        elif enc == "instancebox":
            # Stage-I reference "Color-by-Instance": per-instance colour blended
            # over the photo, PLUS a per-instance box and numbered label.
            canvas = image.copy()
            for i, ins in enumerate(instances):
                m = ins["mask"].astype(bool)
                col = _COLORS[i % len(_COLORS)]
                layer = np.zeros_like(canvas); layer[m] = col
                canvas[m] = cv2.addWeighted(canvas[m], 0.5, layer[m], 0.5, 0)
                x1, y1, x2, y2 = [int(v) for v in ins["bbox"]]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2)
                cv2.putText(canvas, f"{ins['label']} {i + 1}", (x1 + 3, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            head = (f"Segmented {len(instances)} instance(s) matching '{query}' "
                    f"(one distinct colour per instance, each with its own box and "
                    f"numbered label")
        else:  # maskonly
            canvas = np.zeros_like(image)
            for i, ins in enumerate(instances):
                canvas[ins["mask"].astype(bool)] = _COLORS[i % len(_COLORS)]
            head = (f"Segmentation masks for '{query}' shown on a BLACK background — "
                    f"the original image is NOT included in this visualization "
                    f"({len(instances)} instance(s)")

        # numbered label anchors help the controller reference instances
        for i, ins in enumerate(instances):
            x1, y1 = int(ins["bbox"][0]), int(ins["bbox"][1])
            cv2.putText(canvas, f"{i + 1}", (x1 + 3, max(y1 + 18, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out = f"outputs/{self.name}_{self.encoding}_{stem}.png"
        cv2.imwrite(out, canvas)
        raw["output_path"] = out
        raw["description"] = head + ", numbered by instance):\n" + "\n".join(lines)
        return raw


class ReferringSegmentationTool(_EncodedSegTool):
    """Segment the object(s) referred to by a natural-language expression."""

    def __init__(self, encoding: str = "overlay",
                 gdino_url: str = "http://localhost:20022",
                 sam2_url: str = "http://localhost:20020"):
        super().__init__(
            name="referring_segmentation_tool",
            description=("Segment the object(s) in the image that match a referring "
                         "expression (e.g. 'the red car on the left'). Returns "
                         "precise pixel masks of the referred object(s)."),
            encoding=encoding, gdino_url=gdino_url, sam2_url=sam2_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string",
                               "description": "Path to the input image."},
                "referring_expression": {
                    "type": "string",
                    "description": "Natural-language description of the object(s) to segment."},
            },
            "required": ["image_path", "referring_expression"],
        }

    def call(self, image_path: str, referring_expression: str) -> Dict[str, Any]:
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            seg = self._segment(image_path, referring_expression)
            return self._render(seg, Path(image_path).stem, referring_expression)
        except Exception as e:
            logger.error(f"referring segmentation error: {e}")
            return {"success": False, "error": str(e)}


class SemanticSegmentationTool(_EncodedSegTool):
    """Open-vocabulary semantic/instance segmentation of the given classes."""

    def __init__(self, encoding: str = "overlay",
                 gdino_url: str = "http://localhost:20022",
                 sam2_url: str = "http://localhost:20020"):
        super().__init__(
            name="semantic_segmentation_tool",
            description=("Segment ALL instances of the given object classes in the "
                         "image (semantic/instance segmentation). Returns one mask "
                         "per instance and per-class instance counts."),
            encoding=encoding, gdino_url=gdino_url, sam2_url=sam2_url,
            max_instances=15)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {"type": "string",
                               "description": "Path to the input image."},
                "class_names": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Object classes to segment, e.g. ['chair', 'person']."},
            },
            "required": ["image_path", "class_names"],
        }

    def call(self, image_path: str, class_names: List[str]) -> Dict[str, Any]:
        try:
            if not Path(image_path).exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            if isinstance(class_names, str):
                class_names = [c.strip() for c in class_names.replace(".", ",").split(",") if c.strip()]
            prompt = " . ".join(class_names)
            seg = self._segment(image_path, prompt)
            result = self._render(seg, Path(image_path).stem, prompt)
            if result.get("num_instances", 0) > 0:
                counts: Dict[str, int] = {}
                for lb in result.get("labels", []):
                    counts[lb] = counts.get(lb, 0) + 1
                count_str = "; ".join(f"{k}: {v} instance(s)" for k, v in counts.items())
                result["description"] = (result["description"] +
                                         f"\nPer-class instance counts: {count_str}.")
            return result
        except Exception as e:
            logger.error(f"semantic segmentation error: {e}")
            return {"success": False, "error": str(e)}
