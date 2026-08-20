"""
OneFormer Tool — universal image segmentation (semantic / instance / panoptic).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import SEGMENTATION, SegmentationPayload, ToolResult

logger = logging.getLogger(__name__)

_ENVELOPE_KEYS = (
    "success", "description", "error", "category", "payload",
    "output_path", "vis_path", "overlay_path", "crop_paths", "mask_path",
)


class OneFormerTool(Tool):
    """Universal image segmentation using OneFormer (semantic, instance, panoptic)."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        super().__init__(
            name="oneformer_tool",
            description=(
                "OneFormer: universal image segmentation supporting three tasks — "
                "semantic (pixel-level class labels), instance (individual object masks), "
                "and panoptic (combined semantic + instance). Returns an annotated "
                "segmentation image, a mask artifact, and a list of detected segments "
                "with labels and scores."
            ),
        )
        self.use_mock = use_mock
        self._server_url = server_url
        self._client = None
        self._client_kwargs = dict(model_id=model_id, device=device)

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockOneFormerClient()
        elif self._server_url:
            from external_experts.OneFormer.oneformer_client import OneFormerClient
            self._client = OneFormerClient(self._server_url)
        else:
            from external_experts.OneFormer.oneformer_local import OneFormerLocalClient
            self._client = OneFormerLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input RGB image.",
                },
                "task": {
                    "type": "string",
                    "enum": ["semantic", "instance", "panoptic"],
                    "description": (
                        "Segmentation task to run. "
                        "'semantic': label every pixel with a class. "
                        "'instance': detect and mask individual object instances. "
                        "'panoptic': combine semantic and instance segmentation. "
                        "Default: 'panoptic'."
                    ),
                },
            },
            "required": ["image_path"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        task: str = "panoptic",
    ) -> Dict[str, Any]:
        if isinstance(image_path, list):
            image_path = image_path[0]

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        if task not in ("semantic", "instance", "panoptic"):
            return {
                "success": False,
                "error": f"Invalid task '{task}'. Choose semantic/instance/panoptic.",
            }

        try:
            self._ensure_client()
            raw = self._client.segment(image_path=image_path, task=task)
            if not raw.get("success"):
                return {
                    "success": False,
                    "error": raw.get("error", "OneFormer segmentation failed"),
                }

            mask_path = raw.get("mask_path")
            masks = raw.get("masks")
            if masks or mask_path:
                payload = SegmentationPayload(
                    masks=masks if masks else None,
                    mask_path=mask_path,
                )
                category = None
            else:
                payload = None
                category = SEGMENTATION

            description = raw.get("description") or (
                f"OneFormer {task} segmentation: "
                f"{raw.get('num_segments', 0)} segment(s)."
            )
            extras = {k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS}
            # Keep nested result for legacy consumers
            extras.setdefault(
                "result",
                {
                    "task": raw.get("task", task),
                    "segments": raw.get("segments", []),
                    "num_segments": raw.get("num_segments", 0),
                    "mask_path": mask_path,
                },
            )
            return ToolResult(
                success=True,
                payload=payload,
                category=category,
                description=description,
                output_path=raw.get("output_path"),
                vis_path=raw.get("output_path"),
                mask_path=mask_path,
                **extras,
            )
        except Exception as e:
            logger.exception("OneFormerTool error")
            return {"success": False, "error": str(e)}


class _MockOneFormerClient:
    def segment(self, image_path: str, task: str = "panoptic", **kwargs) -> Dict[str, Any]:
        try:
            from PIL import Image
            w, h = Image.open(image_path).size
        except Exception:
            w, h = 64, 64

        segments = [
            {"id": 1, "label": "wall", "score": 0.92},
            {"id": 2, "label": "floor", "score": 0.88},
            {"id": 3, "label": "chair", "score": 0.75},
        ]

        stem = Path(image_path).stem
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"oneformer_{task}_{stem}_mock.png")
        mask_path = str(out_dir / f"oneformer_{task}_{stem}_mock_mask.png")

        # Prefer OpenCV/numpy when available; otherwise write a tiny valid PNG.
        wrote = False
        try:
            import numpy as np
            import cv2
            seg_map = np.zeros((h, w), dtype=np.uint16)
            seg_map[: h // 3, :] = 1
            seg_map[h // 3 : 2 * h // 3, :] = 2
            seg_map[2 * h // 3 :, :] = 3
            color = np.zeros((h, w, 3), dtype=np.uint8)
            color[seg_map == 1] = (120, 120, 120)
            color[seg_map == 2] = (180, 120, 120)
            color[seg_map == 3] = (6, 230, 230)
            cv2.imwrite(output_path, color)
            cv2.imwrite(mask_path, seg_map)
            wrote = True
        except Exception:
            pass
        if not wrote:
            try:
                from PIL import Image as PILImage
                # 3 horizontal bands as L-mode ids
                px = []
                for y in range(h):
                    band = 1 if y < h // 3 else (2 if y < 2 * h // 3 else 3)
                    px.extend([band] * w)
                mask_img = PILImage.new("L", (w, h))
                mask_img.putdata(px)
                mask_img.save(mask_path)
                mask_img.convert("RGB").save(output_path)
                wrote = True
            except Exception:
                # Absolute fallback: minimal 1x1 PNG bytes
                png_1x1 = (
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                    b"\x00\x00\x00\x01\x08\x00\x00\x00\x00:\x7e\x9b\x55"
                    b"\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
                    b"\xe5'\xde\xfc\x00\x00\x00\x00IEND\xaeB`\x82"
                )
                Path(mask_path).write_bytes(png_1x1)
                Path(output_path).write_bytes(png_1x1)

        return {
            "success": True,
            "task": task,
            "segments": segments,
            "num_segments": len(segments),
            "output_path": output_path,
            "mask_path": mask_path,
            "shape": [h, w],
            "description": (
                f"[mock] OneFormer {task} segmentation: 3 segments "
                f"(wall, floor, chair)."
            ),
        }
