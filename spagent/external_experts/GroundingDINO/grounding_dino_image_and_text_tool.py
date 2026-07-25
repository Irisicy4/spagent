"""
GroundingDINO Image-and-Text Tool

Wraps the GroundingDINO detection server and returns both:
- An annotated image with bounding boxes drawn on the original image (output_path / vis_path)
- A text description listing detected objects and their normalised bounding boxes

The text description is injected into the continuation prompt via the `description`
field so the LLM receives redundant visual + textual information about detections.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.tool import Tool
from external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient

logger = logging.getLogger(__name__)


class GroundingDINOImageAndTextTool(Tool):
    """
    Object detection tool that returns both an annotated image and text results.

    The annotated image is passed to the LLM as an additional image in subsequent
    iterations. The `description` field contains a JSON-formatted list of detected
    objects with normalised [x1, y1, x2, y2] bounding boxes, which is injected as
    text into the continuation prompt.
    """

    def __init__(self, server_url: str = "http://localhost:20022"):
        """
        Args:
            server_url: Base URL of the running GroundingDINO Flask server,
                        e.g. "http://10.8.131.51:20022".
        """
        super().__init__(
            name="detect_objects_tool",
            description=(
                "Detect objects in an image using GroundingDINO. "
                "Returns an annotated image with bounding boxes drawn on it, "
                "plus a text list of detected objects with their normalised "
                "[x1, y1, x2, y2] bounding boxes and labels."
            ),
        )
        self.server_url = server_url
        self._client = GroundingDINOClient(server_url=server_url)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the input image.",
                },
                "text_prompt": {
                    "type": "string",
                    "description": (
                        "Comma- or dot-separated list of object categories to detect, "
                        "e.g. \"person, baseball bat\" or \"car.tree\"."
                    ),
                },
                "box_threshold": {
                    "type": "number",
                    "description": "Minimum confidence score for a detected box (default 0.35).",
                    "default": 0.35,
                },
                "text_threshold": {
                    "type": "number",
                    "description": "Minimum text-matching score (default 0.25).",
                    "default": 0.25,
                },
            },
            "required": ["image_path", "text_prompt"],
        }

    def call(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Run GroundingDINO detection and return annotated image + text results.

        Returns:
            {
                "success": True,
                "output_path": "<path to annotated image>",
                "vis_path": "<same path>",
                "detections": [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...],
                "boxes": [[x1, y1, x2, y2], ...],
                "labels": ["label1", ...],
                "confidence": [0.9, ...],
                "description": "Detected objects: [{\"label\": ..., \"bbox\": [...]}, ...]"
            }
        """
        image_path = str(image_path)

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            raw = self._client.infer(
                image_path=image_path,
                text_prompt=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except Exception as exc:
            logger.error("GroundingDINO client error: %s", exc)
            return {"success": False, "error": str(exc)}

        if raw is None or not raw.get("success"):
            msg = (raw or {}).get("message", "No detections or server error")
            logger.warning("Detection returned no results: %s", msg)
            return {
                "success": True,
                "output_path": None,
                "vis_path": None,
                "detections": [],
                "boxes": [],
                "labels": [],
                "confidence": [],
                "description": "Detected objects: []",
            }

        # Build normalised detection list for text description
        detections: List[Dict[str, Any]] = []
        boxes, labels, confidences = [], [], []
        for det in raw.get("detections", []):
            norm_bbox = [round(v, 4) for v in det["bbox"]]
            detections.append({"label": det["label"], "bbox": norm_bbox})
            boxes.append(norm_bbox)
            labels.append(det["label"])
            confidences.append(round(det.get("confidence", 0.0), 4))

        description = "Detected objects: " + json.dumps(detections, ensure_ascii=False)
        logger.info("Detected %d object(s): %s", len(detections), description)

        output_path = raw.get("output_path")
        return {
            "success": True,
            "output_path": output_path,
            "vis_path": output_path,
            "detections": detections,
            "boxes": boxes,
            "labels": labels,
            "confidence": confidences,
            "description": description,
        }
