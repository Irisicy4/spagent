"""
GroundingDINO Text-Only Tool

Wraps the GroundingDINO detection server and returns only text-format
detection results: a list of {"label": ..., "bbox": [x1, y1, x2, y2]}
where bbox values are normalized to [0, 1] by image width/height.

Example output:
    [
        {"label": "baseball bat", "bbox": [0.27, 0.70, 0.51, 0.83]},
        {"label": "person",       "bbox": [0.07, 0.25, 0.48, 1.00]}
    ]
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


class GroundingDINOTextOnlyTool(Tool):
    """
    Object detection tool that returns only text results (no annotated images).

    The server returns bounding boxes already normalized to [0, 1] xyxy format.
    This tool passes them through directly as a JSON-serialisable list of
    {"label": <str>, "bbox": [x1, y1, x2, y2]} dicts.
    """

    def __init__(self, server_url: str = "http://localhost:20022"):
        """
        Args:
            server_url: Base URL of the running GroundingDINO Flask server,
                        e.g. "http://10.8.131.51:20022".
        """
        super().__init__(
            name="detect_objects_text_only",
            description=(
                "Detect objects in an image using GroundingDINO and return "
                "only the text detection results as a list of "
                "{\"label\": <str>, \"bbox\": [x1, y1, x2, y2]} entries "
                "where bbox coordinates are normalized to [0, 1] by image width/height."
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

    # ------------------------------------------------------------------
    # Tool entry point
    # ------------------------------------------------------------------

    def call(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Run GroundingDINO detection and return text-only results.

        Returns:
            {
                "success": True,
                "detections": [
                    {"label": "baseball bat", "bbox": [0.27, 0.70, 0.51, 0.83]},
                    ...
                ],
                "result": "<JSON string of detections>"   # convenient for LLM consumption
            }
            or on failure:
            {
                "success": False,
                "error": "<reason>"
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
            return {"success": True, "detections": [], "result": "[]"}

        # Server returns bbox already normalized to [0, 1] by the image W/H
        detections: List[Dict[str, Any]] = []
        for det in raw.get("detections", []):
            norm_bbox = [round(v, 4) for v in det["bbox"]]
            detections.append({"label": det["label"], "bbox": norm_bbox})

        result_str = json.dumps(detections, ensure_ascii=False)
        logger.info("Detected %d object(s): %s", len(detections), result_str)

        return {
            "success": True,
            "detections": detections,
            "result": result_str,
        }
