"""
Face Detection Tool

Local, CPU-only face detection using OpenCV Haar cascades — no server, no
GPU, no checkpoint download (the cascade ships with opencv). Returns the
standardized detection ToolResult (pixel-xyxy boxes + labels) and an
annotated visualization. Zero faces is a valid finding, not a failure.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import BOX_XYXY_PIXEL, DetectionPayload, ToolResult

logger = logging.getLogger(__name__)


class _MockFaceDetectionClient:
    """Format-faithful mock: same result shape as the real cascade path."""

    def detect(self, image_path: str, **kwargs) -> List[Dict[str, Any]]:
        return [{"bbox_xyxy": [110.0, 80.0, 240.0, 230.0],
                 "label": "face", "confidence": None}]


class FaceDetectionTool(Tool):
    """Detect human faces in an image (OpenCV Haar cascade, CPU-only)."""

    def __init__(
        self,
        use_mock: bool = True,
        min_neighbors: int = 5,
        output_dir: str = "outputs",
    ):
        super().__init__(
            name="face_detection_tool",
            description=(
                "Detect human FACES in an image and return their bounding boxes. "
                "Use this tool when the question is about people's faces: how many "
                "people are visible, where a face is located, or whether anyone is "
                "facing the camera.\n\n"
                "WHEN TO USE:\n"
                "- \"How many people/faces are in this image?\"\n"
                "- \"Where is the person's face?\"\n"
                "- \"Is anyone looking at the camera?\" (only frontal faces are detected)\n\n"
                "WHEN NOT TO USE:\n"
                "- Generic object detection (use detect_objects_tool)\n"
                "- Identity questions — this tool localizes faces, it cannot recognize them\n\n"
                "Detects FRONTAL faces only; a person facing away yields zero "
                "detections, which is a valid finding. Returns pixel-coordinate "
                "boxes and an annotated image."
            ),
        )
        self.use_mock = use_mock
        self.min_neighbors = min_neighbors
        self.output_dir = Path(output_dir)
        self._client = _MockFaceDetectionClient() if use_mock else None
        self._cascade = None
        if use_mock:
            logger.info("Using mock face detection service")
        else:
            logger.info("Using local OpenCV Haar cascade for face detection")

    def _ensure_cascade(self):
        if self._cascade is None:
            import cv2
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self._cascade.empty():
                raise RuntimeError("failed to load haarcascade_frontalface_default.xml")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image.",
                },
                "min_neighbors": {
                    "type": "integer",
                    "description": (
                        "Detector strictness (higher = fewer false positives, "
                        "may miss small faces). Default 5."
                    ),
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["image_path"],
        }

    def _detect_real(self, image_path: str, min_neighbors: int) -> List[Dict[str, Any]]:
        import cv2
        self._ensure_cascade()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"could not decode image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(24, 24))
        return [{"bbox_xyxy": [float(x), float(y), float(x + w), float(y + h)],
                 "label": "face", "confidence": None}
                for (x, y, w, h) in faces]

    def _annotate(self, image_path: str, detections: List[Dict[str, Any]]) -> Optional[str]:
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det["bbox_xyxy"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, det["label"], (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out = str(self.output_dir / f"{Path(image_path).stem}_faces.jpg")
            cv2.imwrite(out, img)
            return out
        except Exception as e:
            logger.warning(f"face annotation failed: {e}")
            return None

    def call(self, image_path: str, min_neighbors: Optional[int] = None) -> Dict[str, Any]:
        try:
            image_path = str(image_path)
            p = Path(image_path)
            if not p.exists():
                return {"success": False, "error": f"Image not found: {image_path}"}

            neighbors = self.min_neighbors if min_neighbors is None else int(min_neighbors)
            if self.use_mock:
                detections = self._client.detect(image_path, min_neighbors=neighbors)
            else:
                detections = self._detect_real(image_path, neighbors)

            from PIL import Image
            with Image.open(image_path) as im:
                width, height = im.size

            output_path = self._annotate(image_path, detections) if detections else None
            summary = (
                f"Detected {len(detections)} face(s) in {p.name}."
                if detections else
                f"No frontal faces detected in {p.name} (people facing away are "
                "not detected — this is a valid finding)."
            )

            payload = DetectionPayload(
                boxes=[d["bbox_xyxy"] for d in detections],
                labels=[d["label"] for d in detections],
                box_format=BOX_XYXY_PIXEL,
                image_width=width,
                image_height=height,
            )
            return ToolResult(
                success=True,
                payload=payload,
                description=summary,
                output_path=output_path,
                result={
                    "image_path": image_path,
                    "num_detections": len(detections),
                    "detections": detections,
                },
            )
        except Exception as e:
            logger.exception("FaceDetectionTool error")
            return {"success": False, "error": str(e)}
