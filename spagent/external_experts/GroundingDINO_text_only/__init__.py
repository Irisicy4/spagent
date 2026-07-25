"""
GroundingDINO Text-Only Tool

Returns only text-format detection results (label + xyxy pixel coordinates),
without saving annotated images.
"""

from .grounding_dino_text_only_tool import GroundingDINOTextOnlyTool

__all__ = ["GroundingDINOTextOnlyTool"]
