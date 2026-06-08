"""Visión por computadora: detección facial, preprocesamiento y calidad.

Centraliza la lógica Haar Cascade duplicada en los scripts legacy.
"""

from .face_detector import BoundingBox, FaceDetector, resolve_cascade_path
from .preprocessing import (
    convert_bgr_to_rgb,
    crop_face_with_margin,
    normalize_pixels,
    preprocess_face,
    resize_face,
)

__all__ = [
    "BoundingBox",
    "FaceDetector",
    "resolve_cascade_path",
    "crop_face_with_margin",
    "resize_face",
    "convert_bgr_to_rgb",
    "normalize_pixels",
    "preprocess_face",
]
