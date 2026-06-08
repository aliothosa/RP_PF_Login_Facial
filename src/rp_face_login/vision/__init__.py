"""Visión por computadora: detección facial, preprocesamiento y calidad.

Centraliza la lógica Haar Cascade duplicada en los scripts legacy.
"""

from .face_detector import BoundingBox, FaceDetector, resolve_cascade_path

__all__ = ["BoundingBox", "FaceDetector", "resolve_cascade_path"]
