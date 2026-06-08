"""Adquisición de imágenes: captura temporal de login y escritura de ZIP.

Reemplaza de forma modular a src/faceIdentifierNoView.py (ver docs/legacy/).
"""

from .camera_capture import (
    CaptureResult,
    build_zip_from_frames,
    capture_to_zip,
)

__all__ = ["CaptureResult", "build_zip_from_frames", "capture_to_zip"]
