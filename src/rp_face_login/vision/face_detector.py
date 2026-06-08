"""Detección facial con Haar Cascade.

Centraliza la lógica de detección que estaba duplicada en los scripts legacy
(``faceIdentifierNoView.py``, ``faceIdentifierView.py``, ``face_extractor.py``).

La carga del clasificador funciona tanto en ejecución normal como empaquetado
con PyInstaller (el XML se incluye con ``--add-data ...:cv2/data`` y se localiza
vía ``sys._MEIPASS``).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import cv2

if TYPE_CHECKING:  # pragma: no cover - solo para tipado
    from ..config import FaceDetectionConfig

# Bounding box (x, y, width, height).
BoundingBox = Tuple[int, int, int, int]

DEFAULT_CASCADE = "haarcascade_frontalface_default.xml"
DEFAULT_SCALE_FACTOR = 1.1
DEFAULT_MIN_NEIGHBORS = 6
DEFAULT_MIN_SIZE: Tuple[int, int] = (100, 100)


def resolve_cascade_path(filename: str = DEFAULT_CASCADE) -> str:
    """Localiza el XML del Haar Cascade.

    Prioriza el bundle de PyInstaller (``sys._MEIPASS/cv2/data``) cuando la app
    está congelada y, en su defecto, usa ``cv2.data.haarcascades``.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = os.path.join(sys._MEIPASS, "cv2", "data", filename)  # type: ignore[attr-defined]
        if os.path.exists(bundled):
            return bundled

    return os.path.join(cv2.data.haarcascades, filename)


class FaceDetector:
    """Detector facial basado en Haar Cascade de OpenCV.

    Ejemplo::

        detector = FaceDetector(config.face_detection)
        boxes = detector.detect_faces(frame)
        box = detector.select_largest_face(boxes)
    """

    def __init__(
        self,
        config: "Optional[FaceDetectionConfig]" = None,
        *,
        cascade_filename: Optional[str] = None,
        scale_factor: Optional[float] = None,
        min_neighbors: Optional[int] = None,
        min_size: Optional[Sequence[int]] = None,
    ) -> None:
        self.cascade_filename = (
            cascade_filename
            or getattr(config, "haar_cascade", None)
            or DEFAULT_CASCADE
        )
        self.scale_factor = (
            scale_factor
            if scale_factor is not None
            else getattr(config, "scale_factor", DEFAULT_SCALE_FACTOR)
        )
        self.min_neighbors = (
            min_neighbors
            if min_neighbors is not None
            else getattr(config, "min_neighbors", DEFAULT_MIN_NEIGHBORS)
        )
        raw_min_size = (
            min_size
            if min_size is not None
            else getattr(config, "min_size", DEFAULT_MIN_SIZE)
        )
        self.min_size: Tuple[int, int] = (int(raw_min_size[0]), int(raw_min_size[1]))

        self.cascade_path = resolve_cascade_path(self.cascade_filename)
        self._classifier = cv2.CascadeClassifier(self.cascade_path)
        if self._classifier.empty():
            raise RuntimeError(
                f"No se pudo cargar el Haar Cascade de OpenCV: {self.cascade_path}"
            )

    def detect_faces(self, frame) -> List[BoundingBox]:
        """Detecta rostros en un frame BGR (o ya en escala de grises).

        Devuelve una lista de bounding boxes ``(x, y, w, h)``; lista vacía si no
        se detecta ningún rostro.
        """
        if frame is None:
            raise ValueError("frame no puede ser None")

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    @staticmethod
    def select_largest_face(
        boxes: Optional[Sequence[BoundingBox]],
    ) -> Optional[BoundingBox]:
        """Devuelve la caja con mayor área ``width * height``.

        Devuelve ``None`` si ``boxes`` es ``None`` o está vacío.
        """
        if boxes is None or len(boxes) == 0:
            return None
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        return (int(x), int(y), int(w), int(h))
