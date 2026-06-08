"""Preprocesamiento facial: funciones puras para preparar el ROI para el modelo.

Pipeline: recorte con margen -> resize -> BGR->RGB -> normalización [0,1].
Todas las funciones son puras (no mutan la entrada) y operan sobre arrays NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - solo tipado
    from ..config import AppConfig

BoundingBox = Tuple[int, int, int, int]


def crop_face_with_margin(frame: np.ndarray, box: BoundingBox, margin: int) -> np.ndarray:
    """Recorta el ROI facial añadiendo ``margin`` píxeles, sin salir de la imagen.

    ``box`` es ``(x, y, w, h)``. Los límites se acotan a ``[0, ancho]`` y
    ``[0, alto]`` del frame, por lo que el recorte nunca excede la imagen.
    """
    if frame is None:
        raise ValueError("frame no puede ser None")
    if margin < 0:
        raise ValueError("margin no puede ser negativo")

    height, width = frame.shape[:2]
    x, y, w, h = box

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Caja inválida o fuera de la imagen: box={box}, frame={width}x{height}")

    return frame[y1:y2, x1:x2].copy()


def resize_face(face: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
    """Redimensiona el rostro a ``target_size`` = ``(alto, ancho)``.

    Devuelve un array con shape ``(alto, ancho, C)``.
    """
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_size debe ser positivo: {target_size}")
    # cv2.resize espera dsize=(ancho, alto).
    return cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_AREA)


def convert_bgr_to_rgb(face: np.ndarray) -> np.ndarray:
    """Convierte un rostro BGR (OpenCV) a RGB."""
    return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def normalize_pixels(face: np.ndarray) -> np.ndarray:
    """Escala los valores de píxeles de ``[0, 255]`` a ``[0, 1]`` (float32)."""
    return face.astype(np.float32) / 255.0


def preprocess_face(frame: np.ndarray, box: BoundingBox, config: "AppConfig") -> np.ndarray:
    """Pipeline completo: recorta, redimensiona, ajusta color y normaliza.

    Devuelve un tensor listo para el modelo con shape ``(H, W, 3)``.
    """
    margin = int(config.face_detection.margin_pixels)
    target_size = config.preprocessing.target_size
    color_format = getattr(config.preprocessing, "color_format", "RGB")
    normalize = getattr(config.preprocessing, "normalize_pixels", True)

    face = crop_face_with_margin(frame, box, margin)
    face = resize_face(face, target_size)

    if str(color_format).upper() == "RGB":
        face = convert_bgr_to_rgb(face)

    if normalize:
        face = normalize_pixels(face)

    if face.ndim != 3 or face.shape[2] != 3:
        raise ValueError(f"El tensor resultante no tiene shape (H, W, 3): {face.shape}")

    return face
