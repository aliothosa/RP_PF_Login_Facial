"""Pruebas de preprocesamiento facial (funciones puras).

Se enfoca en el recorte con margen en bordes de la imagen, además de
normalización, conversión de color, resize y el pipeline completo.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from rp_face_login.vision.preprocessing import (
    convert_bgr_to_rgb,
    crop_face_with_margin,
    normalize_pixels,
    preprocess_face,
    resize_face,
)


def _frame(h=100, w=120, c=3):
    # Frame con valores deterministas para verificar recortes.
    return np.arange(h * w * c, dtype=np.uint8).reshape(h, w, c)


# --- crop_face_with_margin: bordes de la imagen ---

def test_crop_center_with_margin():
    frame = _frame(100, 100)
    crop = crop_face_with_margin(frame, (40, 40, 20, 20), margin=10)
    # x:30..70, y:30..70 -> 40x40
    assert crop.shape == (40, 40, 3)


def test_crop_top_left_corner_clamped():
    frame = _frame(100, 100)
    # Caja pegada a la esquina; el margen se saldría a coords negativas.
    crop = crop_face_with_margin(frame, (0, 0, 20, 20), margin=15)
    # x1=0, y1=0, x2=35, y2=35
    assert crop.shape == (35, 35, 3)


def test_crop_bottom_right_corner_clamped():
    frame = _frame(100, 100)
    crop = crop_face_with_margin(frame, (80, 80, 20, 20), margin=15)
    # x:65..100 (clamp), y:65..100 -> 35x35
    assert crop.shape == (35, 35, 3)


def test_crop_box_larger_than_frame_is_clamped():
    frame = _frame(50, 60)  # alto=50, ancho=60
    crop = crop_face_with_margin(frame, (10, 10, 200, 200), margin=20)
    # x1=0, y1=0, x2=min(60, 230)=60, y2=min(50, 230)=50 -> frame completo.
    assert crop.shape == (50, 60, 3)


def test_crop_zero_margin_exact_box():
    frame = _frame(100, 100)
    crop = crop_face_with_margin(frame, (10, 20, 30, 40), margin=0)
    assert crop.shape == (40, 30, 3)


def test_crop_does_not_mutate_input():
    frame = _frame(100, 100)
    original = frame.copy()
    _ = crop_face_with_margin(frame, (10, 10, 20, 20), margin=5)
    assert np.array_equal(frame, original)


def test_crop_negative_margin_raises():
    frame = _frame(50, 50)
    with pytest.raises(ValueError):
        crop_face_with_margin(frame, (10, 10, 10, 10), margin=-1)


# --- normalize_pixels ---

def test_normalize_to_unit_range():
    face = np.array([[0, 128, 255]], dtype=np.uint8)
    out = normalize_pixels(face)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[0, 2], 1.0)


# --- convert_bgr_to_rgb ---

def test_convert_bgr_to_rgb_swaps_channels():
    bgr = np.zeros((1, 1, 3), dtype=np.uint8)
    bgr[0, 0] = [10, 20, 30]  # B=10, G=20, R=30
    rgb = convert_bgr_to_rgb(bgr)
    assert list(rgb[0, 0]) == [30, 20, 10]


# --- resize_face ---

def test_resize_face_shape():
    face = _frame(50, 70)
    out = resize_face(face, (224, 224))
    assert out.shape == (224, 224, 3)


def test_resize_face_non_square_orientation():
    face = _frame(50, 70)
    out = resize_face(face, (32, 64))  # (alto, ancho)
    assert out.shape == (32, 64, 3)


# --- preprocess_face (pipeline completo) ---

def _config(margin=10, target=(224, 224), color="RGB", normalize=True):
    return SimpleNamespace(
        face_detection=SimpleNamespace(margin_pixels=margin),
        preprocessing=SimpleNamespace(
            target_size=list(target),
            color_format=color,
            normalize_pixels=normalize,
        ),
    )


def test_preprocess_face_returns_model_ready_tensor():
    frame = _frame(200, 200)
    out = preprocess_face(frame, (50, 50, 60, 60), _config())
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_face_without_normalization_keeps_uint_range():
    frame = _frame(200, 200)
    out = preprocess_face(frame, (50, 50, 60, 60), _config(normalize=False))
    assert out.shape == (224, 224, 3)
    assert out.max() > 1.0  # sigue en escala [0,255]
