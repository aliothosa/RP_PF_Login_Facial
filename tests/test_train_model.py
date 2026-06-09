"""Pruebas de train_model.

Los helpers puros se prueban siempre. La construcción del modelo requiere
TensorFlow y se omite si no está instalado (entorno sin extra [ml]).
"""

import json

import pytest

from rp_face_login.training.train_model import (
    SUPPORTED_BACKBONES,
    class_indices_from_names,
    save_json,
)


def test_class_indices_from_names():
    assert class_indices_from_names(["elioth", "emmanuel"]) == {"elioth": 0, "emmanuel": 1}


def test_class_indices_preserves_order():
    assert class_indices_from_names(["b", "a", "c"]) == {"b": 0, "a": 1, "c": 2}


def test_save_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    save_json(path, {"acc": [0.1, 0.9], "name": "café"})
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"acc": [0.1, 0.9], "name": "café"}


def test_supported_backbones():
    assert "MobileNetV2" in SUPPORTED_BACKBONES
    assert "EfficientNetB0" in SUPPORTED_BACKBONES


# --- Construcción del modelo (requiere TensorFlow) ---

def test_build_model_mobilenet_output_shape():
    pytest.importorskip("tensorflow")
    from rp_face_login.training.train_model import build_model

    model = build_model((224, 224, 3), 2, backbone="MobileNetV2", weights=None)
    assert model.output_shape == (None, 2)
    # El backbone debe estar congelado inicialmente.
    base = next(layer for layer in model.layers if "mobilenet" in layer.name.lower())
    assert base.trainable is False


def test_build_model_invalid_backbone_raises():
    pytest.importorskip("tensorflow")
    from rp_face_login.training.train_model import build_model

    with pytest.raises(ValueError):
        build_model((224, 224, 3), 2, backbone="ResNet999", weights=None)
