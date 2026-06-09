"""Pruebas de batch_predictor (sin TensorFlow).

Se prueba el listado de rostros en el ZIP, el preprocesamiento y la conversión
a registros. La inferencia completa requiere un modelo entrenado (no aquí).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rp_face_login.inference.batch_predictor import (
    list_face_entries,
    predictions_to_records,
    prepare_image,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "faces" / "elioth" / "luz_frontal" / "elioth_0001.jpg"


def _make_zip(tmp_path, entries):
    path = tmp_path / "login.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_list_face_entries_filters_and_sorts(tmp_path):
    path = _make_zip(
        tmp_path,
        {
            "faces/face_0002.jpg": b"x",
            "faces/face_0001.jpg": b"x",
            "faces/notes.txt": b"x",
            "frames_anotados/frame_0001.jpg": b"x",
            "faces/": b"",
        },
    )
    with zipfile.ZipFile(path) as zf:
        entries = list_face_entries(zf, "faces")
    assert entries == ["faces/face_0001.jpg", "faces/face_0002.jpg"]


def test_predictions_to_records_structure():
    probs = np.array([[0.91, 0.09], [0.2, 0.8]])
    records = predictions_to_records(["face_0001.jpg", "face_0002.jpg"], probs, ["elioth", "emmanuel"])
    assert records[0] == {"frame": "face_0001.jpg", "elioth": pytest.approx(0.91), "emmanuel": pytest.approx(0.09)}
    assert records[1]["emmanuel"] == pytest.approx(0.8)
    # softmax por frame suma ~1
    for rec in records:
        assert rec["elioth"] + rec["emmanuel"] == pytest.approx(1.0)


def test_prepare_image_shape_and_range():
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((50, 70, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # rojo en BGR
    out = prepare_image(img, (224, 224))
    assert out.shape == (224, 224, 3)
    # BGR->RGB: el canal rojo (índice 0 en RGB) debe estar saturado
    assert out[0, 0, 0] == 255


@pytest.mark.skipif(not SAMPLE.exists(), reason="imagen de dataset no disponible")
def test_decode_real_face_from_zip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    raw = SAMPLE.read_bytes()
    path = _make_zip(tmp_path, {"faces/face_0001.jpg": raw})
    with zipfile.ZipFile(path) as zf:
        names = list_face_entries(zf, "faces")
        buf = np.frombuffer(zf.read(names[0]), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img is not None
    prepared = prepare_image(img, (224, 224))
    assert prepared.shape == (224, 224, 3)


def test_zip_not_modified_by_reading(tmp_path):
    path = _make_zip(tmp_path, {"faces/face_0001.jpg": b"data", "faces/face_0002.jpg": b"data2"})
    before = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        _ = list_face_entries(zf, "faces")
        _ = zf.read("faces/face_0001.jpg")
    assert path.read_bytes() == before
