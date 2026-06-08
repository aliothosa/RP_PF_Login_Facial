"""Pruebas de la captura temporal (sin cámara real).

Se prueba ``build_zip_from_frames`` con frames sintéticos y, si hay una imagen
real del dataset disponible, también el caso con rostros detectados.
"""

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cv2")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rp_face_login.acquisition.camera_capture import build_zip_from_frames  # noqa: E402
from rp_face_login.config import load_config  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
SAMPLE_FACE = (
    Path(__file__).resolve().parents[1]
    / "data" / "faces" / "elioth" / "luz_frontal" / "elioth_0001.jpg"
)


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


def test_blank_frames_produce_empty_but_single_zip(config, tmp_path):
    frames = [np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(5)]
    result = build_zip_from_frames(frames, config, "elioth", tmp_path)

    assert result.frames_read == 5
    assert result.valid_frames == 0
    # Único artefacto de salida: el ZIP (no hay carpeta descomprimida).
    artifacts = list(tmp_path.iterdir())
    assert artifacts == [result.zip_path]
    assert result.zip_path.suffix == ".zip"

    with zipfile.ZipFile(result.zip_path) as zf:
        assert zf.namelist() == []


def test_none_frames_count_as_read_but_invalid(config, tmp_path):
    result = build_zip_from_frames([None, None], config, "x", tmp_path)
    assert result.frames_read == 2
    assert result.valid_frames == 0


def test_no_metadata_csv_in_zip(config, tmp_path):
    frames = [np.zeros((150, 150, 3), dtype=np.uint8)]
    result = build_zip_from_frames(frames, config, "x", tmp_path)
    with zipfile.ZipFile(result.zip_path) as zf:
        assert not any(n.endswith("metadata.csv") for n in zf.namelist())


@pytest.mark.skipif(not SAMPLE_FACE.exists(), reason="imagen de dataset no disponible")
def test_real_faces_are_stored_with_padded_names(config, tmp_path):
    img = cv2.imread(str(SAMPLE_FACE))
    frames = [img.copy() for _ in range(3)]
    result = build_zip_from_frames(frames, config, "elioth", tmp_path, debug_annotated=True)

    assert result.valid_frames >= 1
    with zipfile.ZipFile(result.zip_path) as zf:
        names = zf.namelist()
    assert "faces/face_0001.jpg" in names
    assert any(n.startswith("frames_anotados/frame_0001.jpg") for n in names)
    assert all(n.startswith(("faces/", "frames_anotados/")) for n in names)
