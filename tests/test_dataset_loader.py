"""Pruebas de dataset_loader.

``split_files`` se prueba de forma pura (sin imágenes). La preparación completa
usa imágenes reales del dataset si están disponibles (se omite si no).
"""

from pathlib import Path

import pytest

from rp_face_login.training.dataset_loader import DEFAULT_RATIOS, split_files

ROOT = Path(__file__).resolve().parents[1]


# --- split_files (lógica pura, reproducible) ---

def _fake_files(n):
    return [Path(f"{i:03d}.jpg") for i in range(n)]


def test_split_sizes_70_15_15():
    splits = split_files(_fake_files(100), DEFAULT_RATIOS, seed=42)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 15
    assert len(splits["test"]) == 15


def test_split_is_a_partition():
    files = _fake_files(57)
    splits = split_files(files, DEFAULT_RATIOS, seed=7)
    union = splits["train"] + splits["val"] + splits["test"]
    assert sorted(union, key=str) == sorted(files, key=str)
    assert len(union) == len(set(union)) == 57


def test_split_reproducible_with_same_seed():
    a = split_files(_fake_files(40), DEFAULT_RATIOS, seed=123)
    b = split_files(_fake_files(40), DEFAULT_RATIOS, seed=123)
    assert a == b


def test_split_changes_with_different_seed():
    a = split_files(_fake_files(40), DEFAULT_RATIOS, seed=1)
    b = split_files(_fake_files(40), DEFAULT_RATIOS, seed=2)
    assert a["train"] != b["train"]


def test_split_ratios_must_sum_one():
    with pytest.raises(ValueError):
        split_files(_fake_files(10), (0.7, 0.2, 0.2), seed=0)


def test_split_small_set_remainder_goes_to_test():
    splits = split_files(_fake_files(10), DEFAULT_RATIOS, seed=0)
    # int(10*0.7)=7, int(10*0.15)=1, resto -> test=2
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (7, 1, 2)


# --- prepare_dataset (integración con imágenes reales) ---

_ELIOTH = ROOT / "data" / "faces" / "elioth" / "luz_frontal" / "elioth_0001.jpg"
_EMMANUEL = ROOT / "data" / "faces" / "emmanuel" / "luz_frontal" / "emmanuel_0001.jpg"


@pytest.mark.skipif(
    not (_ELIOTH.exists() and _EMMANUEL.exists()),
    reason="imágenes de dataset no disponibles",
)
def test_prepare_dataset_creates_split_structure(tmp_path):
    import shutil

    cv2 = pytest.importorskip("cv2")  # noqa: F841
    from rp_face_login.config import load_config
    from rp_face_login.training.dataset_loader import prepare_dataset

    # Construye un raw dir pequeño con varias copias por clase.
    raw = tmp_path / "raw"
    for cls, sample in [("elioth", _ELIOTH), ("emmanuel", _EMMANUEL)]:
        (raw / cls).mkdir(parents=True)
        for i in range(8):
            shutil.copy(sample, raw / cls / f"{cls}_{i:02d}.jpg")

    cfg = load_config(ROOT / "configs" / "default.yaml")
    out = tmp_path / "processed"
    stats = prepare_dataset(raw, out, cfg, seed=42)

    for split in ("train", "val", "test"):
        for cls in ("elioth", "emmanuel"):
            assert (out / split / cls).is_dir()

    assert stats.total_accepted > 0
    assert (out / "dataset_stats.json").exists()
    # Cada rostro guardado debe tener el tamaño objetivo.
    target_h, target_w = cfg.preprocessing.target_size
    any_face = next((out / "train" / "elioth").glob("*.jpg"), None)
    if any_face is not None:
        img = cv2.imread(str(any_face))
        assert img.shape[0] == target_h and img.shape[1] == target_w
