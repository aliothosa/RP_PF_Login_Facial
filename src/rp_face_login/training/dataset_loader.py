"""Preparación del dataset procesado para entrenamiento.

A partir de carpetas de imágenes por clase (p. ej. ``data/raw/elioth`` o
``data/faces/elioth/<condicion>``), genera un dataset dividido en
``train/val/test`` con rostros detectados, recortados y redimensionados.

Cada imagen de salida se guarda como JPEG **uint8** (recorte + resize al tamaño
objetivo). La normalización a ``[0, 1]`` y la conversión a RGB se aplican al
cargar para el modelo (no se persisten en disco, pues degradarían el JPEG).

Privacidad: el dataset real NO se versiona (``data/`` y ``dataset/`` están en
``.gitignore``).
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from ..config import AppConfig
from ..vision.face_detector import FaceDetector
from ..vision.preprocessing import crop_face_with_margin, resize_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS: Tuple[str, str, str] = ("train", "val", "test")
DEFAULT_RATIOS: Tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42


@dataclass
class ClassStats:
    accepted: int = 0
    discarded: int = 0
    per_split: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SPLITS})


@dataclass
class DatasetStats:
    seed: int
    ratios: Tuple[float, float, float]
    per_class: Dict[str, ClassStats] = field(default_factory=dict)

    @property
    def total_accepted(self) -> int:
        return sum(c.accepted for c in self.per_class.values())

    @property
    def total_discarded(self) -> int:
        return sum(c.discarded for c in self.per_class.values())

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "ratios": list(self.ratios),
            "total_accepted": self.total_accepted,
            "total_discarded": self.total_discarded,
            "per_class": {name: asdict(stats) for name, stats in self.per_class.items()},
        }


def split_files(
    files: Sequence[Path],
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[Path]]:
    """Divide una lista de archivos en train/val/test de forma reproducible.

    Ordena primero (orden estable independiente del sistema de archivos) y luego
    baraja con una semilla fija.
    """
    train_r, val_r, test_r = ratios
    if min(ratios) < 0:
        raise ValueError("Las proporciones no pueden ser negativas.")
    if abs((train_r + val_r + test_r) - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0 (se obtuvo {sum(ratios)}).")

    ordered = sorted(files, key=lambda p: str(p))
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }


def _list_class_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    config: AppConfig,
    *,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    classes: Optional[Sequence[str]] = None,
    detector: Optional[FaceDetector] = None,
    clean: bool = True,
) -> DatasetStats:
    """Genera el dataset procesado train/val/test desde carpetas por clase."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de entrada: {raw_dir}")

    classes = list(classes) if classes is not None else list(config.model.classes)
    detector = detector or FaceDetector(config.face_detection)
    margin = int(config.face_detection.margin_pixels)
    target_size = config.preprocessing.target_size

    if clean:
        for split in SPLITS:
            for cls in classes:
                shutil.rmtree(output_dir / split / cls, ignore_errors=True)

    stats = DatasetStats(seed=seed, ratios=ratios)

    for cls in classes:
        cls_stats = ClassStats()
        stats.per_class[cls] = cls_stats

        class_dir = raw_dir / cls
        if not class_dir.exists():
            print(f"[dataset] Aviso: no existe la carpeta de clase '{class_dir}', se omite.")
            continue

        files = _list_class_images(class_dir)
        if not files:
            print(f"[dataset] Aviso: sin imágenes en '{class_dir}'.")
            continue

        split_map = split_files(files, ratios=ratios, seed=seed)

        for split, paths in split_map.items():
            out_dir = output_dir / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            saved = 0
            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    cls_stats.discarded += 1
                    continue

                box = detector.select_largest_face(detector.detect_faces(image))
                if box is None:
                    cls_stats.discarded += 1
                    continue

                face = crop_face_with_margin(image, box, margin)
                face = resize_face(face, target_size)

                saved += 1
                cls_stats.accepted += 1
                cls_stats.per_split[split] += 1
                cv2.imwrite(str(out_dir / f"{cls}_{saved:04d}.jpg"), face)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    _report(stats, output_dir)
    return stats


def _report(stats: DatasetStats, output_dir: Path) -> None:
    print("Preparación de dataset finalizada.")
    print(f"  Semilla        : {stats.seed}")
    print(f"  Proporciones   : train={stats.ratios[0]} val={stats.ratios[1]} test={stats.ratios[2]}")
    for cls, cs in stats.per_class.items():
        splits = " ".join(f"{s}={cs.per_split[s]}" for s in SPLITS)
        print(f"  [{cls}] aceptadas={cs.accepted} descartadas={cs.discarded} ({splits})")
    print(f"  Total aceptadas   : {stats.total_accepted}")
    print(f"  Total descartadas : {stats.total_discarded}")
    print(f"  Salida            : {output_dir}")
