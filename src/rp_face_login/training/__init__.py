"""Entrenamiento: preparación de dataset, transfer learning y evaluación."""

from .dataset_loader import (
    DEFAULT_RATIOS,
    DatasetStats,
    prepare_dataset,
    split_files,
)
from .train_model import (
    SUPPORTED_BACKBONES,
    build_model,
    class_indices_from_names,
    train,
)

__all__ = [
    "DEFAULT_RATIOS",
    "DatasetStats",
    "prepare_dataset",
    "split_files",
    "SUPPORTED_BACKBONES",
    "build_model",
    "class_indices_from_names",
    "train",
]
