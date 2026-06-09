"""Entrenamiento: preparación de dataset, transfer learning y evaluación."""

from .dataset_loader import (
    DEFAULT_RATIOS,
    DatasetStats,
    prepare_dataset,
    split_files,
)
from .evaluate_model import (
    classification_report_from_cm,
    compute_confusion_matrix,
    evaluate,
    load_class_indices,
    names_ordered_by_index,
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
    "evaluate",
    "load_class_indices",
    "names_ordered_by_index",
    "compute_confusion_matrix",
    "classification_report_from_cm",
]
