"""Inferencia: carga de modelo, predicción por batch y agregación temporal."""

from .batch_predictor import (
    list_face_entries,
    predict_zip,
    predictions_to_records,
    prepare_image,
)
from .temporal_aggregation import aggregate_predictions

__all__ = [
    "list_face_entries",
    "predict_zip",
    "predictions_to_records",
    "prepare_image",
    "aggregate_predictions",
]
