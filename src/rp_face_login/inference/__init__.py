"""Inferencia: carga de modelo, predicción por batch y agregación temporal."""

from .batch_predictor import (
    list_face_entries,
    predict_zip,
    predictions_to_records,
    prepare_image,
)

__all__ = [
    "list_face_entries",
    "predict_zip",
    "predictions_to_records",
    "prepare_image",
]
