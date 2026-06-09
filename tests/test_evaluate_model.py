"""Pruebas de evaluate_model (métricas puras, sin TensorFlow)."""

import json

import numpy as np
import pytest

from rp_face_login.training.evaluate_model import (
    classification_report_from_cm,
    compute_confusion_matrix,
    load_class_indices,
    names_ordered_by_index,
)


def test_names_ordered_by_index():
    assert names_ordered_by_index({"emmanuel": 1, "elioth": 0}) == ["elioth", "emmanuel"]


def test_load_class_indices_roundtrip(tmp_path):
    path = tmp_path / "class_indices.json"
    path.write_text(json.dumps({"elioth": 0, "emmanuel": 1}), encoding="utf-8")
    assert load_class_indices(path) == {"elioth": 0, "emmanuel": 1}


def test_load_class_indices_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_class_indices(tmp_path / "nope.json")


def test_confusion_matrix_perfect():
    y = [0, 0, 1, 1]
    cm = compute_confusion_matrix(y, y, 2)
    assert cm.tolist() == [[2, 0], [0, 2]]


def test_confusion_matrix_with_errors():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    cm = compute_confusion_matrix(y_true, y_pred, 2)
    # fila=verdadero, col=predicho
    assert cm.tolist() == [[1, 1], [1, 1]]


def test_report_perfect_classification():
    cm = np.array([[2, 0], [0, 2]])
    report = classification_report_from_cm(cm, ["elioth", "emmanuel"])
    assert report["accuracy"] == 1.0
    assert report["per_class"]["elioth"]["precision"] == 1.0
    assert report["per_class"]["elioth"]["recall"] == 1.0
    assert report["per_class"]["elioth"]["f1_score"] == 1.0
    assert report["per_class"]["elioth"]["support"] == 2


def test_report_metrics_values():
    # elioth: TP=8, FN=2 (support 10); emmanuel: TP=5, FP=2
    cm = np.array([[8, 2], [0, 5]])
    report = classification_report_from_cm(cm, ["elioth", "emmanuel"])
    # accuracy = (8+5)/15
    assert report["accuracy"] == pytest.approx(13 / 15)
    # elioth precision = 8/8 = 1.0 ; recall = 8/10 = 0.8
    assert report["per_class"]["elioth"]["precision"] == pytest.approx(1.0)
    assert report["per_class"]["elioth"]["recall"] == pytest.approx(0.8)
    # emmanuel precision = 5/7 ; recall = 5/5 = 1.0
    assert report["per_class"]["emmanuel"]["precision"] == pytest.approx(5 / 7)
    assert report["per_class"]["emmanuel"]["recall"] == pytest.approx(1.0)


def test_report_handles_empty_class():
    # Clase 'emmanuel' sin soporte ni predicciones -> métricas 0, sin división por cero.
    cm = np.array([[3, 0], [0, 0]])
    report = classification_report_from_cm(cm, ["elioth", "emmanuel"])
    assert report["per_class"]["emmanuel"]["precision"] == 0.0
    assert report["per_class"]["emmanuel"]["recall"] == 0.0
    assert report["per_class"]["emmanuel"]["support"] == 0
