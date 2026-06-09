"""Pruebas de la agregación temporal (lógica pura)."""

import pytest

from rp_face_login.inference.temporal_aggregation import aggregate_predictions


def test_empty_predictions_raises():
    with pytest.raises(ValueError):
        aggregate_predictions([])


def test_example_average_and_valid_frames():
    preds = [
        {"elioth": 0.91, "emmanuel": 0.09},
        {"elioth": 0.87, "emmanuel": 0.13},
    ]
    result = aggregate_predictions(preds)
    assert result["valid_frames"] == 2
    assert result["avg_scores"]["elioth"] == pytest.approx(0.89)
    assert result["avg_scores"]["emmanuel"] == pytest.approx(0.11)


def test_best_second_and_margin():
    preds = [
        {"elioth": 0.91, "emmanuel": 0.09},
        {"elioth": 0.87, "emmanuel": 0.13},
    ]
    result = aggregate_predictions(preds)
    assert result["best_user"] == "elioth"
    assert result["best_score"] == pytest.approx(0.89)
    assert result["second_user"] == "emmanuel"
    assert result["second_score"] == pytest.approx(0.11)
    assert result["margin"] == pytest.approx(0.78)


def test_ordering_when_emmanuel_wins():
    preds = [
        {"elioth": 0.2, "emmanuel": 0.8},
        {"elioth": 0.4, "emmanuel": 0.6},
    ]
    result = aggregate_predictions(preds)
    assert result["best_user"] == "emmanuel"
    assert result["second_user"] == "elioth"
    # emmanuel=0.7, elioth=0.3 -> margin=0.4
    assert result["margin"] == pytest.approx(0.4)


def test_ignores_frame_key():
    preds = [
        {"frame": "face_0001.jpg", "elioth": 0.6, "emmanuel": 0.4},
        {"frame": "face_0002.jpg", "elioth": 0.8, "emmanuel": 0.2},
    ]
    result = aggregate_predictions(preds)
    assert set(result["avg_scores"].keys()) == {"elioth", "emmanuel"}
    assert result["avg_scores"]["elioth"] == pytest.approx(0.7)


def test_missing_class_defaults_to_zero():
    preds = [
        {"elioth": 1.0},          # falta emmanuel -> 0.0
        {"elioth": 0.0, "emmanuel": 1.0},
    ]
    result = aggregate_predictions(preds, class_names=["elioth", "emmanuel"])
    assert result["avg_scores"]["elioth"] == pytest.approx(0.5)
    assert result["avg_scores"]["emmanuel"] == pytest.approx(0.5)


def test_single_class_has_no_second():
    result = aggregate_predictions([{"elioth": 0.9}], class_names=["elioth"])
    assert result["best_user"] == "elioth"
    assert result["second_user"] is None
    assert result["second_score"] == 0.0
    assert result["margin"] == pytest.approx(0.9)


def test_tie_is_deterministic_by_name():
    preds = [{"elioth": 0.5, "emmanuel": 0.5}]
    result = aggregate_predictions(preds)
    # Empate -> desempate alfabético: elioth primero.
    assert result["best_user"] == "elioth"
    assert result["second_user"] == "emmanuel"
    assert result["margin"] == pytest.approx(0.0)
