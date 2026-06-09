"""Pruebas de la política de decisión (aceptación / rechazo a guest)."""

import pytest

from rp_face_login.config import DecisionConfig
from rp_face_login.decision.decision_policy import (
    REASON_INSUFFICIENT_FRAMES,
    REASON_LOW_CONFIDENCE,
    REASON_MARGIN_BELOW_THRESHOLD,
    decide,
)


@pytest.fixture
def cfg():
    # Defaults del proyecto: 30 frames, 0.80 confianza, 0.25 margen, fallback guest.
    return DecisionConfig()


def _agg(valid_frames, best_user, best_score, second_user, second_score):
    return {
        "valid_frames": valid_frames,
        "best_user": best_user,
        "best_score": best_score,
        "second_user": second_user,
        "second_score": second_score,
        "margin": best_score - second_score,
    }


def test_accepts_elioth(cfg):
    result = decide(_agg(120, "elioth", 0.95, "emmanuel", 0.05), cfg)
    assert result["accepted"] is True
    assert result["selected_user"] == "elioth"
    assert result["reason"] == "accepted"


def test_accepts_emmanuel(cfg):
    result = decide(_agg(80, "emmanuel", 0.88, "elioth", 0.12), cfg)
    assert result["accepted"] is True
    assert result["selected_user"] == "emmanuel"


def test_rejects_for_few_frames(cfg):
    result = decide(_agg(10, "elioth", 0.99, "emmanuel", 0.01), cfg)
    assert result["accepted"] is False
    assert result["selected_user"] == "guest"
    assert result["reason"] == REASON_INSUFFICIENT_FRAMES


def test_rejects_for_low_score(cfg):
    result = decide(_agg(100, "elioth", 0.70, "emmanuel", 0.30), cfg)
    assert result["accepted"] is False
    assert result["selected_user"] == "guest"
    assert result["reason"] == REASON_LOW_CONFIDENCE


def test_rejects_for_low_margin(cfg):
    # score alto (>=0.80) pero margen 0.16 < 0.25
    result = decide(_agg(120, "elioth", 0.85, "emmanuel", 0.69), cfg)
    assert result["accepted"] is False
    assert result["selected_user"] == "guest"
    assert result["reason"] == REASON_MARGIN_BELOW_THRESHOLD


def test_matches_example_output(cfg):
    result = decide(_agg(120, "elioth", 0.71, "emmanuel", 0.55), cfg)
    assert result["selected_user"] == "guest"
    assert result["accepted"] is False
    # 0.71 < 0.80 -> primero falla la confianza.
    assert result["reason"] == REASON_LOW_CONFIDENCE
    assert result["valid_frames"] == 120
    assert result["best_user"] == "elioth"
    assert result["best_score"] == pytest.approx(0.71)
    assert result["second_user"] == "emmanuel"
    assert result["second_score"] == pytest.approx(0.55)
    assert result["margin"] == pytest.approx(0.16)


def test_frames_checked_before_score(cfg):
    # Falla por frames Y por score; debe reportar frames (primera condición).
    result = decide(_agg(5, "elioth", 0.50, "emmanuel", 0.50), cfg)
    assert result["reason"] == REASON_INSUFFICIENT_FRAMES


def test_boundary_values_are_accepted(cfg):
    # Exactamente en los umbrales: >= acepta.
    result = decide(_agg(30, "elioth", 0.80, "emmanuel", 0.55), cfg)
    assert result["accepted"] is True
    assert result["selected_user"] == "elioth"
