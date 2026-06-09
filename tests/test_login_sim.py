"""Pruebas de la orquestación de login-sim (parte pura, sin cámara ni TF)."""

import json

import pytest

from rp_face_login.config import load_config
from rp_face_login.login_sim import decide_from_records

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


def test_no_records_rejects_to_guest(config):
    decision = decide_from_records([], config)
    assert decision["selected_user"] == "guest"
    assert decision["accepted"] is False
    assert decision["reason"] == "insufficient_frames"
    assert decision["valid_frames"] == 0


def test_strong_elioth_is_accepted(config):
    records = [{"frame": f"face_{i:04d}.jpg", "elioth": 0.95, "emmanuel": 0.05} for i in range(60)]
    decision = decide_from_records(records, config)
    assert decision["selected_user"] == "elioth"
    assert decision["accepted"] is True


def test_ambiguous_is_rejected(config):
    # Suficientes frames y score alto, pero margen pequeño.
    records = [{"elioth": 0.52, "emmanuel": 0.48} for _ in range(60)]
    decision = decide_from_records(records, config)
    assert decision["selected_user"] == "guest"
    assert decision["accepted"] is False


def test_decision_is_json_serializable(config):
    records = [{"elioth": 0.9, "emmanuel": 0.1} for _ in range(40)]
    decision = decide_from_records(records, config)
    # No debe lanzar: la decisión es serializable para --save-decision.
    json.dumps(decision)
