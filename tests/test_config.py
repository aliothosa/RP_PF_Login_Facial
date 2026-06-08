"""Pruebas unitarias de carga, validación y override de configuración."""

from pathlib import Path

import pytest

from rp_face_login.config import (
    AppConfig,
    CLIOverrides,
    ConfigError,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

FULL_CONFIG = """
camera:
  index: 0
  duration_seconds: 5.0
face_detection:
  scale_factor: 1.1
preprocessing:
  color_format: "RGB"
model:
  classes: ["elioth", "emmanuel"]
decision:
  fallback_user: "guest"
output:
  output_dir: "./capturas"
"""


def _write(tmp_path, text: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    return cfg


def test_default_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, AppConfig)
    assert cfg.model.classes == ["elioth", "emmanuel"]
    assert cfg.decision.fallback_user == "guest"
    assert cfg.camera.duration_seconds == 5.0


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="No existe"):
        load_config("configs/no_existe.yaml")


def test_missing_section_raises(tmp_path):
    cfg = _write(tmp_path, "camera:\n  index: 0\n")
    with pytest.raises(ConfigError, match="Faltan secciones"):
        load_config(cfg)


def test_empty_file_raises(tmp_path):
    cfg = _write(tmp_path, "")
    with pytest.raises(ConfigError, match="vacío"):
        load_config(cfg)


def test_unknown_key_raises(tmp_path):
    bad = FULL_CONFIG + "  unexpected_key: 1\n"
    cfg = _write(tmp_path, bad)
    with pytest.raises(ConfigError, match="no reconocidas"):
        load_config(cfg)


def test_cli_overrides_applied(tmp_path):
    cfg = _write(tmp_path, FULL_CONFIG)
    result = load_config(
        cfg,
        overrides=CLIOverrides(camera_index=2, duration=8.0, output_dir="/tmp/out"),
    )
    assert result.camera.index == 2
    assert result.camera.duration_seconds == 8.0
    assert result.output.output_dir == "/tmp/out"


def test_override_none_keeps_defaults(tmp_path):
    cfg = _write(tmp_path, FULL_CONFIG)
    result = load_config(cfg, overrides=CLIOverrides())
    assert result.camera.index == 0
    assert result.camera.duration_seconds == 5.0


def test_invalid_duration_override_raises(tmp_path):
    cfg = _write(tmp_path, FULL_CONFIG)
    with pytest.raises(ConfigError, match="duration"):
        load_config(cfg, overrides=CLIOverrides(duration=0))
