"""Pruebas del despachador de sesión (mapeo y modos)."""

from pathlib import Path

import pytest

from rp_face_login.config import load_config
from rp_face_login.session.dispatcher import (
    MODE_COMMAND,
    MODE_DRY_RUN,
    SessionDispatcher,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_from_config_builds_user_mapping():
    cfg = load_config(CONFIG_PATH)
    dispatcher = SessionDispatcher.from_config(cfg)
    assert dispatcher.mode == MODE_DRY_RUN
    assert set(dispatcher.commands.keys()) == {"elioth", "emmanuel", "guest"}
    assert dispatcher.command_for("elioth") == "echo start elioth"
    assert dispatcher.command_for("guest") == "echo start guest"


def test_unknown_user_returns_none_command():
    dispatcher = SessionDispatcher(mode=MODE_DRY_RUN, commands={"elioth": "echo x"})
    assert dispatcher.command_for("desconocido") is None


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        SessionDispatcher(mode="autostart", commands={})


def test_dry_run_does_not_execute(capsys):
    dispatcher = SessionDispatcher(mode=MODE_DRY_RUN, commands={"elioth": "echo start elioth"})
    result = dispatcher.dispatch("elioth")
    assert result.executed is False
    assert result.returncode is None
    assert result.command == "echo start elioth"
    assert "dry-run" in capsys.readouterr().out


def test_dispatch_unknown_user_raises():
    dispatcher = SessionDispatcher(mode=MODE_DRY_RUN, commands={"elioth": "echo x"})
    with pytest.raises(ValueError):
        dispatcher.dispatch("guest")


def test_command_mode_executes_local_command():
    dispatcher = SessionDispatcher(
        mode=MODE_COMMAND, commands={"emmanuel": "echo starting session for emmanuel"}
    )
    result = dispatcher.dispatch("emmanuel")
    assert result.executed is True
    assert result.returncode == 0
    assert result.stdout == "starting session for emmanuel"


def test_command_mode_reports_nonzero_returncode():
    dispatcher = SessionDispatcher(mode=MODE_COMMAND, commands={"guest": "exit 3"})
    result = dispatcher.dispatch("guest")
    assert result.executed is True
    assert result.returncode == 3
