"""Pruebas de humo del CLI: que el parser se construya y --help funcione."""

import pytest

from rp_face_login.cli import _build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "rp_face_login" in out


def test_no_command_prints_help(capsys):
    code = main([])
    assert code == 0
    assert "comando" in capsys.readouterr().out.lower()


def test_subcommands_registered():
    parser = _build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    registered = set()
    for action in actions:
        registered.update(action.choices.keys())
    for expected in ["capture", "prepare-dataset", "train", "evaluate", "predict-zip", "login-sim"]:
        assert expected in registered


def test_check_config_applies_overrides(capsys, tmp_path):
    import textwrap

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            camera:
              index: 0
            face_detection: {}
            preprocessing: {}
            model: {}
            decision: {}
            output: {}
            """
        ),
        encoding="utf-8",
    )
    code = main(["--config", str(cfg), "--camera-index", "3", "--duration", "9", "check-config"])
    assert code == 0
    out = capsys.readouterr().out
    assert "índice de cámara  : 3" in out
    assert "9.0s" in out


def test_check_config_invalid_returns_one(capsys):
    code = main(["--config", "configs/no_existe.yaml", "check-config"])
    assert code == 1
