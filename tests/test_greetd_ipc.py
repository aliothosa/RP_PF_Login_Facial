"""Pruebas del cliente greetd-ipc con servidor mock."""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
from pathlib import Path

import pytest

from rp_face_login.session.greetd_ipc import (
    GreetdIpcClient,
    GreetdIpcError,
    default_password_callback,
)


def _pack_message(payload: dict) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return struct.pack("=I", len(body)) + body


def _read_message(conn: socket.socket) -> dict:
    header = conn.recv(4)
    if len(header) < 4:
        raise RuntimeError("conexión cerrada")
    length = struct.unpack("=I", header)[0]
    body = b""
    while len(body) < length:
        chunk = conn.recv(length - len(body))
        if not chunk:
            raise RuntimeError("conexión cerrada")
        body += chunk
    return json.loads(body.decode("utf-8"))


def _run_mock_greetd(sock_path: str, *, ask_password: bool = False) -> None:
    ready = threading.Event()

    def handler() -> None:
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(sock_path)
        server.listen(1)
        ready.set()
        conn, _ = server.accept()
        try:
            msg = _read_message(conn)
            assert msg["type"] == "create_session"
            if ask_password:
                conn.sendall(
                    _pack_message(
                        {
                            "type": "auth_message",
                            "auth_message_type": "secret",
                            "auth_message": "Password:",
                        }
                    )
                )
                auth = _read_message(conn)
                assert auth["type"] == "post_auth_message_response"
                assert auth.get("response") == "secret123"
            conn.sendall(_pack_message({"type": "success"}))

            start = _read_message(conn)
            assert start["type"] == "start_session"
            assert start["cmd"] == ["/usr/bin/startplasma-wayland"]
            conn.sendall(_pack_message({"type": "success"}))
        finally:
            conn.close()
            server.close()

    threading.Thread(target=handler, daemon=True).start()
    assert ready.wait(timeout=5), "mock greetd no arrancó a tiempo"


@pytest.fixture
def greetd_sock_path(tmp_path: Path) -> str:
    return str(tmp_path / "greetd.sock")


def test_launch_session_success(greetd_sock_path: str) -> None:
    _run_mock_greetd(greetd_sock_path)
    client = GreetdIpcClient(socket_path=greetd_sock_path)
    client.launch_session("elioth", ["/usr/bin/startplasma-wayland"])


def test_launch_session_with_password_callback(greetd_sock_path: str) -> None:
    _run_mock_greetd(greetd_sock_path, ask_password=True)

    def password_cb(auth_type: str, message: str) -> str:
        assert auth_type == "secret"
        return "secret123"

    client = GreetdIpcClient(socket_path=greetd_sock_path)
    client.launch_session(
        "emmanuel",
        ["/usr/bin/startplasma-wayland"],
        password_callback=password_cb,
    )


def test_default_password_callback_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("FACE_LOGIN_PAM_PASSWORD", "from-env")
    cb = default_password_callback(password_env="FACE_LOGIN_PAM_PASSWORD", prompt_password=False)
    assert cb("secret", "Password:") == "from-env"
    assert cb("info", "msg") is None


def test_default_password_callback_uses_per_user_env(monkeypatch) -> None:
    monkeypatch.setenv("FACE_LOGIN_PAM_PASSWORD_ELIOTH", "elioth-pass")
    monkeypatch.setenv("FACE_LOGIN_PAM_PASSWORD", "fallback")
    cb = default_password_callback(
        password_env="FACE_LOGIN_PAM_PASSWORD",
        username="elioth",
        prompt_password=False,
    )
    assert cb("secret", "Password:") == "elioth-pass"


def test_default_password_callback_missing_env_raises() -> None:
    cb = default_password_callback(password_env="FACE_LOGIN_PAM_PASSWORD", prompt_password=False)
    with pytest.raises(GreetdIpcError, match="FACE_LOGIN_PAM_PASSWORD"):
        cb("secret", "Password:")


def test_missing_greetd_sock_raises(monkeypatch) -> None:
    monkeypatch.delenv("GREETD_SOCK", raising=False)
    with pytest.raises(GreetdIpcError, match="GREETD_SOCK"):
        GreetdIpcClient()


def test_auth_error_from_greetd(greetd_sock_path: str) -> None:
    ready = threading.Event()

    def handler() -> None:
        if os.path.exists(greetd_sock_path):
            os.unlink(greetd_sock_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(greetd_sock_path)
        server.listen(1)
        ready.set()
        conn, _ = server.accept()
        try:
            _read_message(conn)
            conn.sendall(
                _pack_message(
                    {
                        "type": "error",
                        "error_type": "auth_error",
                        "description": "bad password",
                    }
                )
            )
        finally:
            conn.close()
            server.close()

    threading.Thread(target=handler, daemon=True).start()
    assert ready.wait(timeout=5)
    client = GreetdIpcClient(socket_path=greetd_sock_path)
    with pytest.raises(GreetdIpcError, match="bad password"):
        client.launch_session("elioth", ["/bin/true"])
