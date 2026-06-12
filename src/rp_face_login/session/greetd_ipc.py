"""Cliente mínimo del protocolo greetd-ipc (JSON sobre socket UNIX).

Ver greetd-ipc(7): mensajes con prefijo u32 (orden nativo) + payload UTF-8 JSON.
El socket se obtiene de la variable de entorno ``GREETD_SOCK`` (o ruta explícita).
"""

from __future__ import annotations

import json
import os
import socket
import struct
from typing import Callable, Iterable, Optional

PasswordCallback = Callable[[str, str], Optional[str]]


class GreetdIpcError(RuntimeError):
    """Error de comunicación o autenticación con greetd."""


class GreetdIpcClient:
    """Cliente síncrono para crear sesión, completar PAM e iniciar comando."""

    def __init__(self, socket_path: str | None = None, socket_env: str = "GREETD_SOCK") -> None:
        self.socket_env = socket_env
        self.socket_path = socket_path or os.environ.get(socket_env)
        if not self.socket_path:
            raise GreetdIpcError(
                f"No hay socket greetd: define {socket_env} o pasa socket_path."
            )

    def connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(self.socket_path)
        except OSError as exc:
            raise GreetdIpcError(f"No se pudo conectar a {self.socket_path}: {exc}") from exc
        return sock

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise GreetdIpcError("Conexión greetd cerrada inesperadamente.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def send_request(self, sock: socket.socket, request: dict) -> dict:
        payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
        sock.sendall(struct.pack("=I", len(payload)) + payload)
        header = self._recv_exact(sock, 4)
        length = struct.unpack("=I", header)[0]
        body = self._recv_exact(sock, length)
        try:
            response = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise GreetdIpcError(f"Respuesta JSON inválida de greetd: {body!r}") from exc
        if not isinstance(response, dict):
            raise GreetdIpcError(f"Respuesta greetd inesperada: {response!r}")
        return response

    def authenticate_session(
        self,
        sock: socket.socket,
        username: str,
        password_callback: PasswordCallback | None = None,
    ) -> None:
        """Crea sesión y completa el flujo PAM hasta recibir ``success``."""
        response = self.send_request(sock, {"type": "create_session", "username": username})
        self._handle_auth_loop(sock, response, password_callback)

    def start_session(
        self,
        sock: socket.socket,
        cmd: Iterable[str],
        env: Iterable[str] | None = None,
    ) -> None:
        """Solicita iniciar la sesión; greetd la lanza al terminar el greeter."""
        cmd_list = [str(part) for part in cmd]
        env_list = [str(entry) for entry in (env or [])]
        response = self.send_request(
            sock,
            {"type": "start_session", "cmd": cmd_list, "env": env_list},
        )
        if response.get("type") != "success":
            self._raise_error(response, "start_session")

    def cancel_session(self, sock: socket.socket) -> None:
        self.send_request(sock, {"type": "cancel_session"})

    def launch_session(
        self,
        username: str,
        cmd: Iterable[str],
        env: Iterable[str] | None = None,
        password_callback: PasswordCallback | None = None,
    ) -> None:
        """Autentica vía PAM e inicia sesión por greetd-ipc."""
        sock = self.connect()
        try:
            self.authenticate_session(sock, username, password_callback)
            self.start_session(sock, cmd, env)
        except Exception:
            try:
                self.cancel_session(sock)
            except (GreetdIpcError, OSError):
                pass
            raise
        finally:
            sock.close()

    def _handle_auth_loop(
        self,
        sock: socket.socket,
        response: dict,
        password_callback: PasswordCallback | None,
    ) -> None:
        while True:
            msg_type = response.get("type")
            if msg_type == "success":
                return
            if msg_type == "error":
                self._raise_error(response, "autenticación")
            if msg_type == "auth_message":
                auth_type = str(response.get("auth_message_type", ""))
                message = str(response.get("auth_message", ""))
                if auth_type in ("info", "error"):
                    response = self.send_request(sock, {"type": "post_auth_message_response"})
                    continue
                answer: Optional[str] = ""
                if password_callback is not None:
                    answer = password_callback(auth_type, message)
                request: dict = {"type": "post_auth_message_response"}
                if answer is not None:
                    request["response"] = answer
                response = self.send_request(sock, request)
                continue
            raise GreetdIpcError(f"Respuesta greetd inesperada: {response!r}")

    @staticmethod
    def _raise_error(response: dict, phase: str) -> None:
        error_type = response.get("error_type", "error")
        description = response.get("description", "")
        raise GreetdIpcError(
            f"greetd-ipc falló en {phase} ({error_type}): {description}".strip()
        )


def default_password_callback(
    *,
    password_env: str | None = None,
    username: str | None = None,
    prompt_password: bool = False,
) -> PasswordCallback:
    """Resuelve respuestas PAM: env por usuario, env global, prompt o cadena vacía."""

    def callback(auth_message_type: str, message: str) -> Optional[str]:
        if auth_message_type in ("info", "error"):
            return None
        value = _password_from_env(password_env, username)
        if value:
            return value
        if prompt_password:
            return _read_password_interactive(auth_message_type, message)
        if password_env:
            hint = _password_env_hint(password_env, username)
            raise GreetdIpcError(
                f"Define {hint} en el launcher de greetd "
                "(bajo greetd /dev/tty suele no ser escribible)."
            )
        return ""

    return callback


def _password_env_hint(password_env: str, username: str | None) -> str:
    if username:
        return f"{password_env}_{username.upper()} o {password_env}"
    return password_env


def _password_from_env(password_env: str | None, username: str | None) -> str:
    if not password_env:
        return ""
    if username:
        per_user = os.environ.get(f"{password_env}_{username.upper()}", "")
        if per_user:
            return per_user
    return os.environ.get(password_env, "")


def _read_password_interactive(auth_message_type: str, message: str) -> str:
    """Lee contraseña desde TTY; prueba /dev/tty y /dev/console (greetd VT)."""
    prompt = message.strip() or "Contraseña PAM: "
    last_exc: OSError | None = None
    for device in ("/dev/tty", "/dev/console"):
        try:
            return _read_password_from_device(device, auth_message_type, prompt)
        except OSError as exc:
            last_exc = exc
    raise GreetdIpcError(
        "No se pudo pedir contraseña PAM de forma interactiva bajo greetd. "
        f"Define FACE_LOGIN_PAM_PASSWORD en /opt/rp_face_login/bin/greetd-face-login. "
        f"Detalle: {last_exc}"
    )


def _read_password_from_device(device: str, auth_message_type: str, prompt: str) -> str:
    fd = os.open(device, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    try:
        with os.fdopen(fd, "r+", encoding="utf-8", buffering=1, closefd=True) as tty:
            if auth_message_type == "secret":
                import getpass

                tty.write(prompt)
                tty.flush()
                return getpass.getpass("", stream=tty)
            tty.write(f"{prompt}\n")
            tty.flush()
            return tty.readline().rstrip("\n")
    except OSError:
        os.close(fd)
        raise
