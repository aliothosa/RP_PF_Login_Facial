"""Despacho de sesión: abstracción segura para simulación e integración greetd.

Modos (``config.session_dispatch.mode``):

- ``dry-run``: solo imprime qué sesión se despacharía.
- ``command``: ejecuta un comando local (prototipo; no usa PAM).
- ``greetd-ipc``: autentica vía PAM y lanza sesión por el protocolo oficial de greetd.

Garantías:
- No se hardcodean contraseñas en el código.
- PAM sigue siendo la autoridad de autenticación en modo ``greetd-ipc``.
"""

from __future__ import annotations

import dataclasses
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .greetd_ipc import GreetdIpcClient, GreetdIpcError, default_password_callback

MODE_DRY_RUN = "dry-run"
MODE_COMMAND = "command"
MODE_GREETD_IPC = "greetd-ipc"
VALID_MODES = (MODE_DRY_RUN, MODE_COMMAND, MODE_GREETD_IPC)


@dataclass
class DispatchResult:
    user: str
    mode: str
    command: str
    executed: bool
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class SessionDispatcher:
    """Despachador de sesión basado en un mapeo usuario -> comando."""

    mode: str = MODE_DRY_RUN
    commands: Dict[str, str] = field(default_factory=dict)
    greetd_socket_env: str = "GREETD_SOCK"
    greetd_extra_env: List[str] = field(default_factory=list)
    greetd_password_env: Optional[str] = "FACE_LOGIN_PAM_PASSWORD"
    greetd_prompt_password: bool = False
    greetd_default_cmd: List[str] = field(default_factory=lambda: ["/usr/bin/startplasma-wayland"])

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Modo de despacho inválido: '{self.mode}'. Usa uno de {VALID_MODES}."
            )

    @classmethod
    def from_config(cls, config) -> "SessionDispatcher":
        """Construye el despachador desde ``config.session_dispatch``."""
        session_cfg = getattr(config, "session_dispatch", None) or {}
        mode = session_cfg.get("mode", MODE_DRY_RUN)
        users = session_cfg.get("users", {}) or {}
        greetd_cfg = session_cfg.get("greetd_ipc", {}) or {}

        commands: Dict[str, str] = {}
        for user, entry in users.items():
            command = _extract_command(entry)
            if command is not None:
                commands[user] = command

        default_cmd = greetd_cfg.get("default_cmd") or ["/usr/bin/startplasma-wayland"]
        if isinstance(default_cmd, str):
            default_cmd = shlex.split(default_cmd)

        return cls(
            mode=mode,
            commands=commands,
            greetd_socket_env=str(greetd_cfg.get("socket_env", "GREETD_SOCK")),
            greetd_extra_env=[str(x) for x in greetd_cfg.get("env", []) or []],
            greetd_password_env=greetd_cfg.get("password_env", "FACE_LOGIN_PAM_PASSWORD"),
            greetd_prompt_password=bool(greetd_cfg.get("prompt_password", False)),
            greetd_default_cmd=[str(x) for x in default_cmd],
        )

    def with_mode(self, mode: str) -> "SessionDispatcher":
        """Copia con otro modo (p. ej. override ``DISPATCH_MODE`` en el greeter)."""
        if mode and mode != self.mode:
            return dataclasses.replace(self, mode=mode)
        return self

    def command_for(self, user: str) -> Optional[str]:
        """Devuelve el comando configurado para ``user`` (o ``None``)."""
        return self.commands.get(user)

    def command_argv_for(self, user: str) -> List[str]:
        """Devuelve argv para ``start_session`` (usuario o default Plasma)."""
        raw = self.command_for(user)
        if raw is None:
            return list(self.greetd_default_cmd)
        return _command_to_argv(raw)

    def dispatch(self, user: str) -> DispatchResult:
        """Despacha (o simula) la sesión para ``user`` según el modo configurado."""
        command = self.command_for(user)
        if command is None and self.mode != MODE_GREETD_IPC:
            raise ValueError(f"No hay comando de sesión configurado para '{user}'.")

        display_command = command or " ".join(self.greetd_default_cmd)

        if self.mode == MODE_DRY_RUN:
            print(f"[dry-run] Se despacharía sesión para '{user}': {display_command}")
            return DispatchResult(
                user=user, mode=self.mode, command=display_command, executed=False
            )

        if self.mode == MODE_GREETD_IPC:
            return self._dispatch_greetd_ipc(user, display_command)

        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = DispatchResult(
            user=user,
            mode=self.mode,
            command=command or "",
            executed=True,
            returncode=proc.returncode,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
        print(f"[command] Sesión '{user}' -> rc={result.returncode}: {result.stdout}")
        return result

    def _dispatch_greetd_ipc(self, user: str, display_command: str) -> DispatchResult:
        cmd = self.command_argv_for(user)
        password_cb = default_password_callback(
            password_env=self.greetd_password_env,
            prompt_password=self.greetd_prompt_password,
        )
        client = GreetdIpcClient(socket_env=self.greetd_socket_env)
        try:
            client.launch_session(
                username=user,
                cmd=cmd,
                env=self.greetd_extra_env,
                password_callback=password_cb,
            )
        except GreetdIpcError as exc:
            raise RuntimeError(str(exc)) from exc

        summary = f"[greetd-ipc] Sesión solicitada para '{user}': {' '.join(cmd)}"
        print(summary)
        return DispatchResult(
            user=user,
            mode=self.mode,
            command=display_command,
            executed=True,
            returncode=0,
            stdout=summary,
        )


def _extract_command(entry: Any) -> Optional[str]:
    if isinstance(entry, dict):
        if "cmd" in entry:
            cmd = entry["cmd"]
            if isinstance(cmd, list):
                return " ".join(shlex.quote(str(x)) for x in cmd)
            return str(cmd)
        command = entry.get("command")
        return str(command) if command is not None else None
    if entry is None:
        return None
    if isinstance(entry, list):
        return " ".join(shlex.quote(str(x)) for x in entry)
    return str(entry)


def _command_to_argv(command: str | Iterable[str]) -> List[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return [str(part) for part in command]
