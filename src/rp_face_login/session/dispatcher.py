"""Despacho de sesión: abstracción segura, sin tocar el login real del sistema.

Soporta dos modos (leídos desde ``config.session_dispatch``):

- ``dry-run``: solo imprime qué sesión se despacharía (no ejecuta nada).
- ``command``: ejecuta un comando local configurado (p. ej. ``echo start elioth``).

Garantías de seguridad de esta fase:
- No se hardcodean contraseñas.
- No se desactiva ni se interactúa con PAM.
- No se inicia ninguna sesión real (greetd/SDDM/etc.).
El mapeo usuario -> comando es totalmente configurable.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional

MODE_DRY_RUN = "dry-run"
MODE_COMMAND = "command"
VALID_MODES = (MODE_DRY_RUN, MODE_COMMAND)


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

        commands: Dict[str, str] = {}
        for user, entry in users.items():
            if isinstance(entry, dict):
                command = entry.get("command")
            else:
                command = entry
            if command is not None:
                commands[user] = str(command)

        return cls(mode=mode, commands=commands)

    def command_for(self, user: str) -> Optional[str]:
        """Devuelve el comando configurado para ``user`` (o ``None``)."""
        return self.commands.get(user)

    def dispatch(self, user: str) -> DispatchResult:
        """Despacha (o simula) la sesión para ``user`` según el modo configurado."""
        command = self.command_for(user)
        if command is None:
            raise ValueError(f"No hay comando de sesión configurado para '{user}'.")

        if self.mode == MODE_DRY_RUN:
            print(f"[dry-run] Se despacharía sesión para '{user}': {command}")
            return DispatchResult(user=user, mode=self.mode, command=command, executed=False)

        # MODE_COMMAND: ejecuta el comando local configurado (sin PAM, sin contraseñas).
        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        result = DispatchResult(
            user=user,
            mode=self.mode,
            command=command,
            executed=True,
            returncode=proc.returncode,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
        print(f"[command] Sesión '{user}' -> rc={result.returncode}: {result.stdout}")
        return result
