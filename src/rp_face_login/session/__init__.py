"""Despacho de sesión: abstracción segura (dry-run / command).

No modifica PAM/greetd ni inicia sesiones reales en esta fase.
"""

from .dispatcher import (
    MODE_COMMAND,
    MODE_DRY_RUN,
    VALID_MODES,
    DispatchResult,
    SessionDispatcher,
)

__all__ = [
    "SessionDispatcher",
    "DispatchResult",
    "MODE_DRY_RUN",
    "MODE_COMMAND",
    "VALID_MODES",
]
