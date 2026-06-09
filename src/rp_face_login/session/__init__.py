"""Despacho de sesión: dry-run, command local o greetd-ipc."""

from .dispatcher import (
    MODE_COMMAND,
    MODE_DRY_RUN,
    MODE_GREETD_IPC,
    VALID_MODES,
    DispatchResult,
    SessionDispatcher,
)
from .greetd_ipc import GreetdIpcClient, GreetdIpcError

__all__ = [
    "SessionDispatcher",
    "DispatchResult",
    "GreetdIpcClient",
    "GreetdIpcError",
    "MODE_DRY_RUN",
    "MODE_COMMAND",
    "MODE_GREETD_IPC",
    "VALID_MODES",
]
