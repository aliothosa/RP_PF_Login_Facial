"""Política de decisión: acepta identidad conocida o rechaza a 'guest'.

'guest' es un mecanismo de rechazo, NO una clase entrenada.
"""

from .decision_policy import (
    REASON_ACCEPTED,
    REASON_INSUFFICIENT_FRAMES,
    REASON_LOW_CONFIDENCE,
    REASON_MARGIN_BELOW_THRESHOLD,
    decide,
)

__all__ = [
    "decide",
    "REASON_ACCEPTED",
    "REASON_INSUFFICIENT_FRAMES",
    "REASON_LOW_CONFIDENCE",
    "REASON_MARGIN_BELOW_THRESHOLD",
]
