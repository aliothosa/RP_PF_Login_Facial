"""Política de decisión: acepta una identidad conocida o rechaza a ``guest``.

Reglas de aceptación (deben cumplirse las tres):
    1. valid_frames >= min_valid_frames
    2. best_score   >= confidence_threshold
    3. margin       >= margin_threshold

Si alguna falla, ``selected_user`` es el usuario de rechazo (``guest``). ``guest``
NO es una clase entrenada: solo aparece aquí como mecanismo de rechazo.
"""

from __future__ import annotations

from typing import Dict

# Códigos de razón (explicación de la decisión).
REASON_ACCEPTED = "accepted"
REASON_INSUFFICIENT_FRAMES = "insufficient_frames"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_MARGIN_BELOW_THRESHOLD = "margin_below_threshold"


def decide(aggregation: Dict[str, object], decision_config) -> Dict[str, object]:
    """Aplica la política de decisión sobre el resultado de la agregación temporal.

    Args:
        aggregation: salida de ``aggregate_predictions`` (incluye ``valid_frames``,
            ``best_user``, ``best_score``, ``second_user``, ``second_score``,
            ``margin``).
        decision_config: objeto con ``min_valid_frames``, ``confidence_threshold``,
            ``margin_threshold`` y ``fallback_user``.

    Returns:
        dict con la decisión y su explicación.
    """
    min_valid_frames = int(decision_config.min_valid_frames)
    confidence_threshold = float(decision_config.confidence_threshold)
    margin_threshold = float(decision_config.margin_threshold)
    fallback_user = decision_config.fallback_user

    valid_frames = int(aggregation["valid_frames"])
    best_user = aggregation["best_user"]
    best_score = float(aggregation["best_score"])
    margin = float(aggregation["margin"])

    # El orden importa: se reporta la primera condición que falla.
    if valid_frames < min_valid_frames:
        accepted, reason = False, REASON_INSUFFICIENT_FRAMES
    elif best_score < confidence_threshold:
        accepted, reason = False, REASON_LOW_CONFIDENCE
    elif margin < margin_threshold:
        accepted, reason = False, REASON_MARGIN_BELOW_THRESHOLD
    else:
        accepted, reason = True, REASON_ACCEPTED

    selected_user = best_user if accepted else fallback_user

    return {
        "selected_user": selected_user,
        "accepted": accepted,
        "reason": reason,
        "valid_frames": valid_frames,
        "best_user": best_user,
        "best_score": best_score,
        "second_user": aggregation.get("second_user"),
        "second_score": float(aggregation.get("second_score", 0.0)),
        "margin": margin,
        "thresholds": {
            "min_valid_frames": min_valid_frames,
            "confidence_threshold": confidence_threshold,
            "margin_threshold": margin_threshold,
        },
    }
