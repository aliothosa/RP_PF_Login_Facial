"""Agregación temporal de predicciones softmax por frame.

Promedia los scores por clase a lo largo de los frames válidos (Temporal Average
Pooling) y resume la decisión candidata: mejor y segundo usuario y el margen
entre ambos.

Fórmula:
    avg_score(clase_j) = (1 / N) * sum_i  P(clase_j | frame_i)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# Claves que no representan una clase dentro de un registro de predicción.
RESERVED_KEYS = {"frame"}


def aggregate_predictions(
    predictions: Sequence[Dict[str, object]],
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Calcula el promedio temporal por clase y la decisión candidata.

    Args:
        predictions: lista de dicts por frame, p. ej.
            ``[{"elioth": 0.91, "emmanuel": 0.09}, ...]``. Se ignora la clave
            ``"frame"`` si está presente.
        class_names: nombres de clase a considerar. Si es ``None``, se infieren
            de las claves de las predicciones (excluyendo las reservadas).

    Returns:
        dict con ``avg_scores``, ``valid_frames``, ``best_user``, ``best_score``,
        ``second_user``, ``second_score`` y ``margin``.
    """
    if not predictions:
        raise ValueError("No hay predicciones para agregar.")

    if class_names is None:
        keys = set()
        for pred in predictions:
            keys.update(pred.keys())
        keys -= RESERVED_KEYS
        class_names = sorted(keys)

    if not class_names:
        raise ValueError("No se encontraron clases en las predicciones.")

    n = len(predictions)
    avg_scores: Dict[str, float] = {}
    for cls in class_names:
        total = sum(float(pred.get(cls, 0.0)) for pred in predictions)
        avg_scores[cls] = total / n

    # Orden descendente por score; desempate estable por nombre de clase.
    ranking: List[tuple] = sorted(avg_scores.items(), key=lambda kv: (-kv[1], kv[0]))

    best_user, best_score = ranking[0]
    if len(ranking) > 1:
        second_user, second_score = ranking[1]
    else:
        second_user, second_score = None, 0.0

    return {
        "avg_scores": avg_scores,
        "valid_frames": n,
        "best_user": best_user,
        "best_score": best_score,
        "second_user": second_user,
        "second_score": second_score,
        "margin": best_score - second_score,
    }
