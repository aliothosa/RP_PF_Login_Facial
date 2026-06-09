"""Login facial simulado (modo demo).

Orquesta el flujo completo SIN iniciar ninguna sesión real ni tocar greetd/PAM:

    cámara (5s) -> ZIP temporal -> inferencia softmax por frame ->
    agregación temporal -> política de decisión -> usuario seleccionado.

El resultado es ``elioth``, ``emmanuel`` o ``guest`` (rechazo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import AppConfig
from .decision.decision_policy import decide
from .inference.temporal_aggregation import aggregate_predictions

# Agregación neutra cuando no se obtuvo ningún frame válido -> fuerza rechazo.
_EMPTY_AGGREGATION = {
    "valid_frames": 0,
    "best_user": None,
    "best_score": 0.0,
    "second_user": None,
    "second_score": 0.0,
    "margin": 0.0,
}


def decide_from_records(
    records: List[Dict[str, object]], config: AppConfig
) -> Dict[str, object]:
    """Agrega predicciones por frame y aplica la política de decisión.

    Si no hay predicciones, devuelve un rechazo (``guest``) por frames
    insuficientes en lugar de fallar.
    """
    if not records:
        aggregation = dict(_EMPTY_AGGREGATION)
    else:
        aggregation = aggregate_predictions(records, class_names=list(config.model.classes))
    return decide(aggregation, config.decision)


def run_login_sim(
    config: AppConfig,
    *,
    name: str = "login",
    model_path: str | Path = "models/face_auth_model.keras",
    class_indices_path: Optional[str | Path] = None,
    debug_annotated: bool = False,
    save_decision: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Ejecuta el login simulado completo y devuelve la decisión."""
    # Imports diferidos: cámara (OpenCV) y modelo (TensorFlow).
    from .acquisition.camera_capture import capture_to_zip
    from .inference.batch_predictor import predict_zip

    capture = capture_to_zip(
        config,
        name=name,
        output_dir=config.output.output_dir,
        duration=config.camera.duration_seconds,
        camera_index=config.camera.index,
        debug_annotated=debug_annotated,
    )

    records = predict_zip(
        capture.zip_path,
        model_path,
        config,
        class_indices_path=class_indices_path,
    )

    decision = decide_from_records(records, config)
    _print_decision(decision)

    if save_decision is not None:
        save_path = Path(save_decision)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  decisión guardada : {save_path}")

    return decision


def _print_decision(decision: Dict[str, object]) -> None:
    print()
    print("=" * 44)
    print("  LOGIN FACIAL (SIMULACIÓN)")
    print("=" * 44)
    print(f"  Usuario seleccionado : {decision['selected_user']}")
    print(f"  Aceptado             : {decision['accepted']}")
    print(f"  Motivo               : {decision['reason']}")
    print(f"  Frames válidos       : {decision['valid_frames']}")
    if decision["best_user"] is not None:
        print(
            f"  Mejor               : {decision['best_user']} "
            f"({decision['best_score']:.3f})"
        )
        print(
            f"  Segundo             : {decision['second_user']} "
            f"({decision['second_score']:.3f})"
        )
        print(f"  Margen              : {decision['margin']:.3f}")
    print("=" * 44)
    print("  (Demostración: no se inicia ninguna sesión real.)")
