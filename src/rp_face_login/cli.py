"""Punto de entrada de línea de comandos de rp_face_login.

Ejecutable desde la raíz del proyecto con::

    python -m rp_face_login.cli --help
    python -m rp_face_login.cli capture --name elioth --output-dir ./capturas --duration 5

Las opciones ``--camera-index``, ``--duration`` y ``--output-dir`` sobrescriben
la configuración cargada desde ``configs/default.yaml`` y pueden indicarse
después del subcomando. ``--config`` es global (antes del subcomando).
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .config import CLIOverrides, ConfigError, DEFAULT_CONFIG_PATH, load_config


_NOT_IMPLEMENTED_MSG = (
    "[rp_face_login] El subcomando '{name}' aún no está implementado en esta fase. "
    "La estructura modular ya está lista; la lógica se añadirá en fases posteriores."
)


def _placeholder(name: str):
    def _run(args: argparse.Namespace) -> int:
        print(_NOT_IMPLEMENTED_MSG.format(name=name))
        return 0

    return _run


def _overrides_from_args(args: argparse.Namespace) -> CLIOverrides:
    return CLIOverrides(
        camera_index=getattr(args, "camera_index", None),
        duration=getattr(args, "duration", None),
        output_dir=getattr(args, "output_dir", None),
    )


def _load(args: argparse.Namespace):
    """Carga la configuración aplicando overrides; imprime y sale si es inválida."""
    return load_config(args.config, overrides=_overrides_from_args(args))


def _build_parser() -> argparse.ArgumentParser:
    # Parser padre con los overrides; default=SUPPRESS evita pisar el valor
    # global cuando el flag no se usa tras el subcomando.
    overrides = argparse.ArgumentParser(add_help=False)
    overrides.add_argument(
        "--camera-index", type=int, default=argparse.SUPPRESS, help="Override de camera.index"
    )
    overrides.add_argument(
        "--duration", type=float, default=argparse.SUPPRESS,
        help="Override de camera.duration_seconds (segundos)",
    )
    overrides.add_argument(
        "--output-dir", default=argparse.SUPPRESS, help="Override de output.output_dir"
    )

    parser = argparse.ArgumentParser(
        prog="rp_face_login",
        description=(
            "Sistema de autenticación facial 1:N (elioth/emmanuel) con rechazo a "
            "'guest'. Estructura modular del proyecto RP_PF_Login_Facial."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rp_face_login {__version__}")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Ruta al archivo YAML de configuración (default: configs/default.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<comando>")

    # capture: adquisición temporal de login (reemplaza faceIdentifierNoView.py)
    p_capture = subparsers.add_parser(
        "capture", parents=[overrides],
        help="Captura temporal de rostros para login (sin vista previa).",
    )
    p_capture.add_argument("--name", default="usuario", help="Etiqueta del usuario")
    p_capture.add_argument(
        "--debug-annotated", action="store_true",
        help="Incluye frames anotados dentro del ZIP (depuración).",
    )
    p_capture.set_defaults(func=_run_capture)

    # prepare-dataset
    p_prep = subparsers.add_parser(
        "prepare-dataset", help="Genera dataset procesado (train/val/test) desde rostros crudos."
    )
    p_prep.add_argument("--raw-dir", default="data/faces", help="Directorio de imágenes por clase")
    p_prep.add_argument("--output-dir", default="data/processed", help="Directorio de salida")
    p_prep.add_argument("--train-ratio", type=float, default=0.70, help="Proporción de entrenamiento")
    p_prep.add_argument("--val-ratio", type=float, default=0.15, help="Proporción de validación")
    p_prep.add_argument("--test-ratio", type=float, default=0.15, help="Proporción de prueba")
    p_prep.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    p_prep.set_defaults(func=_run_prepare_dataset)

    # train
    p_train = subparsers.add_parser("train", help="Entrena el clasificador (transfer learning).")
    p_train.add_argument("--dataset-dir", default="data/processed", help="Dataset procesado")
    p_train.add_argument("--output", default="models/face_auth_model.keras", help="Modelo de salida")
    p_train.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    p_train.add_argument("--batch-size", type=int, default=32, help="Tamaño de batch")
    p_train.add_argument("--learning-rate", type=float, default=1e-3, help="Tasa de aprendizaje")
    p_train.add_argument("--dropout", type=float, default=0.3, help="Dropout de la cabeza")
    p_train.add_argument("--seed", type=int, default=42, help="Semilla")
    p_train.add_argument(
        "--backbone", default=None,
        help="Backbone (MobileNetV2 | EfficientNetB0); por defecto, el de config.",
    )
    p_train.set_defaults(func=_run_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evalúa el modelo sobre el set de test.")
    p_eval.add_argument("--dataset-dir", default="data/processed/test", help="Set de test")
    p_eval.add_argument("--model", default="models/face_auth_model.keras", help="Modelo a evaluar")
    p_eval.set_defaults(func=_placeholder("evaluate"))

    # predict-zip
    p_pred = subparsers.add_parser("predict-zip", help="Inferencia por frame desde un ZIP de login.")
    p_pred.add_argument("--zip", required=False, help="Ruta del ZIP de login")
    p_pred.add_argument("--model", default="models/face_auth_model.keras", help="Modelo a usar")
    p_pred.set_defaults(func=_placeholder("predict-zip"))

    # login-sim
    p_login = subparsers.add_parser(
        "login-sim", parents=[overrides], help="Login simulado completo (sin sesión real)."
    )
    p_login.add_argument("--model", default="models/face_auth_model.keras", help="Modelo a usar")
    p_login.set_defaults(func=_placeholder("login-sim"))

    # check-config
    p_check = subparsers.add_parser(
        "check-config", parents=[overrides],
        help="Valida el YAML de configuración (con overrides).",
    )
    p_check.set_defaults(func=_run_check_config)

    return parser


def _run_check_config(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"[rp_face_login] Configuración inválida: {exc}", file=sys.stderr)
        return 1
    print(f"[rp_face_login] Configuración válida: {args.config}")
    print(f"  clases del modelo : {cfg.model.classes}")
    print(f"  usuario fallback  : {cfg.decision.fallback_user}")
    print(f"  índice de cámara  : {cfg.camera.index}")
    print(f"  duración cámara   : {cfg.camera.duration_seconds}s")
    print(f"  directorio salida : {cfg.output.output_dir}")
    return 0


def _run_capture(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"[rp_face_login] Configuración inválida: {exc}", file=sys.stderr)
        return 1

    # Import diferido: evita requerir OpenCV para '--help' u otros comandos.
    from .acquisition.camera_capture import capture_to_zip

    try:
        capture_to_zip(
            cfg,
            name=args.name,
            output_dir=cfg.output.output_dir,
            duration=cfg.camera.duration_seconds,
            camera_index=cfg.camera.index,
            debug_annotated=args.debug_annotated,
        )
    except RuntimeError as exc:
        print(f"[rp_face_login] Error de captura: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_prepare_dataset(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[rp_face_login] Configuración inválida: {exc}", file=sys.stderr)
        return 1

    # Import diferido: evita requerir OpenCV para '--help' u otros comandos.
    from .training.dataset_loader import prepare_dataset

    try:
        prepare_dataset(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config=cfg,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[rp_face_login] Error al preparar dataset: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_train(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[rp_face_login] Configuración inválida: {exc}", file=sys.stderr)
        return 1

    # Import diferido: TensorFlow es pesado y opcional (extra [ml]).
    try:
        from .training.train_model import train
    except ImportError as exc:
        print(f"[rp_face_login] {exc}", file=sys.stderr)
        return 1

    try:
        train(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            config=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            seed=args.seed,
            backbone=args.backbone,
        )
    except ImportError as exc:
        print(f"[rp_face_login] {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"[rp_face_login] Error de entrenamiento: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
