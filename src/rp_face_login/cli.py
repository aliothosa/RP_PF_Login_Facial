"""Punto de entrada de línea de comandos de rp_face_login.

Ejecutable desde la raíz del proyecto con::

    python -m rp_face_login.cli --help

Las opciones globales ``--config``, ``--camera-index``, ``--duration`` y
``--output-dir`` permiten sobrescribir la configuración cargada desde
``configs/default.yaml``. En esta fase los subcomandos son *placeholders* con
sus argumentos definidos pero sin lógica de captura/entrenamiento/inferencia.
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rp_face_login",
        description=(
            "Sistema de autenticación facial 1:N (elioth/emmanuel) con rechazo a "
            "'guest'. Estructura modular del proyecto RP_PF_Login_Facial."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rp_face_login {__version__}")

    # Opciones globales de configuración / override (anteriores al subcomando).
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Ruta al archivo YAML de configuración (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override de camera.index",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override de camera.duration_seconds (segundos)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override de output.output_dir",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<comando>")

    # capture: adquisición temporal de login (reemplaza faceIdentifierNoView.py)
    p_capture = subparsers.add_parser(
        "capture", help="Captura temporal de rostros para login (sin vista previa)."
    )
    p_capture.add_argument("--name", default="usuario", help="Etiqueta del usuario")
    p_capture.set_defaults(func=_placeholder("capture"))

    # prepare-dataset: rostros crudos -> dataset procesado (reemplaza face_extractor.py)
    p_prep = subparsers.add_parser(
        "prepare-dataset", help="Genera dataset procesado (train/val/test) desde rostros crudos."
    )
    p_prep.add_argument("--raw-dir", default="data/faces", help="Directorio de rostros crudos")
    p_prep.add_argument("--processed-dir", default="data/processed", help="Directorio de salida")
    p_prep.set_defaults(func=_placeholder("prepare-dataset"))

    # train
    p_train = subparsers.add_parser("train", help="Entrena el clasificador (transfer learning).")
    p_train.add_argument("--dataset-dir", default="data/processed", help="Dataset procesado")
    p_train.add_argument("--output", default="models/face_auth_model.keras", help="Modelo de salida")
    p_train.set_defaults(func=_placeholder("train"))

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
    p_login = subparsers.add_parser("login-sim", help="Login simulado completo (sin sesión real).")
    p_login.add_argument("--model", default="models/face_auth_model.keras", help="Modelo a usar")
    p_login.set_defaults(func=_placeholder("login-sim"))

    # check-config: valida la carga de configuración y muestra valores efectivos
    p_check = subparsers.add_parser("check-config", help="Valida el YAML de configuración (con overrides).")
    p_check.set_defaults(func=_run_check_config)

    return parser


def _run_check_config(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config, overrides=_overrides_from_args(args))
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
