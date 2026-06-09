#!/usr/bin/env bash
# Build reproducible de binarios con PyInstaller (entorno gestionado con uv).
#
# Uso:
#   ./scripts/build_pyinstaller.sh capture
#   ./scripts/build_pyinstaller.sh login-sim
#
# Requisitos:
#   - uv instalado (https://docs.astral.sh/uv/)
#   - extras instalados según el objetivo (ver abajo)
#
# Salida:
#   dist/face-login-capture   o   dist/face-login-sim
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${1:-capture}"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: se requiere 'uv'. Instálalo desde https://docs.astral.sh/uv/" >&2
    exit 1
fi

case "$TARGET" in
    capture)
        ENTRY="$REPO_ROOT/packaging/entry_capture.py"
        EXE_NAME="face-login-capture"
        UV_SYNC_EXTRAS=(--extra dev)
        PYINSTALLER_ARGS=()
        ;;
    login-sim)
        ENTRY="$REPO_ROOT/packaging/entry_login_sim.py"
        EXE_NAME="face-login-sim"
        UV_SYNC_EXTRAS=(--extra dev --extra ml)
        PYINSTALLER_ARGS=(
            --collect-all tensorflow
            --collect-all keras
            --hidden-import=tensorflow
        )
        ;;
    *)
        echo "Uso: $0 {capture|login-sim}" >&2
        exit 1
        ;;
esac

echo "==> Sincronizando entorno uv (${TARGET})..."
uv sync "${UV_SYNC_EXTRAS[@]}"

echo "==> Detectando Haar Cascade (sin rutas absolutas locales)..."
CASCADE="$(uv run python - <<'PY'
import os
import cv2
print(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
PY
)"

if [[ ! -f "$CASCADE" ]]; then
    echo "ERROR: no se encontró el XML del Haar Cascade en: $CASCADE" >&2
    exit 1
fi

echo "==> Construyendo ${EXE_NAME} (PyInstaller --onefile)..."
# No se versiona ningún .spec: todos los parámetros se pasan por CLI.
uv run pyinstaller \
    --onefile \
    --clean \
    --noconfirm \
    --name "$EXE_NAME" \
    --paths "$REPO_ROOT/src" \
    --add-data "${CASCADE}:cv2/data" \
    --add-data "${REPO_ROOT}/configs/default.yaml:configs" \
    "${PYINSTALLER_ARGS[@]}" \
    "$ENTRY"

echo ""
echo "Listo: dist/${EXE_NAME}"
echo "Ejecuta desde la raíz del repo o pasa rutas absolutas a --model / --output-dir."
