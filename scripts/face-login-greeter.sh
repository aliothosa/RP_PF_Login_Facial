#!/usr/bin/env bash
#
# face-login-greeter.sh (EXPERIMENTAL)
#
# Greeter de prototipo que encadena:
#   1. login-sim            -> captura + inferencia + decisión
#   2. lee selected_user    -> elioth | emmanuel | guest
#   3. session dispatcher   -> dry-run o command (según config / DISPATCH_MODE)
#   4. guarda logs          -> logs/face-login.log
#
# Garantías de seguridad de esta fase:
#   - NO modifica ningún archivo de /etc.
#   - NO instala greetd.
#   - NO inicia una sesión KDE real (el dispatcher solo simula o ejecuta un
#     comando local configurable, p. ej. 'echo').
#
# Variables de entorno opcionales:
#   CONFIG=configs/default.yaml          Ruta del YAML de configuración.
#   MODEL=models/face_auth_model.keras   Ruta del modelo entrenado.
#   DISPATCH_MODE=dry-run|command        Sobrescribe el modo del dispatcher.
# Cualquier argumento extra se reenvía a 'login-sim'
#   (p. ej. --duration 5 --camera-index 0 --debug-annotated).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/face-login.log"
mkdir -p "$LOG_DIR"

# Selección de intérprete: usa el venv del proyecto si existe.
if [[ -x "$REPO_ROOT/venv/bin/python" ]]; then
    PY="$REPO_ROOT/venv/bin/python"
else
    PY="python3"
fi
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-configs/default.yaml}"
MODEL="${MODEL:-models/face_auth_model.keras}"
DISPATCH_MODE="${DISPATCH_MODE:-}"   # vacío = usar el modo definido en config

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== face-login-greeter (EXPERIMENTAL: no toca /etc, no inicia KDE real) ==="
log "python=$PY config=$CONFIG model=$MODEL dispatch_mode=${DISPATCH_MODE:-<config>}"

DECISION_JSON="$(mktemp -t face-login-decision-XXXXXX.json)"
trap 'rm -f "$DECISION_JSON"' EXIT

# 1) Ejecutar login-sim y guardar la decisión.
log "Paso 1/3: ejecutando login-sim..."
if ! "$PY" -m rp_face_login.cli --config "$CONFIG" login-sim \
        --model "$MODEL" --save-decision "$DECISION_JSON" "$@" >>"$LOG_FILE" 2>&1; then
    log "ERROR: login-sim falló. Revisa $LOG_FILE (¿cámara / modelo / TensorFlow?)."
    exit 1
fi

# 2) Extraer selected_user del JSON de decisión.
SELECTED_USER="$("$PY" - "$DECISION_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    decision = json.load(fh)
print(decision.get("selected_user", "guest"))
PY
)"
log "Paso 2/3: selected_user = ${SELECTED_USER}"

# 3) Despachar la sesión mediante el dispatcher (sin iniciar sesión real).
log "Paso 3/3: despachando sesión (dry-run/command, sin login real)..."
if ! "$PY" - "$CONFIG" "$SELECTED_USER" "$DISPATCH_MODE" <<'PY' >>"$LOG_FILE" 2>&1; then
import sys
from rp_face_login.config import load_config
from rp_face_login.session.dispatcher import SessionDispatcher

config_path, user, mode = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_config(config_path)
dispatcher = SessionDispatcher.from_config(cfg)
if mode:
    dispatcher = SessionDispatcher(mode=mode, commands=dispatcher.commands)

result = dispatcher.dispatch(user)
print(
    f"dispatch -> user={result.user} mode={result.mode} "
    f"executed={result.executed} returncode={result.returncode}"
)
PY
    log "ERROR: el dispatcher falló para el usuario '${SELECTED_USER}'."
    exit 1
fi

log "Hecho. Usuario=${SELECTED_USER}. Log completo en ${LOG_FILE}"
