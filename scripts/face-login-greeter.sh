#!/usr/bin/env bash
#
# face-login-greeter.sh (EXPERIMENTAL)
#
# Greeter de prototipo que encadena:
#   1. login-sim            -> captura + inferencia + decisión
#   2. lee selected_user    -> elioth | emmanuel | guest
#   3. session dispatcher   -> dry-run o command (según config / DISPATCH_MODE)
#   4. guarda logs          -> logs/face-login.log (o LOG_FILE)
#
# Variables de entorno (despliegue VM / greetd):
#   REPO_ROOT          Raíz del proyecto o /opt/rp_face_login
#   CONFIG             Ruta al YAML (default: $REPO_ROOT/configs/default.yaml)
#   MODEL              Ruta al .keras
#   LOG_FILE           Ruta del log (default: $REPO_ROOT/logs/face-login.log)
#   DISPATCH_MODE      dry-run | command | greetd-ipc (sobrescribe config)
#   FACE_LOGIN_BIN     Binario PyInstaller face-login-sim (opcional; evita uv)
#   PYTHON_BIN         intérprete alternativo (opcional)
#
# Códigos de salida:
#   0  OK (usuario aceptado o despacho completado)
#   1  Error (cámara, modelo, TensorFlow, dispatcher)
#   2  Rechazo a guest (identidad no aceptada; greetd puede reiniciar greeter)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/default.yaml}"
MODEL="${MODEL:-${REPO_ROOT}/models/face_auth_model.keras}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/face-login.log}"
DISPATCH_MODE="${DISPATCH_MODE:-}"

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || LOG_FILE="/tmp/face-login.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

# Intérprete / binario para login-sim
if [[ -n "${FACE_LOGIN_BIN:-}" && -x "$FACE_LOGIN_BIN" ]]; then
    LOGIN_CMD=("$FACE_LOGIN_BIN")
    PY_DESC="$FACE_LOGIN_BIN"
elif [[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]]; then
    # PYTHONPATH lo fija el launcher (p. ej. ~/RP_PF_Login_Facial/src)
    LOGIN_CMD=("$PYTHON_BIN" -m rp_face_login.cli)
    PY_DESC="$PYTHON_BIN"
elif command -v uv >/dev/null 2>&1 && [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
    cd "$REPO_ROOT"
    LOGIN_CMD=(uv run python -m rp_face_login.cli)
    PY_DESC="uv run python (cwd=$REPO_ROOT)"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    LOGIN_CMD=("${REPO_ROOT}/.venv/bin/python" -m rp_face_login.cli)
    PY_DESC="${REPO_ROOT}/.venv/bin/python"
else
    export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    LOGIN_CMD=(python3 -m rp_face_login.cli)
    PY_DESC="python3"
fi

# Python para dispatcher (siempre módulo; ligero, no requiere TF si ya falló antes)
if [[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]]; then
    DISPATCH_PY=("$PYTHON_BIN")
elif command -v uv >/dev/null 2>&1 && [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
    DISPATCH_PY=(uv run python)
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    DISPATCH_PY=("${REPO_ROOT}/.venv/bin/python")
else
    DISPATCH_PY=(python3)
fi
if [[ -z "${PYTHONPATH:-}" && -d "${REPO_ROOT}/src" ]]; then
    export PYTHONPATH="${REPO_ROOT}/src"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== face-login-greeter ==="
log "repo=$REPO_ROOT config=$CONFIG model=$MODEL log=$LOG_FILE"
log "login=$PY_DESC dispatch_mode=${DISPATCH_MODE:-<config>}"
if [[ "${DISPATCH_MODE:-}" == "greetd-ipc" ]]; then
    pam_state="<no definido>"
    [[ -n "${FACE_LOGIN_PAM_PASSWORD:-}" ]] && pam_state="<definido>"
    log "greetd: GREETD_SOCK=${GREETD_SOCK:-<no definido>} FACE_LOGIN_PAM_PASSWORD=${pam_state}"
fi

DECISION_JSON="$(mktemp -t face-login-decision-XXXXXX.json)"
trap 'rm -f "$DECISION_JSON"' EXIT

# 1) login-sim
log "Paso 1/3: ejecutando login-sim..."
if ! "${LOGIN_CMD[@]}" --config "$CONFIG" login-sim \
        --model "$MODEL" --save-decision "$DECISION_JSON" "$@" >>"$LOG_FILE" 2>&1; then
    log "ERROR: login-sim falló. Revisa $LOG_FILE (¿cámara / modelo / TensorFlow?)."
    exit 1
fi

# 2) selected_user
SELECTED_USER="$("${DISPATCH_PY[@]}" - "$DECISION_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
print(d.get("selected_user", "guest"))
PY
)"
ACCEPTED="$("${DISPATCH_PY[@]}" - "$DECISION_JSON" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
print("true" if d.get("accepted") else "false")
PY
)"
log "Paso 2/3: selected_user=$SELECTED_USER accepted=$ACCEPTED"

if [[ "$SELECTED_USER" == "guest" || "$ACCEPTED" != "true" ]]; then
    log "Rechazo: no se despacha sesión (guest o no aceptado)."
    exit 2
fi

# 3) dispatcher
log "Paso 3/3: despachando sesión..."
if ! "${DISPATCH_PY[@]}" - "$CONFIG" "$SELECTED_USER" "$DISPATCH_MODE" <<'PY' >>"$LOG_FILE" 2>&1; then
import sys
import traceback
from rp_face_login.config import load_config
from rp_face_login.session.dispatcher import SessionDispatcher

try:
    config_path, user, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg = load_config(config_path)
    dispatcher = SessionDispatcher.from_config(cfg).with_mode(mode)
    result = dispatcher.dispatch(user)
    print(
        f"dispatch -> user={result.user} mode={result.mode} "
        f"executed={result.executed} returncode={result.returncode}"
    )
except Exception:
    traceback.print_exc()
    raise
PY
    log "ERROR: dispatcher falló para '$SELECTED_USER'. Revisa traceback arriba en $LOG_FILE"
    exit 1
fi

log "Hecho. Usuario=$SELECTED_USER. Log: $LOG_FILE"
exit 0
