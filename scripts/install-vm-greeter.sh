#!/usr/bin/env bash
# Instala el greeter facial bajo /opt/rp_face_login para uso con greetd en VM.
# NO modifica /etc/greetd ni habilita greetd: eso queda documentado en
# docs/deploy_endeavouros_vm.md (§11).
#
# Uso (desde la raíz del repo, con modelo ya entrenado):
#   sudo ./scripts/install-vm-greeter.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/rp_face_login}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "Ejecuta con sudo: sudo $0" >&2
    exit 1
fi

echo "==> Instalando en ${INSTALL_ROOT}"

mkdir -p "${INSTALL_ROOT}"/{bin,configs,models,capturas,logs,reports}

# Config (usa el example de VM si no hay default local personalizado)
if [[ -f "${REPO_ROOT}/configs/default.yaml" ]]; then
    cp "${REPO_ROOT}/configs/default.yaml" "${INSTALL_ROOT}/configs/default.yaml"
else
    cp "${REPO_ROOT}/configs/greetd-vm.example.yaml" "${INSTALL_ROOT}/configs/default.yaml"
fi

# Modelo (obligatorio para login-sim)
if [[ -f "${REPO_ROOT}/models/face_auth_model.keras" ]]; then
    cp "${REPO_ROOT}/models/face_auth_model.keras" "${INSTALL_ROOT}/models/"
    cp "${REPO_ROOT}/models/class_indices.json" "${INSTALL_ROOT}/models/" 2>/dev/null || true
else
    echo "AVISO: no hay models/face_auth_model.keras en el repo. Copia el modelo antes de usar greetd." >&2
fi

# Greeter wrapper instalado
install -m 755 "${REPO_ROOT}/scripts/face-login-greeter.sh" "${INSTALL_ROOT}/bin/face-login-greeter"

# Launcher que fija variables para el usuario greeter de greetd
cat > "${INSTALL_ROOT}/bin/greetd-face-login" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher para greetd default_session (usuario greeter).
export REPO_ROOT="/opt/rp_face_login"
export CONFIG="/opt/rp_face_login/configs/default.yaml"
export MODEL="/opt/rp_face_login/models/face_auth_model.keras"
export LOG_FILE="/opt/rp_face_login/logs/face-login.log"
export DISPATCH_MODE="${DISPATCH_MODE:-dry-run}"
# L3 Plasma: export DISPATCH_MODE=greetd-ipc
# L3 PAM (VM): export FACE_LOGIN_PAM_PASSWORD='...'
# Python del repo — ver docs/deploy_endeavouros_vm.md §11.3.1 (recomendado vs PyInstaller)
# export PYTHON_BIN="/home/elioth/RP_PF_Login_Facial/.venv/bin/python"
# export PYTHONPATH="/home/elioth/RP_PF_Login_Facial/src"
unset FACE_LOGIN_BIN
# PyInstaller --onefile + TF suele fallar bajo greeter; ver doc §12
#if [[ -x "/opt/rp_face_login/bin/face-login-sim" ]]; then
#    export FACE_LOGIN_BIN="/opt/rp_face_login/bin/face-login-sim"
#fi
exec /opt/rp_face_login/bin/face-login-greeter "$@"
LAUNCHER
chmod 755 "${INSTALL_ROOT}/bin/greetd-face-login"

# Symlink opcional en PATH
install -d /usr/local/bin
ln -sf "${INSTALL_ROOT}/bin/greetd-face-login" /usr/local/bin/greetd-face-login

# Permisos: usuario greeter debe escribir logs y capturas
if id greeter &>/dev/null; then
    chown -R greeter:greeter "${INSTALL_ROOT}/capturas" "${INSTALL_ROOT}/logs" "${INSTALL_ROOT}/reports"
    chmod 775 "${INSTALL_ROOT}/capturas" "${INSTALL_ROOT}/logs" "${INSTALL_ROOT}/reports"
fi
chmod -R go+rX "${INSTALL_ROOT}"

echo ""
echo "Instalado:"
echo "  ${INSTALL_ROOT}/bin/greetd-face-login"
echo "  /usr/local/bin/greetd-face-login -> ..."
echo ""
echo "Siguiente paso: editar ${INSTALL_ROOT}/bin/greetd-face-login (§11.3.1) y greetd."
