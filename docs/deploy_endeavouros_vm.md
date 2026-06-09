# Despliegue del login facial en VM con EndeavourOS

Guía paso a paso para desplegar y validar **RP_PF_Login_Facial** en una máquina
virtual con **EndeavourOS + KDE Plasma**, desde la instalación del SO hasta el
login simulado y, opcionalmente, la integración con **greetd**.

> **Alcance de seguridad:** las fases 1–6 no modifican el login real del sistema.
> La fase 7 (greetd) **sí** toca `/etc` y solo debe ejecutarse en la VM, nunca
> primero en la máquina principal. Ver también
> [`vm_test_protocol.md`](vm_test_protocol.md) y
> [`greetd_integration.md`](greetd_integration.md).

---

## Tabla de contenidos

1. [Visión general](#1-visión-general)
2. [Requisitos](#2-requisitos)
3. [Crear la VM](#3-crear-la-vm)
4. [Instalar EndeavourOS + KDE Plasma](#4-instalar-endeavouros--kde-plasma)
5. [Preparar el sistema (paquetes base)](#5-preparar-el-sistema-paquetes-base)
6. [Clonar el proyecto y entorno uv](#6-clonar-el-proyecto-y-entorno-uv)
7. [Datos y entrenamiento del modelo](#7-datos-y-entrenamiento-del-modelo)
8. [Validar el pipeline (fases A–C)](#8-validar-el-pipeline-fases-ac)
9. [Cámara en la VM](#9-cámara-en-la-vm)
10. [Binarios PyInstaller (opcional)](#10-binarios-pyinstaller-opcional)
11. [Integración con greetd (fase D, opcional)](#11-integración-con-greetd-fase-d-opcional)
12. [Solución de problemas](#12-solución-de-problemas)
13. [Rollback y snapshots](#13-rollback-y-snapshots)
14. [Checklist final](#14-checklist-final)

---

## 1. Visión general

El despliegue sigue este orden lógico:

```text
┌─────────────────────────────────────────────────────────────────┐
│  VM EndeavourOS + KDE + SDDM (red de seguridad)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   uv + Python 3.12        data/faces + train      /dev/video0 OK
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    login-sim / face-login-greeter
                                │
                    (opcional) greetd + greeter
```

| Fase | Qué valida | Toca `/etc` |
|---|---|---|
| 1–6 | Pipeline ML + login simulado | No |
| 7 | Login gráfico vía greetd | **Sí** |

---

## 2. Requisitos

### En el host (tu PC físico)

- Hypervisor: **QEMU/KVM + virt-manager** (recomendado), VirtualBox o GNOME Boxes.
- RAM libre: ≥ 8 GB (VM con 4–6 GB).
- Disco libre: ≥ 40 GB para la VM.
- ISO de EndeavourOS: [endeavouros.com/download](https://endeavouros.com/download/).
- Repositorio del proyecto clonado o copiado al host (para transferir datos/modelo).

### En la VM (EndeavourOS)

| Componente | Versión / nota |
|---|---|
| SO | EndeavourOS (Arch rolling, compatible con tu máquina real) |
| Escritorio | KDE Plasma |
| Gestor de login | **SDDM** (mantener hasta validar greetd) |
| Python (uv) | **3.10–3.12** (TensorFlow **no** funciona en 3.14) |
| Cámara | Webcam USB passthrough **o** `v4l2loopback` + video |
| GPU | Opcional; entrenamiento CPU es lento pero válido para demo |

### Artefactos que debes tener antes del login simulado

```text
models/face_auth_model.keras
models/class_indices.json
data/processed/          # generado por prepare-dataset (opcional pero recomendado)
```

Estos archivos **no** están en git (`.gitignore`). Debes generarlos en la VM o
copiarlos desde el host.

---

## 3. Crear la VM

### 3.1 virt-manager (QEMU/KVM)

1. Abre **Virtual Machine Manager** → *Create a new virtual machine*.
2. ISO: selecciona la imagen de EndeavourOS.
3. Recursos sugeridos:
   - **vCPU:** 2–4
   - **RAM:** 4096–6144 MB
   - **Disco:** 25–40 GB (virtio)
4. Red: NAT o bridge (NAT basta para instalar paquetes).
5. **Firmware:** UEFI si tu host lo usa (EndeavourOS lo soporta).
6. Antes del primer arranque post-instalación: **snapshot** `"baseline vacía"`.

### 3.2 Passthrough de webcam (recomendado si tienes webcam física)

En virt-manager → detalles de la VM → *Add Hardware* → **USB Host Device** →
elige tu webcam.

Comprueba dentro de la VM:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices   # paquete: v4l-utils
```

### 3.3 Shared folder (opcional)

Para copiar el repo y `data/` desde el host sin git clone completo:

- **virt-manager:** Filesystem passthrough (virtiofs) o `scp` desde el host.
- **Alternativa simple:** clona desde GitHub dentro de la VM y copia solo
  `data/` + `models/` con `scp`:

```bash
# Desde el host hacia la VM (ajusta IP/usuario)
scp -r data/ models/ usuario@IP_VM:~/Projects/PF_PAT/
```

---

## 4. Instalar EndeavourOS + KDE Plasma

### 4.1 Instalación gráfica (instalador Calamares)

1. Arranca la ISO → *Install EndeavourOS*.
2. Particionado: automático o manual (ext4/btrfs + swap opcional).
3. Usuario principal: p. ej. `elioth` (tu usuario de desarrollo en la VM).
4. **Desktop environment:** elige **KDE Plasma** si el instalador lo ofrece.

Si instalaste sin Plasma, añádelo después:

```bash
sudo pacman -S plasma-meta kde-applications sddm
sudo systemctl enable sddm
sudo reboot
```

### 4.2 Usuarios del sistema de prueba

Crea las cuentas que el proyecto reconoce (para pruebas de despacho futuro):

```bash
sudo useradd -m elioth
sudo useradd -m emmanuel
sudo passwd elioth
sudo passwd emmanuel
# 'guest' es solo identidad de rechazo en el ML, no hace falta crear usuario guest
```

### 4.3 Snapshot post-instalación

Nombre sugerido: **`EndeavourOS-KDE-SDDM-OK`**.

A partir de aquí, cualquier cambio en greetd/PAM debe ir precedido de otro snapshot.

---

## 5. Preparar el sistema (paquetes base)

Actualiza e instala dependencias de sistema. EndeavourOS usa **pacman**; para AUR
puedes usar `yay` (viene preinstalado en muchas ediciones) o instalarlo.

```bash
sudo pacman -Syu

# Build tools, git, cámara, OpenCV runtime deps
sudo pacman -S --needed \
  base-devel git \
  opencv v4l-utils \
  ffmpeg \
  cage                    # compositor mínimo (fase greetd opcional)

# uv (gestor de entorno Python)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Reinicia la shell o:
source "$HOME/.local/bin/env"   # según instalador uv
```

### 5.1 v4l2loopback (cámara virtual con video)

Si **no** pasaste webcam USB, instala el módulo desde AUR:

```bash
yay -S v4l2loopback-dkms
sudo modprobe v4l2loopback
```

Verifica:

```bash
lsmod | grep v4l2loopback
ls /dev/video*
```

---

## 6. Clonar el proyecto y entorno uv

### 6.1 Clonar

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone git@github.com:aliothosa/RP_PF_Login_Facial.git PF_PAT
cd PF_PAT
git checkout feature/fatial-recognition   # o la rama activa del proyecto
```

### 6.2 Fijar Python 3.12 con uv

TensorFlow requiere Python **3.10–3.12**. EndeavourOS puede tener 3.13+ en
system; **uv** resuelve esto aislando la versión:

```bash
cd ~/Projects/PF_PAT

# Instalar intérprete 3.12 y crear .venv del proyecto
uv python install 3.12
uv sync --extra dev --extra ml

# Verificar
uv run python --version          # Python 3.12.x
uv run python -c "import tensorflow as tf; print(tf.__version__)"
uv run python -c "import cv2; print(cv2.__version__)"
```

Si `import tensorflow` falla:

- Confirma que usaste `--extra ml`.
- No uses el `python` del sistema (3.14); siempre **`uv run`**.

### 6.3 Validar configuración

```bash
uv run python -m rp_face_login.cli check-config
uv run pytest
```

---

## 7. Datos y entrenamiento del modelo

### 7.1 Estructura de datos esperada

```text
data/
├── raw/
│   ├── elioth/      *.mp4
│   └── emmanuel/    *.mp4
├── faces/
│   ├── elioth/
│   │   ├── luz_frontal/
│   │   └── luz_ambiental/
│   └── emmanuel/
│       ├── luz_frontal/
│       └── luz_ambiental/
└── processed/       # se genera con prepare-dataset
```

**Privacidad:** estos directorios están en `.gitignore`. Transfiérelos desde el
host o genera rostros en la VM; no los subas al repositorio.

### 7.2 Preparar dataset procesado

```bash
cd ~/Projects/PF_PAT

uv run python -m rp_face_login.cli prepare-dataset \
  --raw-dir data/faces \
  --output-dir data/processed \
  --seed 42
```

Revisa el resumen impreso (aceptadas/descartadas por clase) y
`data/processed/dataset_stats.json`.

### 7.3 Entrenar

```bash
uv run python -m rp_face_login.cli train \
  --dataset-dir data/processed \
  --output models/face_auth_model.keras \
  --epochs 10 \
  --batch-size 32
```

Salida esperada:

```text
models/face_auth_model.keras
models/class_indices.json
models/history.json
```

Entrenamiento en CPU dentro de la VM puede tardar **varios minutos u horas**.
Para acelerar, entrena en el host y copia `models/` a la VM.

### 7.4 Evaluar

```bash
uv run python -m rp_face_login.cli evaluate \
  --dataset-dir data/processed/test \
  --model models/face_auth_model.keras
```

Revisa `reports/evaluation.json` y `reports/confusion_matrix.png`.

---

## 8. Validar el pipeline (fases A–C)

### Fase A — Login simulado directo

Con cámara disponible (`/dev/video0`):

```bash
cd ~/Projects/PF_PAT

uv run python -m rp_face_login.cli login-sim \
  --output-dir ./capturas \
  --model models/face_auth_model.keras \
  --save-decision reports/decision.json \
  --duration 5
```

Salida esperada (demo académica):

```text
============================================
  LOGIN FACIAL (SIMULACIÓN)
============================================
  Usuario seleccionado : elioth
  Aceptado             : True
  ...
============================================
  (Demostración: no se inicia ninguna sesión real.)
```

Inspecciona la decisión:

```bash
cat reports/decision.json | jq .
unzip -l capturas/login_*.zip | head
```

### Fase B — Greeter en dry-run

```bash
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5
tail -n 25 logs/face-login.log
```

Debe mostrar: login-sim OK → `selected_user` → `[dry-run] Se despacharía sesión...`

### Fase C — Greeter en command

```bash
DISPATCH_MODE=command ./scripts/face-login-greeter.sh --duration 5
```

Por defecto los comandos en `configs/default.yaml` son `echo start <user>` (inofensivos).

### Ajustar umbrales (si siempre cae en guest)

Edita `configs/default.yaml`:

```yaml
decision:
  min_valid_frames: 30        # bajar temporalmente p. ej. a 15 en VM lenta
  confidence_threshold: 0.80  # bajar con cuidado solo para depuración
  margin_threshold: 0.25
```

> Bajar umbrales en producción aumenta falsos positivos. Documenta cualquier
> cambio para la defensa académica.

---

## 9. Cámara en la VM

### Opción A — Webcam USB (virt-manager)

1. VM apagada → Add USB Host Device → tu webcam.
2. Arranca la VM → `v4l2-ctl --list-devices`.
3. Prueba captura rápida:

```bash
uv run python -m rp_face_login.cli capture \
  --name test --output-dir ./capturas --duration 3
```

### Opción B — v4l2loopback + video del dataset

Útil para demos reproducibles sin depender de iluminación real.

Terminal 1 (inyecta video como cámara):

```bash
cd ~/Projects/PF_PAT
sudo modprobe v4l2loopback

ffmpeg -re -stream_loop -1 \
  -i data/raw/elioth/video_elioth_luz_frontal.mp4 \
  -vf format=yuv420p -f v4l2 /dev/video0
```

Terminal 2 (login simulado):

```bash
uv run python -m rp_face_login.cli login-sim \
  --model models/face_auth_model.keras \
  --output-dir ./capturas
```

Para simular a **emmanuel**, cambia el video de entrada.

### Permisos de cámara

El usuario debe poder leer `/dev/video0`:

```bash
groups $USER | grep -E 'video|wheel'
sudo usermod -aG video "$USER"
# cerrar sesión y volver a entrar
```

El usuario **`greeter`** (fase greetd) también necesita grupo `video`:

```bash
sudo useradd -M -G video greeter   # si no existe aún
```

---

## 10. Binarios PyInstaller (opcional)

Genera ejecutables autocontenidos (config + Haar embebidos; **modelo aparte**):

```bash
cd ~/Projects/PF_PAT

./scripts/build_pyinstaller.sh capture
./scripts/build_pyinstaller.sh login-sim    # incluye TensorFlow; build largo (~GB)

./dist/face-login-capture --help
./dist/face-login-sim \
  --output-dir ./capturas \
  --model models/face_auth_model.keras \
  --save-decision reports/decision.json
```

Instalar el binario en la VM (sin clonar repo completo):

```bash
sudo install -m 755 dist/face-login-sim /usr/local/bin/
sudo install -m 755 scripts/face-login-greeter.sh /usr/local/bin/
# Copiar configs y models a rutas conocidas
sudo mkdir -p /opt/rp_face_login/{configs,models}
sudo cp configs/default.yaml /opt/rp_face_login/configs/
sudo cp models/* /opt/rp_face_login/models/
```

---

## 11. Integración con greetd (fase D, opcional)

> **Snapshot obligatorio:** `antes-de-greetd`.  
> **No elimines SDDM.** Mantén rollback probado.

### 11.1 Instalar greetd

```bash
sudo pacman -S greetd
sudo useradd -M -G video greeter 2>/dev/null || true
sudo mkdir -p /etc/greetd
sudo chmod -R go+rX /etc/greetd
```

### 11.2 Instalar el greeter en el sistema

```bash
cd ~/Projects/PF_PAT
sudo install -m 755 scripts/face-login-greeter.sh /usr/local/bin/face-login-greeter
sudo mkdir -p /opt/rp_face_login
sudo cp -r configs models /opt/rp_face_login/
# Asegura que el script encuentre uv y el repo, o empaqueta binarios PyInstaller
```

Wrapper recomendado `/usr/local/bin/face-login-greeter`:

```bash
#!/usr/bin/env bash
export CONFIG=/opt/rp_face_login/configs/default.yaml
export MODEL=/opt/rp_face_login/models/face_auth_model.keras
export DISPATCH_MODE=dry-run
cd /opt/rp_face_login
exec /home/TU_USUARIO/Projects/PF_PAT/scripts/face-login-greeter.sh "$@"
```

(Ajusta rutas; en producción usarías rutas bajo `/opt` y binarios PyInstaller.)

### 11.3 Configuración conceptual de greetd

Backup primero:

```bash
sudo cp /etc/greetd/config.toml /etc/greetd/config.toml.bak 2>/dev/null || true
```

Ejemplo **solo para VM** (`/etc/greetd/config.toml`):

```toml
[terminal]
vt = 1

[default_session]
command = "cage -s -- /usr/local/bin/face-login-greeter --duration 5"
user = "greeter"
```

PAM (`/etc/pam.d/greetd`) — **no quites** la autenticación estándar:

```text
#%PAM-1.0
auth       include      system-local-login
account    include      system-local-login
session    include      system-local-login
```

### 11.4 Cambiar gestor de login (solo VM)

```bash
sudo systemctl disable --now sddm
sudo systemctl enable --now greetd
sudo reboot
```

TTY de rescate: `Ctrl+Alt+F3`.

### 11.5 Qué esperar hoy

En la fase actual del proyecto, el greeter:

- Ejecuta reconocimiento facial y obtiene `selected_user`.
- Despacha en **`dry-run`** o **`echo`** (no abre Plasma automáticamente).
- **No sustituye PAM:** la autenticación real sigue siendo responsabilidad de
  greetd + contraseña.

La integración "facial → sesión Plasma sin contraseña" requiere un módulo PAM o
un greeter que hable `greetd-ipc` **después** de autenticar. Ver
[`greetd_integration.md`](greetd_integration.md) §8.

---

## 12. Solución de problemas

### `TensorFlow no está instalado`

```text
[rp_face_login] TensorFlow no está instalado...
```

**Causa:** ejecutaste con Python 3.14 o sin extra `[ml]`.  
**Solución:**

```bash
uv sync --extra dev --extra ml
uv run python -c "import tensorflow"
# Siempre: uv run python -m rp_face_login.cli ...
```

### `No se pudo abrir la cámara con índice 0`

**Causa:** no hay `/dev/video0` en la VM.  
**Solución:** USB passthrough, `v4l2loopback` + `ffmpeg`, o `--camera-index 1`.

### `login-sim` captura bien pero siempre `guest`

**Causas posibles:**

- Modelo no entrenado o mal copiado (`models/` vacío).
- Pocos frames válidos (`min_valid_frames`).
- Umbrales demasiado estrictos.
- Persona no reconocida o iluminación muy distinta al entrenamiento.

**Diagnóstico:**

```bash
uv run python -m rp_face_login.cli predict-zip \
  --zip capturas/login_*.zip \
  --model models/face_auth_model.keras \
  --save-json reports/predictions.json
cat reports/decision.json
```

### Greeter falla con `uv: command not found`

Instala uv en el PATH del usuario que ejecuta el greeter, o usa binarios
PyInstaller en `/usr/local/bin`.

### greetd no arranca / pantalla negra

1. `Ctrl+Alt+F3` → login por consola.
2. `journalctl -u greetd -b --no-pager`
3. Rollback a SDDM (§13).

---

## 13. Rollback y snapshots

### Volver a SDDM (sin snapshot)

```bash
sudo systemctl disable --now greetd
sudo systemctl enable --now sddm
sudo reboot
```

### Restaurar configs

```bash
sudo cp /etc/greetd/config.toml.bak /etc/greetd/config.toml
sudo cp /etc/pam.d/greetd.bak /etc/pam.d/greetd
sudo systemctl restart greetd
```

### Snapshots recomendados (virt-manager)

| Nombre | Cuándo |
|---|---|
| `EndeavourOS-KDE-SDDM-OK` | Post-instalación |
| `pipeline-A-OK` | Tras login-sim exitoso |
| `antes-de-greetd` | Antes de tocar `/etc/greetd` |
| `greetd-dry-run-OK` | Tras validar greeter en VM |

---

## 14. Checklist final

### Pipeline (obligatorio antes de greetd)

```text
[ ] VM EndeavourOS + KDE + SDDM operativa
[ ] Snapshot baseline tomado
[ ] uv + Python 3.12 + TensorFlow importan OK
[ ] data/faces/ presente (copiado o generado)
[ ] prepare-dataset completado
[ ] models/face_auth_model.keras existe
[ ] evaluate genera reports/
[ ] login-sim imprime elioth|emmanuel|guest coherente
[ ] face-login-greeter.sh dry-run OK
[ ] /dev/video0 funciona (USB o v4l2loopback)
```

### greetd (opcional, solo VM)

```text
[ ] Snapshot "antes-de-greetd"
[ ] SDDM sigue instalado (rollback probado)
[ ] greetd + cage instalados
[ ] /etc/pam.d/greetd intacto (PAM no saltado)
[ ] greeter en dry-run antes de command real
[ ] TTY Ctrl+Alt+F3 verificado
[ ] Máquina principal NO modificada
```

---

## Referencias cruzadas

| Documento | Contenido |
|---|---|
| [`README.md`](../README.md) | Arquitectura técnica, comandos CLI, reglas de decisión |
| [`vm_test_protocol.md`](vm_test_protocol.md) | Fases A–D resumidas, checklist de seguridad |
| [`greetd_integration.md`](greetd_integration.md) | PAM vs identidad, opciones de integración |
| [`plan.md`](plan.md) | Plan completo del proyecto académico |

---

## Resumen de comandos (copiar/pegar)

```bash
# --- Setup ---
uv python install 3.12
uv sync --extra dev --extra ml
uv run pytest

# --- ML pipeline ---
uv run python -m rp_face_login.cli prepare-dataset --raw-dir data/faces --output-dir data/processed
uv run python -m rp_face_login.cli train --dataset-dir data/processed --output models/face_auth_model.keras
uv run python -m rp_face_login.cli evaluate --dataset-dir data/processed/test --model models/face_auth_model.keras

# --- Login simulado ---
uv run python -m rp_face_login.cli login-sim --output-dir ./capturas --model models/face_auth_model.keras --save-decision reports/decision.json

# --- Greeter ---
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5

# --- Rollback greetd ---
sudo systemctl disable --now greetd && sudo systemctl enable --now sddm && sudo reboot
```
