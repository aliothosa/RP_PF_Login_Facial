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

## 11. Integración KDE Plasma + greetd (fase D)

> **Snapshot obligatorio:** `antes-de-greetd`.  
> **No elimines SDDM** hasta validar rollback (§13).

Esta sección describe **qué aplicar en el sistema** para integrar el login facial
con **KDE Plasma** vía **greetd** en EndeavourOS, qué **limitaciones** tiene el
código actual y qué **cambios en scripts** ya están preparados.

### 11.0 Niveles de integración (elige uno)

| Nivel | Objetivo | Inicia Plasma | PAM | Cambios en `/etc` |
|---|---|---|---|---|
| **L1 — Validación** | greetd ejecuta ML al arranque; dispatcher `dry-run` | No | Intacto | Sí |
| **L2 — Demo híbrida** | Reconocimiento facial + **tuigreet/ReGreet** con contraseña | Sí | Intacto | Sí |
| **L3 — Objetivo** | Sesión Plasma tras rostro vía **greetd-ipc** o módulo PAM | Sí | Integrado | Sí (avanzado) |

**Estado del repositorio hoy:** L1 listo (`dry-run`). L2 documentado (tuigreet +
contraseña). **L3 implementado** en `session/greetd_ipc.py` + modo
`greetd-ipc` del dispatcher (PAM + `startplasma-wayland` vía IPC oficial).

```text
Lo que NO funciona (modo command legacy):

  greetd → cage → face-login-greeter.sh → subprocess("startplasma-wayland")
                                              ↑
                         No pasa por greetd-ipc ni PAM → no hay sesión KDE

Lo que SÍ funciona (modo greetd-ipc, L3):

  greetd → face-login-greeter → login-sim → selected_user
         → create_session (PAM) → start_session(startplasma-wayland)
         → greeter termina → greetd abre Plasma
```

La identidad facial (`selected_user`) y la autenticación del SO (PAM/greetd)
siguen siendo capas distintas. Ver [`greetd_integration.md`](greetd_integration.md) §7.

---

### 11.1 Paquetes del sistema (EndeavourOS / Arch)

```bash
sudo pacman -Syu
sudo pacman -S --needed \
  greetd cage \
  plasma-meta sddm \
  pipewire pipewire-pulse wireplumber \
  v4l-utils ffmpeg \
  jq

# Greeter gráfico con greetd-ipc (nivel L2 demo)
sudo pacman -S greetd-tuigreet    # o: yay -S greetd-regreet
```

Comprueba sesiones Plasma:

```bash
ls /usr/share/wayland-sessions/plasma.desktop
ls /usr/share/xsessions/plasma.desktop
grep ^Exec= /usr/share/wayland-sessions/plasma.desktop
# Típico: /usr/bin/startplasma-wayland
```

---

### 11.2 Usuario `greeter` y permisos de cámara

greetd ejecuta el greeter como usuario dedicado (no root):

```bash
sudo useradd -M -G video,input greeter 2>/dev/null || \
  sudo usermod -aG video,input greeter

# Comprobar acceso a cámara COMO greeter (tras passthrough USB o v4l2loopback)
sudo -u greeter v4l2-ctl --list-devices
sudo -u greeter test -r /dev/video0 && echo "OK video0"
```

Si la cámara solo funciona para tu usuario, el greeter fallará en greetd aunque
`login-sim` funcione en tu sesión.

---

### 11.3 Instalar artefactos del proyecto en `/opt`

Desde el repo (con modelo entrenado):

```bash
cd ~/Projects/PF_PAT
sudo ./scripts/install-vm-greeter.sh
```

Esto crea:

```text
/opt/rp_face_login/
├── bin/greetd-face-login      # launcher con env vars fijas
├── bin/face-login-greeter     # script principal
├── configs/default.yaml
├── models/face_auth_model.keras
├── capturas/   logs/   reports/
/usr/local/bin/greetd-face-login -> /opt/rp_face_login/bin/greetd-face-login
```

**Opcional (recomendado en VM):** binario PyInstaller para no depender de `uv`
en el PATH del usuario `greeter`:

```bash
./scripts/build_pyinstaller.sh login-sim
sudo cp dist/face-login-sim /opt/rp_face_login/bin/
sudo chown root:greeter /opt/rp_face_login/bin/face-login-sim
sudo chmod 750 /opt/rp_face_login/bin/face-login-sim
```

El launcher detecta `FACE_LOGIN_BIN` automáticamente si existe ese binario.

Plantilla de config para VM: [`configs/greetd-vm.example.yaml`](../configs/greetd-vm.example.yaml).

---

### 11.4 Archivos de sistema a crear o editar

#### A) `/etc/greetd/config.toml` (backup primero)

```bash
sudo cp /etc/greetd/config.toml /etc/greetd/config.toml.bak 2>/dev/null || true
sudo tee /etc/greetd/config.toml >/dev/null <<'EOF'
[terminal]
vt = 1

# Nivel L1 (validación): dry-run — no abre Plasma
# command = "cage -s -- /opt/rp_face_login/bin/greetd-face-login --duration 5"

# Nivel L3 (Plasma vía greetd-ipc): greeter habla PAM + start_session
[default_session]
command = "/opt/rp_face_login/bin/greetd-face-login --duration 5"
user = "greeter"
EOF
```

Variables en el launcher (`/opt/rp_face_login/bin/greetd-face-login`):

```bash
export DISPATCH_MODE=greetd-ipc          # L3 (default tras install-vm-greeter.sh)
# export DISPATCH_MODE=dry-run           # L1
export FACE_LOGIN_PAM_PASSWORD=          # opcional: contraseña PAM sin prompt TTY
```

En `configs/default.yaml` (o `greetd-vm.example.yaml`):

```yaml
session_dispatch:
  mode: greetd-ipc
  greetd_ipc:
    default_cmd: ["/usr/bin/startplasma-wayland"]
    password_env: FACE_LOGIN_PAM_PASSWORD
    prompt_password: true
  users:
    elioth:
      command: "/usr/bin/startplasma-wayland"
```

#### B) `/etc/pam.d/greetd` — **no saltar PAM**

```bash
sudo cp /etc/pam.d/greetd /etc/pam.d/greetd.bak 2>/dev/null || true
```

Contenido mínimo (EndeavourOS / Arch):

```text
#%PAM-1.0
auth       include      system-local-login
account    include      system-local-login
password   include      system-local-login
session    include      system-local-login
```

No elimines estas líneas para "acelerar" el login facial.

#### C) Habilitar greetd (solo VM; SDDM sigue instalado)

```bash
sudo systemctl disable --now sddm
sudo systemctl enable --now greetd
sudo reboot
```

TTY de rescate: `Ctrl+Alt+F3`.

#### D) Logs tras el arranque

```bash
journalctl -u greetd -b --no-pager
sudo tail -f /opt/rp_face_login/logs/face-login.log
```

Códigos de salida del greeter:

| Exit | Significado |
|---|---|
| 0 | Usuario aceptado; despacho completado |
| 1 | Error (cámara, TF, modelo) |
| 2 | Rechazo a `guest`; greetd reinicia el greeter |

---

### 11.5 Nivel L2 — Demo KDE con contraseña (Plasma real + PAM)

Para **abrir Plasma en la VM** sin implementar greetd-ipc todavía, usa un greeter
gráfico estándar **después** del reconocimiento facial, o en paralelo:

**Opción recomendada en VM:** mantener **SDDM** para la demo de Plasma y usar
`face-login-greeter.sh` solo en consola (fases A–C). greetd queda como L1.

**Opción híbrida con greetd:**

```toml
# /etc/greetd/config.toml — L2 demo (contraseña obligatoria vía tuigreet)
[default_session]
command = "cage -s -- tuigreet"
user = "greeter"
```

Flujo manual de demo académica:

1. Ejecuta `./scripts/face-login-greeter.sh` en TTY o antes de iniciar sesión.
2. Anota `selected_user` en el log.
3. Inicia sesión en **tuigreet/ReGreet/SDDM** con ese usuario y **contraseña PAM**.

Esto demuestra identificación + autenticación separadas (defendible ante profesor).

---

### 11.6 Nivel L3 — Plasma vía greetd-ipc (implementado)

Tras reconocimiento facial, el greeter usa el protocolo oficial de greetd:

1. `create_session` con `selected_user` (`elioth` / `emmanuel`).
2. Bucle PAM: responde `post_auth_message_response` (contraseña si PAM la pide).
3. `start_session` con `cmd: ["/usr/bin/startplasma-wayland"]`.
4. El proceso greeter **termina**; greetd abre Plasma.

**Código:** `src/rp_face_login/session/greetd_ipc.py`, modo `greetd-ipc` en
`session/dispatcher.py`.

#### Contraseña PAM (el rostro no sustituye PAM)

El modelo solo sugiere el **username**. PAM sigue autenticando. Opciones en VM:

| Método | Uso |
|---|---|
| `prompt_password: true` en config | Lee contraseña desde `/dev/tty` tras la captura |
| `FACE_LOGIN_PAM_PASSWORD` en env | Contraseña fija para pruebas (no producción) |
| Módulo PAM facial (Howdy) | Factor `auth sufficient` + fallback contraseña |
| `pam_permit.so` solo en VM | Demo académica sin contraseña (inseguro) |

Ejemplo PAM de prueba **solo VM** (antes de `system-local-login`):

```text
# /etc/pam.d/greetd — SOLO laboratorio, NO producción
auth       sufficient   pam_permit.so
auth       include      system-local-login
...
```

#### Pasos L3 en la VM

```bash
sudo ./scripts/install-vm-greeter.sh
sudo cp configs/greetd-vm.example.yaml /opt/rp_face_login/configs/default.yaml
# Editar /etc/greetd/config.toml (§11.4, L3 sin cage obligatorio)
sudo systemctl restart greetd
```

Tras login facial aceptado, si PAM pide contraseña verás el prompt en TTY o usa
`export FACE_LOGIN_PAM_PASSWORD=tu_clave` en el launcher.

---

### 11.7 Cambios aplicados en scripts (este repo)

| Script | Cambio | Motivo |
|---|---|---|
| `face-login-greeter.sh` | `REPO_ROOT`, `LOG_FILE`, `FACE_LOGIN_BIN`, `CONFIG`, `MODEL` por env | Funcionar instalado en `/opt` y como usuario `greeter` |
| `face-login-greeter.sh` | Fallback de log a `/tmp` si no hay permiso de escritura | Evitar fallo silencioso en greetd |
| `face-login-greeter.sh` | Exit codes 0/1/2 (guest = 2) | greetd puede reiniciar greeter |
| `face-login-greeter.sh` | Soporte binario PyInstaller `face-login-sim` | No requerir `uv` en PATH de greeter |
| `face-login-greeter.sh` | `with_mode()` preserva opciones greetd-ipc | Override `DISPATCH_MODE` sin perder config |
| `session/greetd_ipc.py` | **Nuevo** — cliente greetd-ipc | L3: PAM + `start_session` |
| `session/dispatcher.py` | Modo `greetd-ipc` | Despacho oficial hacia Plasma |
| `install-vm-greeter.sh` | **Nuevo** — instala en `/opt/rp_face_login` | Despliegue reproducible en VM |
| `configs/greetd-vm.example.yaml` | Rutas `/opt/...`, modo `greetd-ipc` | Config lista para L3 |
| `tests/test_greetd_ipc.py` | Mock de socket greetd | Regresión del protocolo |

**Sin cambios necesarios:** `build_pyinstaller.sh`, `login_sim.py`.

---

### 11.8 Checklist integración KDE + greetd (VM)

```text
[ ] Snapshot "antes-de-greetd"
[ ] greetd, cage, plasma-meta, sddm instalados (SDDM no borrado)
[ ] Usuario greeter en grupos video,input
[ ] sudo -u greeter puede leer /dev/video0
[ ] models/ copiado a /opt/rp_face_login/models/
[ ] sudo ./scripts/install-vm-greeter.sh
[ ] (opcional) face-login-sim PyInstaller en /opt/rp_face_login/bin/
[ ] configs/default.yaml con mode: greetd-ipc (o greetd-vm.example.yaml)
[ ] /etc/greetd/config.toml → greetd-face-login (L3) o dry-run (L1)
[ ] Contraseña PAM: prompt_password o FACE_LOGIN_PAM_PASSWORD definido
[ ] /etc/pam.d/greetd incluye system-local-login (PAM intacto)
[ ] systemctl enable greetd; reboot
[ ] Log en /opt/rp_face_login/logs/face-login.log muestra selected_user + greetd-ipc
[ ] Rollback a SDDM probado (§13)
[ ] Para demo solo contraseña sin rostro: usar L2 (tuigreet) o SDDM
```

---

### 11.9 Qué esperar según el nivel

| Nivel | Al arrancar la VM |
|---|---|
| **L1** | Pantalla negra/cage breve → log con `selected_user` → `[dry-run]` → greeter termina; **no entras a Plasma** automáticamente |
| **L2** | tuigreet pide usuario/contraseña → Plasma si PAM OK; ML puede correr aparte |
| **L3** | Rostro aceptado → greetd-ipc → PAM → Plasma Wayland (contraseña si PAM la exige) |

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

### Greeter falla con `uv: command not found` (usuario greeter)

El usuario `greeter` no tiene `uv` en PATH. Soluciones:

1. **Recomendado:** PyInstaller + copiar a `/opt/rp_face_login/bin/face-login-sim`
2. Instalar con `sudo ./scripts/install-vm-greeter.sh` (usa launcher con env fijos)
3. Probar manualmente: `sudo -u greeter /opt/rp_face_login/bin/greetd-face-login --duration 5`

### greetd reinicia el greeter en bucle (exit code 2)

**Causa:** reconocimiento rechazó a `guest` (exit 2). Comportamiento esperado en L1.  
**Solución:** ajustar umbrales, iluminación, modelo; o usar v4l2loopback con video conocido.

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
[ ] greetd + cage + plasma-meta instalados
[ ] install-vm-greeter.sh ejecutado (/opt/rp_face_login)
[ ] greeter user: grupos video,input; /dev/video0 OK con sudo -u greeter
[ ] /etc/greetd/config.toml → greetd-face-login (L1 dry-run)
[ ] /etc/pam.d/greetd intacto (PAM no saltado)
[ ] journalctl -u greetd sin errores; log en /opt/rp_face_login/logs/
[ ] Entendido: L1 no abre Plasma; L2 requiere tuigreet/SDDM + contraseña
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

# --- Greeter (consola, fases B–C) ---
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5

# --- Instalar en /opt para greetd (VM) ---
sudo ./scripts/install-vm-greeter.sh
# Luego editar /etc/greetd/config.toml (ver §11.4) y enable greetd

# --- Rollback greetd ---
sudo systemctl disable --now greetd && sudo systemctl enable --now sddm && sudo reboot
```
