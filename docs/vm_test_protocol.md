# Protocolo de pruebas en máquina virtual (VM)

> **Objetivo:** validar la integración `face-login-greeter` + greetd **sin
> arriesgar** el login gráfico de la máquina principal. Todo lo riesgoso se
> prueba primero en una VM con snapshots y plan de rollback.

Regla de oro: **nada de tocar `/etc` ni reemplazar SDDM en la máquina real**
hasta que el flujo completo funcione en la VM.

---

## 0. Resumen de fases

| Fase | Dónde | Riesgo | Toca `/etc`/greetd |
|---|---|---|---|
| A. Smoke test del pipeline | Máquina real o VM | Nulo | No |
| B. Greeter en `dry-run` | VM | Bajo | No |
| C. Greeter en `command` | VM | Bajo | No |
| D. Integración real con greetd | VM | **Alto** | Sí |
| E. Despliegue en máquina principal | Real | **Muy alto** | Sí (con rollback) |

Las fases A–C **no** modifican el sistema y son las que cubre este proyecto hoy.
D–E son el objetivo futuro y solo se intentan tras superar A–C en VM.

---

## 1. Preparar la VM

1. Crea una VM (p. ej. **virt-manager/QEMU-KVM**, VirtualBox o GNOME Boxes) con
   la misma distro objetivo (Arch/KDE Plasma) que la máquina real.
2. Instala KDE Plasma y **mantén SDDM** como gestor por defecto (red de
   seguridad).
3. Crea los usuarios de prueba: `elioth`, `emmanuel` y `guest`.
4. **Cámara dentro de la VM:**
   - QEMU/virt-manager: añade un dispositivo *USB Host Device* (passthrough de la
     webcam) o un *video* emulado.
   - Alternativa sin webcam física: usa `v4l2loopback` + `ffmpeg` para inyectar
     un **video** como `/dev/video0`:
     ```bash
     sudo modprobe v4l2loopback
     ffmpeg -re -stream_loop -1 -i data/raw/elioth/video_elioth_luz_frontal.mp4 \
            -vf format=yuv420p -f v4l2 /dev/video0
     ```
5. **Toma un snapshot limpio** ("baseline KDE+SDDM OK"). Repetirás snapshots
   antes de cada cambio peligroso.

---

## 2. Fase A — Smoke test del pipeline (sin tocar el sistema)

Verifica que el paquete funciona end-to-end antes de pensar en greetd.

```bash
# Entorno con dependencias ML (Python 3.10–3.12)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"

pytest                                   # toda la suite debe pasar

# Datos -> dataset -> modelo
python -m rp_face_login.cli prepare-dataset --raw-dir data/faces --output-dir data/processed
python -m rp_face_login.cli train    --dataset-dir data/processed --output models/face_auth_model.keras
python -m rp_face_login.cli evaluate --dataset-dir data/processed/test --model models/face_auth_model.keras

# Login simulado (no inicia sesión real)
python -m rp_face_login.cli login-sim --output-dir ./capturas --model models/face_auth_model.keras \
       --save-decision reports/decision.json
cat reports/decision.json
```

Criterio de éxito: `login-sim` imprime `elioth | emmanuel | guest` de forma
coherente y `reports/decision.json` es válido.

---

## 3. Fase B — Greeter experimental en `dry-run`

Ejecuta el greeter de prototipo. **No** toca `/etc`, **no** instala greetd y
**no** inicia KDE.

```bash
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5
tail -n 20 logs/face-login.log
```

Criterio de éxito:
- El log muestra los 3 pasos (login-sim → selected_user → dispatch).
- El dispatcher imprime `[dry-run] Se despacharía sesión para '<user>': ...`.
- `executed=False` (no se ejecutó nada).

---

## 4. Fase C — Greeter en modo `command`

Comprueba que el dispatcher ejecuta el comando local configurado en
`configs/default.yaml` → `session_dispatch.users.<user>.command` (por defecto un
`echo`, **inofensivo**).

```bash
DISPATCH_MODE=command ./scripts/face-login-greeter.sh --duration 5
tail -n 20 logs/face-login.log
```

Criterio de éxito: el log muestra `[command] Sesión '<user>' -> rc=0: ...` con
`executed=True`. **Aún no** se inicia una sesión KDE real (el comando es solo un
`echo` mientras no lo cambies).

> Mantén los comandos como `echo ...` hasta haber validado todo. No pongas aún
> `startplasma-wayland` ni nada que abra sesión real fuera de la fase D.

---

## 5. Fase D — Integración real con greetd (solo VM, alto riesgo)

> **Snapshot obligatorio antes de empezar.** A partir de aquí sí se toca `/etc`.

1. Snapshot: "antes de greetd".
2. Instala greetd y un compositor mínimo (p. ej. `cage`) **en la VM**.
3. Configura `/etc/greetd/config.toml` para lanzar el greeter (ver
   `docs/greetd_integration.md`, §5). Empieza apuntando el `default_session` a un
   `agreety`/shell para confirmar que greetd arranca antes de meter el greeter
   facial.
4. **No borres SDDM.** Cambia el gestor activo con cuidado:
   ```bash
   sudo systemctl disable --now sddm
   sudo systemctl enable  --now greetd
   ```
5. Verifica `/etc/pam.d/greetd`: **no** elimines la autenticación PAM estándar.
   El reconocimiento facial aporta identidad; PAM autentica (ver §7 del doc de
   integración).
6. Reinicia la VM y valida que greetd arranca y permite iniciar Plasma.
7. Solo entonces, integra el greeter facial y, finalmente, comandos de sesión
   reales.

Criterio de éxito: la VM inicia sesión Plasma a través de greetd, con PAM activo
y con la posibilidad de fallback a contraseña.

---

## 6. Plan de rollback (imprescindible)

Ten esto preparado **antes** de la fase D:

1. **TTY de rescate:** `Ctrl+Alt+F3`, inicia sesión en consola.
2. **Volver a SDDM:**
   ```bash
   sudo systemctl disable --now greetd
   sudo systemctl enable  --now sddm
   sudo reboot
   ```
3. **Restaurar snapshot** de la VM si algo queda inconsistente.
4. Conserva copias de los archivos antes de editarlos:
   ```bash
   sudo cp /etc/greetd/config.toml{,.bak}
   sudo cp /etc/pam.d/greetd{,.bak}
   ```

---

## 7. Checklist de seguridad

```text
[ ] Snapshot limpio de la VM con KDE+SDDM funcionando.
[ ] Fases A–C superadas sin tocar /etc.
[ ] Webcam o /dev/video0 disponible en la VM.
[ ] SDDM permanece instalado durante toda la fase D.
[ ] /etc/pam.d/greetd conserva la autenticación PAM (no se salta).
[ ] No hay contraseñas hardcodeadas en scripts ni configs.
[ ] TTY de rescate verificado (Ctrl+Alt+F3).
[ ] Comando de rollback a SDDM probado.
[ ] Comandos de sesión en 'echo' hasta validar el flujo completo.
[ ] Máquina principal intacta hasta superar todo en VM.
```

---

## 8. Qué NO hacer

- ❌ Probar greetd directamente en la máquina principal.
- ❌ Eliminar SDDM antes de validar greetd en VM.
- ❌ Quitar líneas de PAM para "que funcione más rápido".
- ❌ Hardcodear contraseñas o usar el reconocimiento facial como **único** factor.
- ❌ Subir rostros, videos o modelos al repositorio (`data/`, `models/` están en
  `.gitignore`).
