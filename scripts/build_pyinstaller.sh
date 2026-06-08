#!/usr/bin/env bash
# Build reproducible del binario con PyInstaller.
# Detecta dinámicamente la ruta del Haar Cascade (sin rutas absolutas locales).
set -euo pipefail

CASCADE=$(python - <<'PY'
import cv2, os
print(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
PY
)

pyinstaller \
  --onefile \
  --clean \
  --name face-login \
  --add-data "${CASCADE}:cv2/data" \
  main.py
