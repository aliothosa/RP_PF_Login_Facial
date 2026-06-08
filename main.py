"""Shim de entrada para empaquetado con PyInstaller.

Permite construir un binario a partir del CLI modular del paquete.
"""

import sys
from pathlib import Path

# Permite ejecutar sin instalar el paquete (src-layout).
SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rp_face_login.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
