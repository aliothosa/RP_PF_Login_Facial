"""Punto de entrada PyInstaller: subcomando ``login-sim``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rp_face_login.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["login-sim", *sys.argv[1:]]))
