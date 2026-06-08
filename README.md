# RP_PF_Login_Facial

Autenticación / identificación facial **1:N** para login en KDE Plasma, reducida a
dos identidades conocidas (`elioth`, `emmanuel`) con un mecanismo de **rechazo**
(`guest`). Ver el plan completo en [`docs/plan.md`](docs/plan.md).

## Estructura del proyecto

```text
src/rp_face_login/        # Paquete modular
  cli.py                  # Punto de entrada (python -m rp_face_login.cli)
  config.py               # Carga de configs/default.yaml
  acquisition/            # Captura temporal de login + ZIP
  vision/                 # Detección facial, preprocesamiento, calidad
  training/               # Dataset, transfer learning, evaluación
  inference/              # Carga de modelo, predicción batch, agregación temporal
  decision/               # Política de aceptación / rechazo (guest)
  session/                # Despacho de sesión (dry-run / command)
configs/default.yaml      # Configuración central
scripts/                  # Utilidades (build, captura, etc.)
tests/                    # Pruebas unitarias
docs/legacy/              # Scripts monolíticos originales (solo referencia)
data/                     # Datos locales (NO versionado, ver más abajo)
```

## Uso (estructura base)

```bash
# Sin instalar el paquete (src-layout):
PYTHONPATH=src python -m rp_face_login.cli --help

# O instalándolo en modo editable:
pip install -e .
rp-face-login --help

# Validar la configuración:
PYTHONPATH=src python -m rp_face_login.cli check-config
```

> En esta fase los subcomandos (`capture`, `prepare-dataset`, `train`,
> `evaluate`, `predict-zip`, `login-sim`) son *placeholders*: la estructura
> modular está lista y la lógica se irá implementando por fases.

## Organización de datos (`data/`, no versionado)

```text
data/
├── raw/                  # Videos fuente, agrupados por identidad
│   ├── elioth/
│   └── emmanuel/
├── faces/                # Rostros recortados (fuente del dataset) por clase y condición
│   ├── elioth/{luz_ambiental, luz_frontal}/
│   └── emmanuel/{luz_ambiental, luz_frontal}/
└── processed/            # Reservado: split train/val/test generado por 'prepare-dataset'
```

**Privacidad:** `data/`, `models/`, `reports/` y `logs/` están en `.gitignore`.
No subas rostros reales ni videos biométricos al repositorio.

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```
