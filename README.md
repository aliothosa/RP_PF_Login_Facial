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
packaging/                # Entry points para PyInstaller
scripts/                  # Utilidades (build, greeter, etc.)
tests/                    # Pruebas unitarias
docs/legacy/              # Scripts monolíticos originales (solo referencia)
data/                     # Datos locales (NO versionado)
```

## Entorno con uv

Este proyecto usa **[uv](https://docs.astral.sh/uv/)** para el entorno virtual.
TensorFlow requiere **Python 3.10–3.12** (no 3.14).

```bash
# Crear entorno e instalar dependencias de desarrollo
uv sync --extra dev

# Con ML (entrenamiento, inferencia, login-sim)
uv sync --extra dev --extra ml

# Ejecutar CLI
uv run python -m rp_face_login.cli --help
uv run python -m rp_face_login.cli check-config
```

## Uso del CLI

```bash
# Captura temporal de login (sin vista previa) -> ZIP
uv run python -m rp_face_login.cli capture --name elioth --output-dir ./capturas --duration 5

# Login simulado completo (requiere modelo entrenado + extra [ml])
uv run python -m rp_face_login.cli login-sim --output-dir ./capturas --model models/face_auth_model.keras

# Greeter experimental (dry-run por defecto en config)
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5
```

## Empaquetado con PyInstaller

Build reproducible **sin `.spec` versionado** ni rutas absolutas locales.
El Haar Cascade y `configs/default.yaml` se incluyen con `--add-data`.

```bash
# Binario solo de captura (OpenCV; más liviano)
./scripts/build_pyinstaller.sh capture

# Binario de login simulado (incluye TensorFlow; más pesado)
./scripts/build_pyinstaller.sh login-sim
```

Salida en `dist/`:

| Binario | Equivalente CLI |
|---|---|
| `dist/face-login-capture` | `capture ...` |
| `dist/face-login-sim` | `login-sim ...` |

### Cómo ejecutar los binarios

Los binarios embeben `configs/default.yaml` y el Haar Cascade. Aun así, el
**modelo entrenado** (`models/face_auth_model.keras`) y los directorios de
salida deben existir en disco (no se empaquetan por tamaño/privacidad).

Desde la **raíz del repositorio**:

```bash
# Captura -> ZIP en ./capturas/
./dist/face-login-capture --name elioth --output-dir ./capturas --duration 5

# Login simulado (requiere models/ entrenado previamente)
./dist/face-login-sim --output-dir ./capturas --model models/face_auth_model.keras \
    --save-decision reports/decision.json

# Sobrescribir config externa si hace falta
./dist/face-login-capture --config /ruta/a/custom.yaml --camera-index 0
```

Notas:
- `build/`, `dist/` y `*.spec` generados están en `.gitignore`.
- `login-sim` requiere haber construido con `./scripts/build_pyinstaller.sh login-sim`
  (extra `[ml]`) y tener el modelo en `models/`.
- Para depuración: `./dist/face-login-capture --help`.

## Organización de datos (`data/`, no versionado)

```text
data/
├── raw/                  # Videos fuente por identidad
├── faces/                # Rostros recortados por clase/condición
└── processed/            # Split train/val/test (prepare-dataset)
```

**Privacidad:** `data/`, `models/`, `reports/` y `logs/` están en `.gitignore`.

## Desarrollo

```bash
uv sync --extra dev --extra ml
uv run pytest
```

Documentación adicional:
- [`docs/greetd_integration.md`](docs/greetd_integration.md)
- [`docs/vm_test_protocol.md`](docs/vm_test_protocol.md)
