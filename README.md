# RP_PF_Login_Facial

Sistema de autenticación / identificación facial **1:N** para login en KDE Plasma.
Reconoce dos identidades conocidas (`elioth`, `emmanuel`) y redirige a **`guest`**
mediante un mecanismo de **rechazo** cuando la confianza no es suficiente.

Documentación extendida: [`docs/explicacion_proyecto.md`](docs/explicacion_proyecto.md),
[`docs/reconocimiento_patrones.md`](docs/reconocimiento_patrones.md),
[`docs/plan.md`](docs/plan.md),
[`docs/deploy_endeavouros_vm.md`](docs/deploy_endeavouros_vm.md),
[`docs/greetd_integration.md`](docs/greetd_integration.md),
[`docs/vm_test_protocol.md`](docs/vm_test_protocol.md).

---

## 1. Objetivo

Implementar un flujo de login facial reproducible y defendible académicamente que:

- Capture rostros en tiempo real durante una ventana corta (~5 s).
- Clasifique la identidad con una red neuronal entrenada por **transfer learning**.
- Estabilice la decisión promediando predicciones **softmax** en el tiempo.
- Rechace accesos ambiguos o poco confiables asignando **`guest`** (sin entrenar
  a `guest` como clase facial).
- Mantenga separada la **identificación** (visión + ML) de la **autenticación del
  SO** (greetd/PAM), documentada para integración futura.

---

## 2. Pipeline general

```text
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Adquisición │ -> │   Visión     │ -> │ Entrenamiento│ -> │   Modelo     │
│  (cámara /   │    │  (Haar, ROI, │    │  (dataset,   │    │  .keras +    │
│   video)     │    │  preprocess) │    │  train)      │    │  class_idx   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                                  │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────┴───────┐
│  Despacho   │ <- │   Decisión   │ <- │ Agregación  │ <- │  Inferencia  │
│  de sesión  │    │  (aceptar /  │    │  temporal   │    │  (softmax /  │
│  (dry-run)  │    │   guest)     │    │  (promedio) │    │   frame)     │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
       ▲
       │  login-sim / greeter experimental
       │
┌─────────────┐
│ ZIP login   │  captura temporal sin vista previa
└─────────────┘
```

Módulos del paquete `src/rp_face_login/`:

| Módulo | Responsabilidad |
|---|---|
| `acquisition/` | Captura temporal, escritura de ZIP |
| `vision/` | Detección Haar, recorte, preprocesamiento |
| `training/` | Dataset, entrenamiento, evaluación |
| `inference/` | Predicción por batch, agregación temporal |
| `decision/` | Política de aceptación / rechazo |
| `session/` | Despacho de sesión (simulado) |

Configuración central: [`configs/default.yaml`](configs/default.yaml).

---

## 3. Entrenamiento vs. inferencia temporal

Son **dos fases distintas** que no deben mezclarse:

### Entrenamiento (offline)

- **Entrada:** imágenes por clase en `data/faces/` (o `data/raw/` → extracción).
- **Proceso:** `prepare-dataset` detecta rostros, recorta, redimensiona y divide
  en `train/val/test`. `train` ajusta pesos con transfer learning.
- **Salida:** `models/face_auth_model.keras`, `class_indices.json`, `history.json`.
- **Objetivo:** aprender a distinguir `elioth` vs `emmanuel` en condiciones
  variadas (luz frontal, luz ambiental, etc.).

### Inferencia temporal (login)

- **Entrada:** ~5 s de cámara en vivo → ZIP con `faces/face_0001.jpg`, ...
- **Proceso:** el modelo **no se entrena**; solo predice softmax por frame.
  Las predicciones se promedian (Temporal Average Pooling) y se aplica la
  política de decisión.
- **Salida:** `selected_user ∈ { elioth, emmanuel, guest }`.
- **Objetivo:** decisión estable en tiempo real sin actualizar pesos.

> Un batch de login **nunca** debe usarse para reentrenar el modelo.

---

## 4. Arquitectura de red

**Transfer Learning** con backbone CNN preentrenado en ImageNet:

```text
Input (224×224×3, BGR→RGB, [0,255])
        │
        ▼
preprocess_input (backbone-specific)
        │
        ▼
Backbone CNN (MobileNetV2 por defecto; EfficientNetB0 opcional)
  └── congelado inicialmente (trainable=False)
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dropout (0.3)
        │
        ▼
Dense(2, activation="softmax")
        │
        ▼
P(elioth | x) , P(emmanuel | x)     con Σ = 1
```

- **Pérdida:** `categorical_crossentropy`.
- **Métricas:** accuracy, precision, recall.
- **Clases entrenadas:** solo `elioth` y `emmanuel` (2 salidas softmax).
- Implementación: [`src/rp_face_login/training/train_model.py`](src/rp_face_login/training/train_model.py).

---

## 5. Regla de decisión

Tras la agregación temporal, cada frame válido aporta un vector softmax. Se
calcula el **promedio por clase**:

```text
avg_score(c_j) = (1 / N) · Σ_i  P(c_j | frame_i)
```

Se ordenan las clases por `avg_score` y se obtiene `best_user`, `best_score`,
`second_user`, `second_score` y `margin = best_score - second_score`.

La aceptación exige **las tres condiciones** (valores por defecto en
`configs/default.yaml`):

| Parámetro | Default | Condición |
|---|---|---|
| `min_valid_frames` | 30 | `valid_frames >= min_valid_frames` |
| `confidence_threshold` | 0.80 | `best_score >= confidence_threshold` |
| `margin_threshold` | 0.25 | `margin >= margin_threshold` |

Pseudocódigo:

```text
if valid_frames < min_valid_frames:  → guest
if best_score   < confidence_threshold: → guest
if margin       < margin_threshold:     → guest
else:                                   → best_user
```

Implementación: [`src/rp_face_login/decision/decision_policy.py`](src/rp_face_login/decision/decision_policy.py).

---

## 6. Guest como mecanismo de rechazo

**`guest` no es una clase entrenada.** La red solo produce softmax para
`elioth` y `emmanuel`. Si la inferencia no cumple los umbrales de la §5, la
política externa asigna `selected_user = guest` (`fallback_user` en config).

Motivos de rechazo registrados en la decisión:

- `insufficient_frames` — pocos rostros válidos capturados.
- `low_confidence` — el score promedio del mejor candidato es bajo.
- `margin_below_threshold` — empate o ambigüedad entre las dos identidades.

Esto evita el error conceptual de entrenar "persona desconocida" como tercera
clase y mitiga la falsa confianza del softmax ante rostros no vistos.

---

## 7. Comandos de uso

Entorno gestionado con **[uv](https://docs.astral.sh/uv/)**. TensorFlow requiere
**Python 3.10–3.12**.

```bash
uv sync --extra dev --extra ml
```

### `capture` — captura temporal de login (sin vista previa)

Genera un ZIP con rostros recortados en `faces/`.

```bash
uv run python -m rp_face_login.cli capture \
  --name elioth --output-dir ./capturas --duration 5
```

### `prepare-dataset` — dataset procesado train/val/test

```bash
uv run python -m rp_face_login.cli prepare-dataset \
  --raw-dir data/faces --output-dir data/processed
```

### `train` — entrenamiento con transfer learning

```bash
uv run python -m rp_face_login.cli train \
  --dataset-dir data/processed --output models/face_auth_model.keras
```

### `evaluate` — evaluación sobre el set de test

```bash
uv run python -m rp_face_login.cli evaluate \
  --dataset-dir data/processed/test --model models/face_auth_model.keras
```

Genera `reports/evaluation.json` y `reports/confusion_matrix.png`.

### `predict-zip` — inferencia softmax por frame desde un ZIP

```bash
uv run python -m rp_face_login.cli predict-zip \
  --zip ./capturas/elioth_20260609_120000.zip \
  --model models/face_auth_model.keras \
  --save-json reports/predictions.json
```

### `login-sim` — login simulado completo (sin sesión real)

Captura → inferencia → agregación → decisión. Imprime `elioth | emmanuel | guest`.

```bash
uv run python -m rp_face_login.cli login-sim \
  --output-dir ./capturas \
  --model models/face_auth_model.keras \
  --save-decision reports/decision.json
```

Greeter experimental (encadena `login-sim` + dispatcher):

```bash
DISPATCH_MODE=dry-run ./scripts/face-login-greeter.sh --duration 5
```

---

## 8. Advertencias de seguridad y privacidad

- **Datos biométricos:** rostros, videos y modelos entrenados son sensibles. No
  subirlos al repositorio (`data/`, `models/`, `capturas/`, `reports/`, `logs/`
  están en `.gitignore`).
- **Softmax ≠ autenticación:** la red puede asignar alta probabilidad a una clase
  conocida ante un rostro desconocido. Por eso existen `margin_threshold` y
  `confidence_threshold`, y `guest` como rechazo.
- **No saltarse PAM:** el reconocimiento facial produce identidad; la
  autenticación del SO pertenece a greetd/PAM (ver
  [`docs/greetd_integration.md`](docs/greetd_integration.md)).
- **Probar greetd solo en VM** con snapshot y plan de rollback a SDDM
  ([`docs/vm_test_protocol.md`](docs/vm_test_protocol.md)).
- **No hardcodear contraseñas** ni desactivar PAM en entornos reales.
- **Fase actual:** `login-sim` y el greeter **no inician sesión KDE real**; el
  dispatcher opera en `dry-run` o ejecuta comandos locales inofensivos (`echo`).

---

## 9. Datos del repositorio

Este repositorio **no incluye datasets reales** ni modelos entrenados. La
estructura local esperada (generada por el usuario, ignorada por git):

```text
data/
├── raw/           # videos fuente por identidad
├── faces/         # rostros recortados (elioth/, emmanuel/, condiciones de luz)
└── processed/     # train/val/test generado por prepare-dataset

models/            # face_auth_model.keras, class_indices.json (post-train)
capturas/          # ZIPs de login temporal
reports/           # evaluation.json, predictions.json, decision.json
logs/              # face-login.log
```

---

## 10. Diagrama textual del flujo completo

### Flujo de entrenamiento (offline)

```text
data/faces/{elioth,emmanuel}/
        │
        ▼  prepare-dataset (Haar → ROI → 224×224, split 70/15/15)
data/processed/{train,val,test}/
        │
        ▼  train (MobileNetV2 congelado + cabeza softmax)
models/face_auth_model.keras
models/class_indices.json
        │
        ▼  evaluate (test set)
reports/evaluation.json + confusion_matrix.png
```

### Flujo de login simulado (inferencia temporal)

```text
cámara (~5 s, sin imshow)
        │
        ▼  capture / login-sim paso 1
capturas/{name}_{timestamp}.zip
  └── faces/face_0001.jpg … face_N.jpg
        │
        ▼  predict-zip (softmax por frame)
[{ "frame": "…", "elioth": 0.91, "emmanuel": 0.09 }, …]
        │
        ▼  agregación temporal (promedio por clase)
{ best_user, best_score, second_user, second_score, margin, valid_frames }
        │
        ▼  política de decisión (§5)
selected_user: elioth | emmanuel | guest
        │
        ▼  session dispatcher (dry-run / command; sin login real aún)
[dry-run] Se despacharía sesión para '<user>': …
```

---

## Estructura del proyecto

```text
src/rp_face_login/
  acquisition/   vision/   training/   inference/   decision/   session/
  cli.py   config.py
configs/default.yaml
packaging/       # entry points PyInstaller
scripts/         # build_pyinstaller.sh, face-login-greeter.sh
tests/
docs/
```

## Empaquetado (PyInstaller + uv)

```bash
./scripts/build_pyinstaller.sh capture      # dist/face-login-capture
./scripts/build_pyinstaller.sh login-sim    # dist/face-login-sim (requiere [ml])

./dist/face-login-capture --name elioth --output-dir ./capturas --duration 5
./dist/face-login-sim --output-dir ./capturas --model models/face_auth_model.keras
```

Build reproducible sin `.spec` versionado; Haar Cascade y config embebidos.

## Desarrollo

```bash
uv sync --extra dev --extra ml
uv run pytest
uv run python -m rp_face_login.cli check-config
```
