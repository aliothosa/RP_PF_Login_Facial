# Reconocimiento de patrones en RP_PF_Login_Facial

Documento detallado sobre la parte de **visión por computadora y aprendizaje
automático**: preparación de datos, arquitectura de la red, entrenamiento,
inferencia temporal y **criterios de decisión** que determinan qué sesión se
despacha (o se rechaza).

---

## 1. Formulación del problema

### Tipo de problema

- **Clasificación supervisada multi-clase** con **2 clases positivas**:
  `elioth` y `emmanuel`.
- En login se usa como **identificación 1:N restringida**: el modelo elige entre
  las identidades conocidas; cualquier otra situación se trata como **rechazo**
  (`guest`), no como tercera clase entrenada.

### Por qué no se entrena la clase `guest`

Entrenar "desconocido" como clase adicional suele producir **falsa confianza**:
la red puede asignar alta probabilidad a una clase cualquiera ante rostros no
vistos. En su lugar, se usa una **política de umbrales** externa al softmax que
convierte incertidumbre en rechazo explícito.

---

## 2. Pipeline de reconocimiento (visión + ML)

```text
Frame BGR (1280×720)
      │
      ▼  Haar Cascade (OpenCV)
Bounding box del rostro mayor
      │
      ▼  Recorte + margen 30 px + resize 224×224 + BGR→RGB
Tensor de entrada (224, 224, 3), valores [0, 255]
      │
      ▼  CNN (MobileNetV2 congelado + cabeza entrenada)
Vector softmax de 2 componentes
      │
      ▼  Repetir por cada frame válido (~5 s de captura)
      │
      ▼  Promedio temporal por clase (Temporal Average Pooling)
avg_score(elioth), avg_score(emmanuel)
      │
      ▼  Política de decisión (3 umbrales)
selected_user ∈ { elioth, emmanuel, guest }
```

---

## 3. Preparación del dataset

### 3.1 Datos de entrada

Imágenes organizadas por identidad, idealmente con variación de iluminación:

```text
data/faces/
├── elioth/
│   ├── luz_frontal/
│   └── luz_ambiental/
└── emmanuel/
    ├── luz_frontal/
    └── luz_ambiental/
```

También se pueden usar videos en `data/raw/` y extraer frames (flujo legacy).

### 3.2 Proceso `prepare-dataset`

Implementado en `training/dataset_loader.py`:

1. **Por cada imagen** de cada clase:
   - Lectura con OpenCV.
   - Detección Haar → si no hay rostro, **descarte**.
   - Recorte con margen → resize **224×224** → guardado JPEG uint8.
2. **Partición estratificada por archivos** (no por píxeles):
   - Train: **70 %**
   - Validation: **15 %**
   - Test: **15 %**
   - Semilla: **42** (reproducible).
3. Salida:

```text
data/processed/
├── train/{elioth,emmanuel}/*.jpg
├── val/{elioth,emmanuel}/*.jpg
├── test/{elioth,emmanuel}/*.jpg
└── dataset_stats.json
```

### 3.3 Detección Haar (previa al CNN)

No forma parte del modelo neuronal entrenado, pero condiciona la calidad de los
datos:

| Parámetro | Valor default | Efecto |
|---|---|---|
| `scale_factor` | 1.1 | Pirámide de escalas para multiescala |
| `min_neighbors` | 6 | Exige consenso entre ventanas → menos falsos positivos |
| `min_size` | 100×100 | Ignora rostros demasiado pequeños |
| `margin_pixels` | 30 | Incluye contexto (frente, barbilla) en el recorte |

---

## 4. Arquitectura de la red neuronal

### 4.1 Estrategia: Transfer Learning

Se reutiliza un **backbone CNN preentrenado en ImageNet** como extractor de
 características fijas y se añade una **cabeza clasificadora** pequeña entrenada
 solo con rostros de `elioth` y `emmanuel`.

Ventajas académicas:

- Menos datos necesarios que entrenar desde cero.
- Convergencia más rápida.
- Features de bajo/medio nivel (bordes, texturas, partes de objetos) ya útiles.

### 4.2 Grafo del modelo (como se construye en código)

Implementación en `training/train_model.py`, función `build_model()`:

```text
┌──────────────────────────────────────────────────────────────────┐
│ CAPA / BLOQUE                    │ PARÁMETROS ENTRENABLES        │
├──────────────────────────────────┼───────────────────────────────┤
│ 1. Input                         │ —                             │
│    shape = (224, 224, 3)         │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ 2. preprocess_input              │ — (normalización fija del     │
│    (MobileNetV2 / EfficientNet)  │   backbone, no son pesos)     │
│    Escala entradas al rango que  │                               │
│    espera la red ImageNet        │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ 3. Backbone MobileNetV2          │ NO (freeze_backbone=True)     │
│    include_top=False             │ Pesos ImageNet congelados     │
│    Salida: mapa de características│                              │
│    espacial H'×W'×1280           │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ 4. GlobalAveragePooling2D        │ NO                            │
│    Promedia cada canal espacial  │ Convierte H'×W'×1280 → 1280   │
│    → vector de 1280 features     │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ 5. Dropout(p=0.3)                │ NO (regularización en train)  │
│    Apaga aleatoriamente 30 % de  │                               │
│    activaciones para evitar      │                               │
│    sobreajuste en la cabeza      │                               │
├──────────────────────────────────┼───────────────────────────────┤
│ 6. Dense(2, softmax)             │ SÍ (~2×1280 + 2 sesgos)       │
│    Salida: [P(elioth), P(emma)]  │ Única capa con pesos nuevos   │
└──────────────────────────────────┴───────────────────────────────┘
```

**Resumen de capas “propias” del proyecto:** 6 bloques en el grafo Keras; los
**únicos pesos aprendidos en este TFG** están en la capa `Dense(2)`.

### 4.3 Interior del backbone MobileNetV2 (referencia)

MobileNetV2 no es una sola capa: es una red profunda (~**53 bloques residuales
invertidos** en la arquitectura publicada; en TensorFlow/Keras el conteo total de
capas internas incluyendo convoluciones, BatchNorm y activaciones supera **150
capas**).

Estructura conceptual (Sandler et al., 2018):

| Etapa | Contenido | Función |
|---|---|---|
| Conv inicial | 3×3, stride 2 | Reducción espacial inicial |
| Bloques IR | 17 grupos de *Inverted Residuals* | Extracción eficiente de features |
| Conv 1×1 final | Expansión a 1280 canales | Embedding compacto antes del pooling |
| (en nuestro modelo) | Sin clasificador ImageNet | Se descarta el `top` original |

Cada **Inverted Residual Block** contiene típicamente:

1. **Expand** — conv 1×1 que aumenta canales.
2. **Depthwise** — conv 3×3 por canal (eficiente en móviles).
3. **Project** — conv 1×1 que reduce canales.
4. **Conexión residual** — si las dimensiones coinciden.

**Dedicación global del backbone:** transformar la imagen 224×224 en un mapa de
 características semánticas; las capas bajas capturan bordes/tonos, las altas
 patrones compatibles con identidad facial dentro del dominio entrenado.

### 4.4 Alternativa: EfficientNetB0

Configurable en `configs/default.yaml` (`model.backbone: EfficientNetB0`). Misma
cabeza (`GAP → Dropout → Dense(2)`), distinto `preprocess_input` y distinto
backbone congelado.

---

## 5. Entrenamiento del modelo

### 5.1 Hiperparámetros por defecto

| Parámetro | Valor | Ubicación |
|---|---|---|
| Épocas | 10 | CLI `--epochs` |
| Batch size | 32 | CLI `--batch-size` |
| Learning rate | 1e-3 | Adam |
| Dropout cabeza | 0.3 | `build_model()` |
| Backbone congelado | Sí | `freeze_backbone=True` |
| Función de pérdida | Categorical crossentropy | Multiclase one-hot |
| Métricas | Accuracy, Precision, Recall | Keras |

### 5.2 Procedimiento

Comando:

```bash
uv run python -m rp_face_login.cli train \
  --dataset-dir data/processed \
  --output models/face_auth_model.keras \
  --epochs 10 --batch-size 32
```

Pasos internos:

1. `image_dataset_from_directory` sobre `train/` y `val/` (labels categóricas).
2. Construcción del modelo con 2 salidas softmax.
3. `model.fit(train_ds, validation_data=val_ds)`.
4. Guardado:
   - `models/face_auth_model.keras`
   - `models/class_indices.json` — mapa `{ "elioth": 0, "emmanuel": 1 }`
   - `models/history.json` — curvas de entrenamiento.

### 5.3 Evaluación offline

```bash
uv run python -m rp_face_login.cli evaluate \
  --dataset-dir data/processed/test \
  --model models/face_auth_model.keras
```

Genera métricas agregadas y **matriz de confusión** en `reports/` para cuantificar
confusiones entre `elioth` y `emmanuel` en imágenes estáticas.

> **Nota:** métricas en test (imagen fija) ≠ comportamiento en video de 5 s;
> el login usa agregación temporal y umbrales adicionales.

---

## 6. Inferencia en tiempo de login

### 6.1 Captura (~5 segundos)

- Resolución de captura: 1280×720 (configurable).
- Solo frames con rostro detectado entran al ZIP.
- Típicamente decenas de JPEG en `faces/` (objetivo: ≥ `min_valid_frames`).

### 6.2 Predicción por frame

Para cada `face_NNNN.jpg`:

```text
x_i  →  model(x_i)  →  [p_i(elioth), p_i(emmanuel)]   con Σ p = 1
```

Implementado en `inference/batch_predictor.py` con inferencia por lotes
(`batch_size=32`).

### 6.3 Agregación temporal (Temporal Average Pooling)

Fórmula implementada en `inference/temporal_aggregation.py`:

```text
avg_score(c_j) = (1 / N) · Σ_{i=1..N}  P(c_j | frame_i)
```

Donde:

- `N` = número de frames válidos en el ZIP (`valid_frames`).
- `c_j` ∈ {`elioth`, `emmanuel`}.

A partir de `avg_score`:

```text
best_user, best_score   = clase con mayor avg_score
second_user, second_score = segunda clase en el ranking
margin = best_score - second_score
```

**Intuición:** una sola predicción ruidosa pesa poco; la decisión es **estable**
si la mayoría de frames coinciden.

---

## 7. Política de decisión y despacho de sesión

Son **dos etapas distintas**:

1. **Decisión ML** → `selected_user` + `accepted`.
2. **Despacho** → solo si `accepted=true`.

### 7.1 Reglas de aceptación / rechazo

Implementadas en `decision/decision_policy.py`. Se evalúan **en orden**; la
primera condición que falla define el `reason`:

```text
SI valid_frames < min_valid_frames (30)
   → selected_user = guest, reason = insufficient_frames

SI NO best_score >= confidence_threshold (0.80)
   → selected_user = guest, reason = low_confidence

SI NO margin >= margin_threshold (0.25)
   → selected_user = guest, reason = margin_below_threshold

SI NO
   → selected_user = best_user, reason = accepted
```

Tabla resumen:

| Condición | Umbral default | Interpretación |
|---|---|---|
| Frames suficientes | ≥ 30 | Evita decidir con pocas muestras |
| Confianza alta | avg ≥ 0.80 | El mejor candidato debe dominar |
| Margen claro | ≥ 0.25 | Evita confundir elioth ↔ emmanuel |

Ejemplo numérico de **aceptación**:

```text
valid_frames = 45
avg(elioth)   = 0.91
avg(emmanuel) = 0.09
margin = 0.82  →  accepted, selected_user = elioth
```

Ejemplo de **rechazo por ambigüedad**:

```text
valid_frames = 40
avg(elioth)   = 0.55
avg(emmanuel) = 0.45
margin = 0.10  →  guest (margin_below_threshold)
```

Ejemplo de **rechazo por baja confianza**:

```text
valid_frames = 35
avg(elioth)   = 0.62
avg(emmanuel) = 0.38
margin = 0.24  →  guest (low_confidence en 0.62 < 0.80)
```

### 7.2 Qué sesión se despacha

El despacho **no elige** entre clases: recibe el `selected_user` ya decidido.

```text
login_sim / greeter
      │
      ├─ accepted=false  o  selected_user=guest
      │       → NO hay despacho (exit code 2 en greeter)
      │
      └─ accepted=true  y  user ∈ {elioth, emmanuel}
              → SessionDispatcher.dispatch(user)
```

Configuración en `session_dispatch` (`configs/default.yaml`):

```yaml
session_dispatch:
  mode: dry-run | command | greetd-ipc
  users:
    elioth:
      command: "/usr/bin/startplasma-wayland"
    emmanuel:
      command: "/usr/bin/startplasma-wayland"
```

| Modo | Efecto sobre sesión |
|---|---|
| `dry-run` | Solo log: “se despacharía sesión para elioth” |
| `command` | Ejecuta comando shell configurado (prototipo) |
| `greetd-ipc` | PAM + `start_session` vía greetd → Plasma real |

**Mapeo usuario → sesión:** ambos usuarios conocidos apuntan a la misma sesión
Plasma Wayland; la diferencia está en **qué cuenta PAM autentica** (username
facial sugerido = username Linux).

### 7.3 Diagrama completo decisión + despacho

```text
                    ┌─────────────────┐
                    │  N frames ZIP   │
                    └────────┬────────┘
                             │
                    softmax + promedio
                             │
              ┌──────────────┴──────────────┐
              │ best_score, margin, N       │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │  decide()       │
                    │  3 umbrales     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       accepted          guest            guest
       elioth/emmanuel  (baja conf)    (pocos frames)
            │                │                │
            ▼                ▼                ▼
     dispatch(user)      exit 2           exit 2
     greetd-ipc/PAM      sin sesión       sin sesión
```

---

## 8. Salida JSON de la decisión (para defensa)

Ejemplo de campos en `reports/decision.json`:

```json
{
  "selected_user": "elioth",
  "accepted": true,
  "reason": "accepted",
  "valid_frames": 42,
  "best_user": "elioth",
  "best_score": 0.893,
  "second_user": "emmanuel",
  "second_score": 0.107,
  "margin": 0.786,
  "thresholds": {
    "min_valid_frames": 30,
    "confidence_threshold": 0.80,
    "margin_threshold": 0.25
  }
}
```

Estos campos permiten **explicar** cada rechazo sin caja negra.

---

## 9. Limitaciones y mejoras posibles (honestidad académica)

| Limitación | Mitigación actual / futura |
|---|---|
| Solo 2 identidades | Ampliar clases y reentrenar cabeza `Dense(K)` |
| Haar frágil con pose/luz | RetinaFace / MediaPipe en detección |
| Backbone congelado | Fine-tuning parcial del backbone |
| Contraseña PAM separada | Módulo PAM facial (estilo Howdy) |
| Softmax puede ser confiado erróneamente | Umbrales + margen + rechazo `guest` |
| PyInstaller + TF pesado | venv en greeter o build `--onedir` |

---

## 10. Comandos de reproducción (ML)

```bash
# Entorno
uv sync --extra dev --extra ml

# Dataset
uv run python -m rp_face_login.cli prepare-dataset \
  --raw-dir data/faces --output-dir data/processed

# Entrenamiento
uv run python -m rp_face_login.cli train \
  --dataset-dir data/processed \
  --output models/face_auth_model.keras

# Login simulado (inferencia + decisión)
uv run python -m rp_face_login.cli login-sim \
  --model models/face_auth_model.keras \
  --save-decision reports/decision.json \
  --duration 5
```

---

## 11. Referencias bibliográficas sugeridas

- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*.
- Viola, P. & Jones, M. (2001). *Rapid Object Detection using Boosted Cascade*.
- Documentación TensorFlow Keras Applications (MobileNetV2, transfer learning).

---

## 12. Archivos de código relacionados

| Tema | Archivo |
|---|---|
| Arquitectura y train | `src/rp_face_login/training/train_model.py` |
| Dataset | `src/rp_face_login/training/dataset_loader.py` |
| Evaluación | `src/rp_face_login/training/evaluate_model.py` |
| Inferencia | `src/rp_face_login/inference/batch_predictor.py` |
| Agregación temporal | `src/rp_face_login/inference/temporal_aggregation.py` |
| Decisión | `src/rp_face_login/decision/decision_policy.py` |
| Orquestación login | `src/rp_face_login/login_sim.py` |
| Umbrales | `configs/default.yaml` → sección `decision` |
