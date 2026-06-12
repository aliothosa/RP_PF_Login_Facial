# Explicación del proyecto RP_PF_Login_Facial

Documento orientado a la defensa académica del proyecto. Describe **qué hace el
sistema**, **cómo está organizado**, **qué componentes intervienen** y **cómo se
integra (o no) con el login real del sistema operativo**.

---

## 1. Problema y objetivo

### Problema

Se necesita un mecanismo de **identificación facial 1:N** en el arranque de sesión
de un equipo con **KDE Plasma**, capaz de:

- Reconocer a usuarios **conocidos** (`elioth`, `emmanuel`).
- **Rechazar** accesos poco fiables sin abrir sesión (usuario lógico `guest`).
- Mantener separada la **identificación** (¿quién parece ser?) de la
  **autenticación** del SO (¿tiene permiso legal de entrar? vía PAM/greetd).

### Objetivo del proyecto

Construir un pipeline reproducible:

```text
cámara (~5 s) → rostros recortados → clasificador CNN → agregación temporal
→ política de decisión → despacho de sesión (simulado o vía greetd-ipc)
```

El resultado de la parte de visión es siempre:

```text
selected_user ∈ { elioth, emmanuel, guest }
```

donde `guest` **no es una clase entrenada**, sino un mecanismo de rechazo.

---

## 2. Alcance y límites de seguridad

| Qué hace el proyecto | Qué NO hace por sí solo |
|---|---|
| Sugiere identidad a partir del rostro | Sustituir contraseña/PAM |
| Simula o solicita sesión vía greetd-ipc | Garantizar ausencia de suplantación |
| Registra decisiones y logs | Entrenar con datos de terceros sin consentimiento |

Principio de diseño: **el modelo propone username; PAM/greetd autentican y crean
la sesión gráfica**.

---

## 3. Arquitectura general del software

El código vive en el paquete Python `src/rp_face_login/`, organizado por
responsabilidades:

```text
rp_face_login/
├── acquisition/     Captura desde cámara → ZIP en memoria
├── vision/          Detección Haar + preprocesamiento de ROI
├── training/        Dataset, entrenamiento, evaluación
├── inference/       Predicción por frame + agregación temporal
├── decision/        Política aceptar / rechazar (guest)
├── session/         Despacho de sesión (dry-run, command, greetd-ipc)
├── config.py        Carga de configs/default.yaml
├── cli.py           Interfaz de línea de comandos
└── login_sim.py     Orquestación del login simulado
```

Configuración central: `configs/default.yaml` (cámara, umbrales, clases, despacho).

---

## 4. Flujo de datos end-to-end

### 4.1 Fase offline (entrenamiento)

```text
data/faces/{elioth,emmanuel}/...     Imágenes o videos por persona
        │
        ▼ prepare-dataset
data/processed/{train,val,test}/    Rostros recortados 224×224 por split
        │
        ▼ train
models/face_auth_model.keras         Modelo Keras + class_indices.json
```

### 4.2 Fase online (login simulado o greeter)

```text
/dev/video0
        │
        ▼ capture / login-sim (5 s, sin ventana gráfica)
capturas/login_YYYYMMDD_HHMMSS.zip
   └── faces/face_0001.jpg ...
        │
        ▼ predict_zip (softmax por frame)
Lista de predicciones [{elioth: p1, emmanuel: p2}, ...]
        │
        ▼ aggregate_predictions (promedio temporal)
best_user, best_score, margin, valid_frames
        │
        ▼ decide (política de umbrales)
selected_user, accepted, reason
        │
        ▼ SessionDispatcher (si accepted)
dry-run | command | greetd-ipc → sesión KDE (en producción vía PAM)
```

---

## 5. Descripción de cada módulo

### 5.1 Adquisición (`acquisition/camera_capture.py`)

**Función:** capturar video en vivo durante un intervalo configurable (por defecto
5 segundos) y empaquetar solo los **rostros válidos** en un ZIP.

**Comportamiento clave:**

- No usa `cv2.imshow` (adecuado para greeter headless).
- Por cada frame leído de la cámara:
  1. Detecta rostros con Haar Cascade.
  2. Se queda con el rostro de **mayor área**.
  3. Recorta con margen y guarda `faces/face_NNNN.jpg` dentro del ZIP.
- El ZIP se genera **en memoria**; no deja carpetas temporales descomprimidas.

**Salida:** `CaptureResult` con rutas, contadores `frames_read` / `valid_frames`.

---

### 5.2 Visión (`vision/`)

#### Detección (`face_detector.py`)

- Algoritmo: **Haar Cascade frontal** de OpenCV
  (`haarcascade_frontalface_default.xml`).
- Parámetros configurables: `scale_factor`, `min_neighbors`, `min_size`.
- Selección: **un rostro por frame** (el de mayor bounding box).

#### Preprocesamiento (`preprocessing.py`)

Pipeline sobre el ROI facial:

1. Recorte con margen (`margin_pixels`, default 30 px).
2. Redimensionado a **224×224** (`INTER_AREA`).
3. Conversión BGR → RGB (OpenCV usa BGR).
4. En inferencia el modelo aplica su propio `preprocess_input` del backbone;
   los píxeles se mantienen en rango **[0, 255]** al entrar al grafo Keras.

---

### 5.3 Entrenamiento (`training/`)

#### Preparación del dataset (`dataset_loader.py`)

- Entrada: carpetas por clase en `data/faces/{clase}/...`.
- Por imagen: detecta rostro → recorta → resize → guarda JPEG en
  `data/processed/{train,val,test}/{clase}/`.
- Split **70 % / 15 % / 15 %** con semilla fija (`seed=42`) para reproducibilidad.
- Imágenes sin rostro detectable se **descartan** (estadísticas en
  `dataset_stats.json`).

#### Entrenamiento (`train_model.py`)

- **Transfer learning** con backbone preentrenado en ImageNet.
- Backbone por defecto: **MobileNetV2** (alternativa: EfficientNetB0).
- Backbone **congelado**; solo se entrenan las capas de la cabeza clasificadora.
- Optimizador: Adam (`lr=1e-3`), pérdida: `categorical_crossentropy`.
- Métricas: accuracy, precision, recall.
- Salidas: `face_auth_model.keras`, `class_indices.json`, `history.json`.

#### Evaluación (`evaluate_model.py`)

- Métricas sobre el conjunto **test**.
- Genera `reports/evaluation.json` y matriz de confusión.

> Detalle de capas, entrenamiento y criterios de decisión:
> ver [`reconocimiento_patrones.md`](reconocimiento_patrones.md).

---

### 5.4 Inferencia (`inference/`)

#### Predicción por batch (`batch_predictor.py`)

- Lee el ZIP de login **sin extraerlo a disco**.
- Decodifica cada JPEG de `faces/`, prepara tensor 224×224×3.
- Ejecuta `model.predict` → vector softmax `[P(elioth), P(emmanuel)]` por frame.

#### Agregación temporal (`temporal_aggregation.py`)

- **Temporal Average Pooling:** promedia las probabilidades de cada clase a lo
  largo de todos los frames válidos del ZIP.
- Calcula ranking, margen entre 1.º y 2.º candidato:
  `margin = best_score - second_score`.

---

### 5.5 Decisión (`decision/decision_policy.py`)

Aplica **tres umbrales simultáneos** (AND lógico). Si falla alguno → `guest`.

| Umbral | Default | Significado |
|---|---|---|
| `min_valid_frames` | 30 | Mínimo de rostros capturados en la ventana |
| `confidence_threshold` | 0.80 | Confianza media mínima del mejor candidato |
| `margin_threshold` | 0.25 | Separación mínima entre 1.º y 2.º clase |

Motivos registrados: `accepted`, `insufficient_frames`, `low_confidence`,
`margin_below_threshold`.

---

### 5.6 Login simulado (`login_sim.py`)

Orquestador del flujo completo **sin tocar el SO**:

```python
capture_to_zip → predict_zip → aggregate → decide → imprimir resultado
```

Comando CLI: `uv run python -m rp_face_login.cli login-sim`.

---

### 5.7 Despacho de sesión (`session/`)

#### `dispatcher.py`

Traduce `selected_user` en una acción de sesión según el modo configurado:

| Modo | Comportamiento |
|---|---|
| `dry-run` | Solo imprime qué sesión se abriría |
| `command` | Ejecuta comando local (prototipo) |
| `greetd-ipc` | Protocolo JSON oficial de greetd + PAM + `startplasma-wayland` |

#### `greetd_ipc.py`

Cliente del socket `GREETD_SOCK`:

1. `create_session(username)`
2. Bucle PAM (`post_auth_message_response`)
3. `start_session(cmd=[...])`
4. El greeter termina; greetd lanza Plasma.

**Importante:** si `accepted=false` o `selected_user=guest`, el greeter **no
despacha** y sale con código 2.

---

## 6. Interfaz de línea de comandos (`cli.py`)

| Comando | Propósito |
|---|---|
| `capture` | Solo captura → ZIP |
| `prepare-dataset` | Genera dataset procesado |
| `train` | Entrena el clasificador |
| `evaluate` | Evalúa en test |
| `predict-zip` | Inferencia sobre ZIP existente |
| `login-sim` | Flujo completo simulado |
| `check-config` | Valida YAML |

Entorno recomendado: **uv + Python 3.12 + extra `[ml]`** (TensorFlow).

---

## 7. Scripts de despliegue e integración

| Script | Función |
|---|---|
| `scripts/face-login-greeter.sh` | Encadena login-sim + decisión + dispatcher |
| `scripts/install-vm-greeter.sh` | Instala artefactos en `/opt/rp_face_login` |
| `scripts/build_pyinstaller.sh` | Binarios autocontenidos (opcional) |

Integración con **greetd + KDE Plasma** documentada en:

- [`greetd_integration.md`](greetd_integration.md) — conceptos PAM vs identidad.
- [`deploy_endeavouros_vm.md`](deploy_endeavouros_vm.md) — pasos en EndeavourOS VM.
- [`vm_test_protocol.md`](vm_test_protocol.md) — fases A–D y rollback.

---

## 8. Datos y privacidad

Estructura esperada (no versionada en git):

```text
data/
├── raw/              Videos originales por persona
├── faces/            Imágenes por clase y condición de luz
└── processed/        Salida de prepare-dataset

models/               Pesos entrenados (.keras)
reports/              Evaluación y matrices de confusión
capturas/             ZIPs de login en runtime
logs/                 Trazas del greeter
```

Todo lo anterior está en `.gitignore` por privacidad.

---

## 9. Empaquetado

- **PyInstaller** (`packaging/entry_*.py`): ejecutables `face-login-capture` y
  `face-login-sim` con config y Haar embebidos; el modelo `.keras` va aparte.
- Limitación conocida: bundles `--onefile` con TensorFlow son muy pesados y pueden
  fallar al extraer en `/tmp` bajo el usuario `greeter`; alternativa: venv + Python
  en el launcher.

---

## 10. Separación identificación vs autenticación (defensa ante el profesor)

```text
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  CAPA ML (este proyecto)    │     │  CAPA SO (greetd + PAM)     │
├─────────────────────────────┤     ├─────────────────────────────┤
│ Entrada: píxeles de cámara  │     │ Entrada: username + credencial│
│ Salida: selected_user       │     │ Salida: sesión Plasma válida  │
│ Incertidumbre → guest       │     │ Fallo → no hay sesión         │
└─────────────────────────────┘     └─────────────────────────────┘
```

El proyecto demuestra la **capa ML** de forma aislada (`login-sim`, dry-run) y
opcionalmente la integración real (`greetd-ipc`) respetando PAM.

---

## 11. Resultados esperables en demo

1. Persona conocida, buena luz → `elioth` o `emmanuel`, `accepted=true`.
2. Persona no registrada o mala luz → `guest`, `accepted=false`.
3. Caso ambiguo (scores parecidos) → `guest` por `margin_below_threshold`.
4. Con greetd-ipc activo y PAM OK → apertura de Plasma Wayland tras el greeter.

---

## 12. Referencias internas

| Documento | Contenido |
|---|---|
| [`README.md`](../README.md) | Referencia técnica resumida |
| [`reconocimiento_patrones.md`](reconocimiento_patrones.md) | ML, capas, entrenamiento, decisión |
| [`plan.md`](plan.md) | Plan académico completo del proyecto |
| [`greetd_integration.md`](greetd_integration.md) | Integración con greetd/KDE |
