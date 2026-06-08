# Plan de desarrollo con Cursor — Login Facial KDE Plasma

## Contexto actualizado del dataset

El proyecto trabajará con **dos identidades conocidas**:

- `elioth`
- `emmanuel`

Y únicamente con **dos condiciones de iluminación**:

- `luz_ambiental`
- `luz_frontal`

Estas condiciones **no son clases del modelo**.  
Las clases entrenadas siguen siendo únicamente:

```text
elioth
emmanuel
```

La condición de luz se usará para balancear el dataset, evaluar robustez y documentar la variación de captura.

---

## Estructura de datos definida

Los videos crudos estarán guardados así:

```text
data/
└── raw/
    ├── elioth/
    │   ├── elioth_luz_ambiental_01.mp4
    │   ├── elioth_luz_ambiental_02.mp4
    │   ├── elioth_luz_frontal_01.mp4
    │   └── elioth_luz_frontal_02.mp4
    └── emmanuel/
        ├── emmanuel_luz_ambiental_01.mp4
        ├── emmanuel_luz_ambiental_02.mp4
        ├── emmanuel_luz_frontal_01.mp4
        └── emmanuel_luz_frontal_02.mp4
```

La estructura queda basada en el usuario.  
La condición de luz se infiere por el nombre del archivo.

Convención obligatoria de nombres:

```text
<usuario>_<condicion>_<numero>.mp4
```

Ejemplos válidos:

```text
elioth_luz_ambiental_01.mp4
elioth_luz_frontal_01.mp4
emmanuel_luz_ambiental_01.mp4
emmanuel_luz_frontal_01.mp4
```

---

## Estructura de salida del dataset procesado

Después de extraer los rostros desde videos, la salida recomendada será:

```text
data/
└── processed/
    ├── all/
    │   ├── elioth/
    │   │   ├── elioth_luz_ambiental_000001.jpg
    │   │   ├── elioth_luz_frontal_000002.jpg
    │   │   └── ...
    │   └── emmanuel/
    │       ├── emmanuel_luz_ambiental_000001.jpg
    │       ├── emmanuel_luz_frontal_000002.jpg
    │       └── ...
    ├── train/
    │   ├── elioth/
    │   └── emmanuel/
    ├── val/
    │   ├── elioth/
    │   └── emmanuel/
    └── test/
        ├── elioth/
        └── emmanuel/
```

La condición de luz se conserva en el nombre del archivo para análisis posterior, pero no aparece como clase del modelo.

---

# Fase 1 — Refactor de captura facial actual

## Objetivo

Tomar el script actual de captura sin vista previa y convertirlo en una arquitectura modular.

El script actual cumple una función útil para la fase de login:

```text
cámara → captura temporal → detección facial → ZIP de rostros
```

Pero para el proyecto completo se necesita separar responsabilidades.

## Prompt para Cursor

```text
Analiza el repositorio completo y refactoriza el script actual de captura facial sin vista previa hacia una arquitectura modular.

Requisitos funcionales:
- Mantener captura sin preview de cámara.
- Mantener duración configurable con --duration.
- Mantener --output-dir como carpeta de salida.
- Mantener un único output final: un archivo ZIP.
- Mantener detección de rostro con OpenCV Haar Cascade por ahora.
- Mantener selección del rostro de mayor área.
- Mantener recorte facial con margen.
- No generar metadata.csv.
- No dejar carpetas temporales sin comprimir.

Crea los módulos:
- src/face_login/acquisition/camera_capture.py
- src/face_login/acquisition/face_detector.py
- src/face_login/acquisition/zip_writer.py
- src/face_login/cli/capture_cli.py

Define:
- FaceDetector
- CameraCaptureConfig
- capture_faces_to_zip()

El comando final debe funcionar así:

python -m face_login.cli.capture_cli --name elioth --output-dir capturas_rostro --duration 5
```

---

# Fase 2 — Extracción de rostros desde videos para dataset

## Objetivo

Crear un script que lea videos desde:

```text
data/raw/elioth/
data/raw/emmanuel/
```

y extraiga rostros limpios para entrenamiento.

La clase se obtiene por la carpeta padre:

```text
data/raw/elioth/video.mp4     → clase elioth
data/raw/emmanuel/video.mp4   → clase emmanuel
```

La condición de luz se obtiene por el nombre del archivo:

```text
luz_ambiental
luz_frontal
```

## Reglas técnicas

El extractor debe:

1. Leer todos los videos dentro de `data/raw/elioth/` y `data/raw/emmanuel/`.
2. Validar que la clase sea solo `elioth` o `emmanuel`.
3. Validar que el nombre del video contenga una de las dos condiciones permitidas:
   - `luz_ambiental`
   - `luz_frontal`
4. Detectar rostro por frame.
5. Seleccionar el rostro de mayor área si hay varios.
6. Recortar la ROI facial.
7. Redimensionar a `224x224`.
8. Guardar solamente rostros limpios, sin rectángulos, sin elipses y sin texto.
9. Conservar la condición de luz en el nombre del archivo.
10. Generar un resumen de extracción.

## Prompt para Cursor

```text
Implementa un extractor de rostros desde videos para construir el dataset de entrenamiento.

Estructura de entrada obligatoria:
data/raw/
  elioth/
  emmanuel/

Cada carpeta contiene videos con nombres que incluyen la condición de luz:
- luz_ambiental
- luz_frontal

Ejemplos:
data/raw/elioth/elioth_luz_ambiental_01.mp4
data/raw/elioth/elioth_luz_frontal_01.mp4
data/raw/emmanuel/emmanuel_luz_ambiental_01.mp4
data/raw/emmanuel/emmanuel_luz_frontal_01.mp4

Crea:
- src/face_login/dataset/extract_faces_from_videos.py
- src/face_login/cli/extract_dataset_cli.py

Requisitos:
1. Recorrer automáticamente data/raw/elioth y data/raw/emmanuel.
2. Las únicas clases válidas son elioth y emmanuel.
3. Las únicas condiciones válidas son luz_ambiental y luz_frontal.
4. La clase se infiere por la carpeta padre.
5. La condición se infiere del nombre del archivo.
6. Si el video no contiene una condición válida en el nombre, debe ignorarse con warning.
7. Detectar rostro con Haar Cascade.
8. Si hay varios rostros, seleccionar el de mayor área:
   A = width * height.
9. Recortar la ROI facial con margen configurable.
10. Redimensionar a 224x224.
11. Guardar imágenes en:
    data/processed/all/<usuario>/
12. El nombre de salida debe conservar usuario y condición:
    elioth_luz_ambiental_000001.jpg
    elioth_luz_frontal_000002.jpg
13. No guardar frames anotados para entrenamiento.
14. Agregar opción --save-debug para guardar imágenes con bounding box en una carpeta separada:
    data/processed/debug/
15. Agregar filtros:
    - tamaño mínimo de rostro
    - varianza del Laplaciano para descartar blur
16. Imprimir resumen por usuario y condición.

Comando esperado:
python -m face_login.cli.extract_dataset_cli --input data/raw --output data/processed --image-size 224 --sample-every 5
```

---

# Fase 3 — Balance del dataset por iluminación

## Objetivo

Evitar que el modelo aprenda una condición de iluminación en lugar de la identidad.

El dataset debe estar balanceado aproximadamente así:

```text
elioth / luz_ambiental  ≈ emmanuel / luz_ambiental
elioth / luz_frontal    ≈ emmanuel / luz_frontal
elioth total            ≈ emmanuel total
```

## Prompt para Cursor

```text
Agrega un reporte de balance del dataset procesado.

Crea:
- src/face_login/dataset/dataset_report.py
- src/face_login/cli/dataset_report_cli.py

El reporte debe contar imágenes por:
- usuario
- condición de luz
- total por usuario
- total por condición

Condiciones válidas:
- luz_ambiental
- luz_frontal

Entrada:
data/processed/all/

Salida:
reports/dataset_balance.json
reports/dataset_balance.txt

El reporte debe advertir si:
- una clase tiene más de 20% de diferencia respecto a la otra
- una condición de luz está subrepresentada
- faltan imágenes para alguna combinación usuario/condición

Comando:
python -m face_login.cli.dataset_report_cli --input data/processed/all --output reports
```

---

# Fase 4 — Split train / validation / test

## Objetivo

Separar el dataset procesado en entrenamiento, validación y prueba.

Proporción recomendada:

```text
train: 70%
val:   15%
test:  15%
```

El split debe preservar:

```text
usuario
condición de luz
```

Es decir, debe ser un split estratificado por:

```text
(usuario, condición)
```

## Prompt para Cursor

```text
Implementa separación train/val/test para el dataset facial.

Entrada:
data/processed/all/
  elioth/
  emmanuel/

Los nombres de archivo incluyen la condición:
- luz_ambiental
- luz_frontal

Salida:
data/processed/train/
data/processed/val/
data/processed/test/

Requisitos:
1. Mantener clases solo como elioth y emmanuel.
2. No crear clase guest.
3. Hacer split estratificado por usuario y condición.
4. Proporción default:
   train=0.70
   val=0.15
   test=0.15
5. Usar seed configurable para reproducibilidad.
6. Copiar archivos, no moverlos.
7. Generar reporte:
   reports/split_report.json

Comando:
python -m face_login.cli.split_dataset_cli --input data/processed/all --output data/processed --train 0.70 --val 0.15 --test 0.15 --seed 42
```

---

# Fase 5 — Preprocesamiento y filtros de calidad

## Objetivo

Centralizar el preprocesamiento facial para que entrenamiento e inferencia usen la misma lógica.

## Procesos

- Recorte de ROI.
- Validación de límites del bounding box.
- Redimensionamiento.
- Conversión de color.
- Filtro por tamaño.
- Filtro por nitidez.
- Filtro por rostro ausente.

## Prompt para Cursor

```text
Crea una capa de preprocesamiento reutilizable para todo el proyecto.

Crea:
- src/face_login/preprocessing/image_preprocessor.py
- src/face_login/preprocessing/quality_filters.py

Implementa:
1. expand_bbox(frame_shape, bbox, margin_ratio)
2. crop_face(frame, bbox, margin_ratio)
3. resize_face(face, image_size)
4. convert_bgr_to_rgb(image)
5. variance_of_laplacian(image)
6. is_blurry(image, threshold)
7. is_valid_face_size(bbox, min_width, min_height)
8. preprocess_face_for_dataset(frame, bbox, config)

Requisitos:
- Manejar bounding boxes cerca de los bordes sin salirse del frame.
- No dibujar sobre imágenes usadas para entrenamiento.
- Regresar numpy.ndarray.
- Agregar type hints.
- Agregar tests unitarios.
```

---

# Fase 6 — Entrenamiento con Transfer Learning

## Objetivo

Entrenar un clasificador facial para dos identidades conocidas:

```text
elioth
emmanuel
```

La condición de luz no es clase, pero debe aparecer balanceada en el dataset.

## Arquitectura propuesta

```text
Input 224x224x3
↓
MobileNetV2 pretrained ImageNet, include_top=False
↓
GlobalAveragePooling2D
↓
Dropout(0.3)
↓
Dense(2, activation="softmax")
```

## Prompt para Cursor

```text
Implementa entrenamiento con transfer learning usando TensorFlow/Keras.

Crea:
- src/face_login/training/train_model.py
- src/face_login/cli/train_cli.py
- configs/training.yaml

Arquitectura:
- MobileNetV2 pretrained on ImageNet.
- include_top=False.
- input_shape=(224, 224, 3).
- Congelar backbone inicialmente.
- Añadir GlobalAveragePooling2D.
- Añadir Dropout(0.3).
- Añadir Dense(2, activation="softmax").
- Clases exactas:
  ["elioth", "emmanuel"]

No incluir guest como clase.

Dataset:
- data/processed/train
- data/processed/val

Métricas:
- accuracy
- precision
- recall

Guardar:
- models/face_classifier.keras
- models/class_indices.json
- reports/training_history.json
- reports/training_curves.png

Comando:
python -m face_login.cli.train_cli --data data/processed --config configs/training.yaml
```

---

# Fase 7 — Evaluación del modelo

## Objetivo

Evaluar el modelo en el conjunto de prueba.

La evaluación debe reportar desempeño general y, si es posible, desempeño por condición de luz usando el nombre del archivo.

## Prompt para Cursor

```text
Implementa evaluación del modelo facial.

Crea:
- src/face_login/training/evaluate_model.py
- src/face_login/cli/evaluate_cli.py

Entrada:
- models/face_classifier.keras
- models/class_indices.json
- data/processed/test

Métricas:
- accuracy
- precision
- recall
- F1-score
- confusion matrix

Además, intenta generar métricas por condición:
- luz_ambiental
- luz_frontal

La condición debe inferirse del nombre del archivo.

Guardar:
- reports/evaluation.json
- reports/confusion_matrix.png
- reports/evaluation_by_condition.json
- reports/misclassified/

Comando:
python -m face_login.cli.evaluate_cli --model models/face_classifier.keras --data data/processed/test
```

---

# Fase 8 — Inferencia temporal desde ZIP de login

## Objetivo

Usar el ZIP generado durante el login para decidir entre:

```text
elioth
emmanuel
guest
```

El modelo solo predice:

```text
elioth
emmanuel
```

`guest` se decide después como mecanismo de rechazo.

## Prompt para Cursor

```text
Implementa inferencia temporal desde un ZIP de capturas faciales.

Crea:
- src/face_login/inference/predictor.py
- src/face_login/inference/temporal_aggregator.py
- src/face_login/inference/decision_policy.py
- src/face_login/cli/infer_cli.py
- configs/decision.yaml

Requisitos:
1. Leer ZIP generado por el capturador de login.
2. Cargar solo imágenes de rostros recortados.
3. Preprocesar a 224x224.
4. Cargar models/face_classifier.keras.
5. Ejecutar softmax por frame.
6. Promediar scores por clase.
7. Aplicar política de decisión:
   - min_valid_frames
   - confidence_threshold
   - margin_threshold
8. Si falla cualquier condición, selected_user = guest.
9. Guest no aparece como clase del modelo.

Comando:
python -m face_login.cli.infer_cli --zip capturas_rostro/test.zip --model models/face_classifier.keras --config configs/decision.yaml
```

---

# Fase 9 — Política de decisión Elioth / Emmanuel / Guest

## Configuración sugerida

```yaml
min_valid_frames: 30
confidence_threshold: 0.80
margin_threshold: 0.25
known_classes:
  - elioth
  - emmanuel
fallback_user: guest
```

## Regla formal

Sea:

```text
s1 = mayor score promedio
s2 = segundo mayor score promedio
margin = s1 - s2
```

Aceptar identidad si:

```text
valid_frames >= min_valid_frames
s1 >= confidence_threshold
margin >= margin_threshold
```

Si no:

```text
guest
```

## Prompt para Cursor

```text
Implementa DecisionPolicy como clase pura y testeable.

Entrada:
- avg_scores: dict[str, float]
- valid_frames: int

Salida:
DecisionResult:
- selected_user
- accepted
- reason
- best_user
- best_score
- second_score
- margin
- valid_frames

Reglas:
1. Si valid_frames < min_valid_frames:
   selected_user="guest", reason="insufficient_valid_frames".
2. Si best_score < confidence_threshold:
   selected_user="guest", reason="low_confidence".
3. Si margin < margin_threshold:
   selected_user="guest", reason="low_margin".
4. Si todo pasa:
   selected_user=best_user, accepted=True.

Tests obligatorios:
- acepta Elioth
- acepta Emmanuel
- rechaza por pocos frames
- rechaza por baja confianza
- rechaza por margen ambiguo
- nunca espera guest como clase del modelo
```

---

# Fase 10 — Login CLI completo en modo seguro

## Objetivo

Unir captura + inferencia + decisión.

Primero debe funcionar en `dry-run`.

## Prompt para Cursor

```text
Implementa un CLI de login facial completo en modo seguro.

Crea:
- src/face_login/cli/login_cli.py

Flujo:
1. Capturar rostros durante 5 segundos.
2. Generar ZIP temporal.
3. Ejecutar inferencia temporal.
4. Aplicar DecisionPolicy.
5. Llamar a SessionDispatcher.

Modos:
- --dry-run: no inicia sesión, solo imprime qué usuario se seleccionaría.
- --simulate-dispatch: imprime el comando que se ejecutaría.
- --real-dispatch: reservado para integración posterior con greetd.

Comando:
python -m face_login.cli.login_cli --model models/face_classifier.keras --output-dir /tmp/face-login --dry-run
```

---

# Fase 11 — Dispatcher de sesión simulado

## Objetivo

Separar la decisión del modelo del inicio de sesión real.

## Prompt para Cursor

```text
Implementa una capa de despacho de sesión simulada.

Crea:
- src/face_login/session/dispatcher.py
- configs/session.yaml

Config:
users:
  elioth:
    linux_user: elioth
    session: plasma-wayland
  emmanuel:
    linux_user: emmanuel
    session: plasma-wayland
  guest:
    linux_user: guest
    session: plasma-wayland

Requisitos:
- Validar usuario permitido.
- No ejecutar comandos peligrosos.
- En dry-run, solo imprimir la sesión seleccionada.
- Agregar tests unitarios.
```

---

# Fase 12 — Integración experimental con greetd

## Objetivo

Documentar la integración con KDE Plasma y greetd sin activarla por defecto.

## Prompt para Cursor

```text
Crea una integración experimental con greetd, pero mantenla desactivada por defecto.

Crea:
- src/face_login/session/greetd_dispatcher.py
- examples/greetd/config.toml.example
- examples/greetd/face-login-greeter.example
- docs/GREETD_INTEGRATION.md

Requisitos:
1. Documentar que debe probarse en una VM.
2. No modificar archivos reales del sistema.
3. No ejecutar sudo.
4. Explicar flujo:
   systemd -> greetd.service -> face-login-greeter -> login_cli -> selected_user -> session dispatch.
5. Si no existe GREETD_SOCK, fallar con mensaje claro.
6. dry-run por defecto.
7. No guardar contraseñas en texto plano.
8. No asumir que reconocimiento facial reemplaza automáticamente a PAM.
```

---

# Fase 13 — Tests mínimos

## Prompt para Cursor

```text
Agrega pruebas unitarias con pytest.

Crea carpeta:
tests/

Pruebas:
1. DecisionPolicy acepta usuario cuando cumple condiciones.
2. DecisionPolicy rechaza por min_valid_frames.
3. DecisionPolicy rechaza por confidence_threshold.
4. DecisionPolicy rechaza por margin_threshold.
5. TemporalAggregator promedia scores correctamente.
6. SessionDispatcher no permite usuarios fuera de config.
7. Extractor ignora videos sin condición válida.
8. Extractor detecta condición luz_ambiental desde filename.
9. Extractor detecta condición luz_frontal desde filename.
10. Preprocessor maneja bounding boxes cerca de bordes.
```

---

# Fase 14 — Requirements

## Prompt para Cursor

```text
Actualiza requirements.txt y pyproject.toml.

requirements.txt:
- opencv-python
- numpy
- pillow
- tensorflow
- pyyaml
- scikit-learn
- matplotlib
- pytest
- pyinstaller

pyproject.toml:
- paquete face_login desde src/
- metadatos básicos
- scripts CLI si aplica

Evita dependencias innecesarias.
```

---

# Fase 15 — README técnico

## Prompt para Cursor

```text
Actualiza README.md con el flujo real del proyecto.

Debe incluir:
1. Objetivo:
   Autenticación 1:N en KDE Plasma mediante Reconocimiento Facial.
2. Clases entrenadas:
   - elioth
   - emmanuel
3. Condiciones de luz:
   - luz_ambiental
   - luz_frontal
4. Aclaración:
   Las condiciones de luz no son clases del modelo.
5. Estructura de datos:
   data/raw/elioth
   data/raw/emmanuel
6. Convención de nombres de videos.
7. Extracción de rostros desde videos.
8. Split train/val/test.
9. Entrenamiento con transfer learning.
10. Inferencia temporal.
11. Guest como mecanismo de rechazo.
12. Integración simulada con greetd.
```

---

# Orden recomendado de trabajo

```text
1. Refactor captura actual.
2. Implementar extracción de rostros desde videos.
3. Generar dataset con luz_ambiental y luz_frontal.
4. Crear reporte de balance.
5. Hacer split train/val/test estratificado.
6. Implementar preprocesamiento común.
7. Entrenar modelo con transfer learning.
8. Evaluar modelo.
9. Implementar inferencia temporal desde ZIP.
10. Implementar DecisionPolicy.
11. Crear login_cli con dry-run.
12. Simular dispatcher de sesión.
13. Documentar integración con greetd.
14. Agregar tests.
15. Actualizar README.
```

---

# Comandos esperados

## Extraer dataset desde videos

```bash
python -m face_login.cli.extract_dataset_cli \
  --input data/raw \
  --output data/processed \
  --image-size 224 \
  --sample-every 5
```

## Revisar balance del dataset

```bash
python -m face_login.cli.dataset_report_cli \
  --input data/processed/all \
  --output reports
```

## Separar train / val / test

```bash
python -m face_login.cli.split_dataset_cli \
  --input data/processed/all \
  --output data/processed \
  --train 0.70 \
  --val 0.15 \
  --test 0.15 \
  --seed 42
```

## Entrenar

```bash
python -m face_login.cli.train_cli \
  --data data/processed \
  --config configs/training.yaml
```

## Evaluar

```bash
python -m face_login.cli.evaluate_cli \
  --model models/face_classifier.keras \
  --data data/processed/test
```

## Inferencia temporal

```bash
python -m face_login.cli.infer_cli \
  --zip capturas_rostro/test.zip \
  --model models/face_classifier.keras \
  --config configs/decision.yaml
```

## Login facial en dry-run

```bash
python -m face_login.cli.login_cli \
  --model models/face_classifier.keras \
  --output-dir /tmp/face-login \
  --dry-run
```

---

# Nota conceptual importante

El modelo aprende únicamente a discriminar:

```text
elioth vs emmanuel
```

Las condiciones:

```text
luz_ambiental
luz_frontal
```

sirven para robustez y evaluación, pero no son etiquetas de salida.

La salida:

```text
guest
```

no se entrena.  
Se activa cuando no se cumplen las condiciones de decisión:

```text
valid_frames >= min_valid_frames
best_score >= confidence_threshold
margin >= margin_threshold
```