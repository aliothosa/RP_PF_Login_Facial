# plan.md — Desarrollo con Cursor
# RP_PF_Login_Facial: Autenticación 1:N en KDE Plasma mediante Reconocimiento Facial

## 0. Objetivo

Este documento define un plan de desarrollo guiado por prompts para construir la parte de **login facial** del proyecto de Reconocimiento de Patrones.

El sistema final debe implementar autenticación/identificación facial 1:N reducida a dos identidades conocidas:

- `elioth`
- `emmanuel`

y una salida de rechazo:

- `guest`

`guest` **no debe modelarse como una identidad facial entrenada**. Debe implementarse como un **mecanismo de rechazo/fallback** cuando la inferencia no cumple condiciones mínimas de confianza.

---

## 1. Estado actual del repositorio

Repositorio:

```text
https://github.com/aliothosa/RP_PF_Login_Facial
```

Estado inicial observado:

```text
RP_PF_Login_Facial/
├── src/
│   ├── faceIdentifierNoView.py
│   └── faceIdentifierView.py
├── faceIdentifierNoView.spec
└── .gitignore
```

El proyecto ya contiene:

- Captura de frames con OpenCV.
- Detección facial mediante Haar Cascade.
- Selección de la cara principal por mayor área.
- Recorte de rostro.
- Generación de ZIP.
- Un script con vista de cámara.
- Un script sin vista de cámara.
- Un `.spec` de PyInstaller.

El objetivo del plan es evolucionar esa base hacia una arquitectura modular:

```text
captura facial
↓
preprocesamiento
↓
dataset
↓
entrenamiento
↓
inferencia sobre batch temporal
↓
agregación temporal
↓
regla de decisión
↓
despacho de sesión
```

---

## 2. Principios técnicos del proyecto

### 2.1 Separar entrenamiento e inferencia

El sistema tendrá dos fases diferentes.

#### Fase de entrenamiento

Se usa un dataset amplio de rostros de Elioth y Emmanuel en distintos entornos:

```text
dataset/
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

Esta fase entrena o ajusta el modelo.

#### Fase de inferencia durante login

Durante el login se capturan aproximadamente 5 segundos de video.

La salida de esta etapa es un batch temporal:

```text
batch_login = {face_0001, face_0002, ..., face_n}
```

Este batch **no actualiza pesos ni entrena el modelo**. Se usa para obtener múltiples predicciones y estabilizar la decisión final.

---

### 2.2 Guest como rechazo, no como clase entrenada

La red debe producir softmax para las identidades conocidas:

```text
elioth
emmanuel
```

Después, una regla externa decide si se acepta la identidad o si se envía a `guest`.

Esto evita el error conceptual de entrenar `guest` como si fuera una persona.

---

### 2.3 Agregación temporal

Cada frame válido genera una predicción:

```text
frame_i → [P(elioth), P(emmanuel)]
```

Después se promedia por clase:

```text
score_promedio_elioth = promedio de P(elioth) en todos los frames válidos
score_promedio_emmanuel = promedio de P(emmanuel) en todos los frames válidos
```

Finalmente se aplican tres condiciones:

```text
1. número mínimo de frames válidos
2. umbral mínimo de confianza
3. margen mínimo entre primer y segundo lugar
```

---

## 3. Arquitectura final recomendada

Estructura objetivo:

```text
RP_PF_Login_Facial/
├── src/
│   └── rp_face_login/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── acquisition/
│       │   ├── __init__.py
│       │   ├── camera_capture.py
│       │   └── zip_writer.py
│       ├── vision/
│       │   ├── __init__.py
│       │   ├── face_detector.py
│       │   ├── preprocessing.py
│       │   └── quality.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train_model.py
│       │   ├── dataset_loader.py
│       │   └── evaluate_model.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── model_loader.py
│       │   ├── batch_predictor.py
│       │   └── temporal_aggregation.py
│       ├── decision/
│       │   ├── __init__.py
│       │   └── decision_policy.py
│       └── session/
│           ├── __init__.py
│           ├── dispatcher.py
│           └── greetd_notes.md
├── configs/
│   └── default.yaml
├── scripts/
│   ├── capture_login_batch.sh
│   ├── train.sh
│   ├── infer_zip.sh
│   └── build_pyinstaller.sh
├── tests/
│   ├── test_decision_policy.py
│   ├── test_preprocessing.py
│   └── test_zip_writer.py
├── docs/
│   ├── architecture.md
│   ├── dataset_protocol.md
│   └── greetd_integration.md
├── requirements.txt
├── pyproject.toml
├── README.md
└── plan.md
```

---

## 4. Configuración base recomendada

Archivo objetivo:

```text
configs/default.yaml
```

Contenido recomendado:

```yaml
camera:
  index: 0
  duration_seconds: 5.0
  width: 1280
  height: 720
  flip_horizontal: true

face_detection:
  method: "haar"
  haar_cascade: "haarcascade_frontalface_default.xml"
  scale_factor: 1.1
  min_neighbors: 6
  min_size: [100, 100]
  margin_pixels: 30

preprocessing:
  target_size: [224, 224]
  color_format: "RGB"
  normalize_pixels: true

model:
  backbone: "MobileNetV2"
  input_shape: [224, 224, 3]
  classes:
    - "elioth"
    - "emmanuel"
  model_path: "models/face_auth_model.keras"

decision:
  min_valid_frames: 30
  confidence_threshold: 0.80
  margin_threshold: 0.25
  fallback_user: "guest"

output:
  zip_faces_folder: "faces"
  zip_annotated_folder: "frames_anotados"

session_dispatch:
  mode: "dry-run"
  users:
    elioth:
      command: "echo start elioth"
    emmanuel:
      command: "echo start emmanuel"
    guest:
      command: "echo start guest"
```

---

## 5. Flujo de trabajo por ramas

Crear una rama por fase:

```bash
git checkout -b feature/project-structure
git checkout -b feature/acquisition-pipeline
git checkout -b feature/dataset-training
git checkout -b feature/inference-pipeline
git checkout -b feature/decision-policy
git checkout -b feature/greetd-dispatcher
git checkout -b feature/packaging-docs
```

Reglas:

```text
1. Un cambio conceptual por commit.
2. Probar cada fase antes de avanzar.
3. No subir datasets biométricos reales al repositorio.
4. No integrar greetd en la máquina principal sin probar primero en VM.
```

---

# 6. Prompts para Cursor

## Prompt 0 — Auditoría inicial del repositorio

```text
Actúa como un ingeniero senior de visión computacional y Python.

Analiza el repositorio actual RP_PF_Login_Facial. Identifica:
1. Qué hace actualmente src/faceIdentifierNoView.py.
2. Qué hace actualmente src/faceIdentifierView.py.
3. Qué problemas de arquitectura hay por tener toda la lógica en scripts monolíticos.
4. Qué dependencias faltan en requirements.txt o pyproject.toml.
5. Qué archivos deberían ignorarse en .gitignore por privacidad y por build.
6. Qué partes deben conservarse de la implementación actual.

No modifiques todavía el código. Devuélveme un diagnóstico técnico y una propuesta de estructura modular.
```

Resultado esperado:

```text
- Diagnóstico del estado actual.
- Lista de problemas.
- Lista de mejoras.
- Confirmación de que la captura actual con OpenCV se puede reutilizar.
```

---

## Prompt 1 — Crear estructura de proyecto

```text
Refactoriza el repositorio para usar una estructura modular de paquete Python.

Crea la siguiente estructura:

src/rp_face_login/
  acquisition/
  vision/
  training/
  inference/
  decision/
  session/

Crea archivos __init__.py donde corresponda.

Crea también:
- configs/default.yaml
- scripts/
- tests/
- docs/
- requirements.txt
- pyproject.toml

No borres los scripts actuales todavía. Muévelos o consérvalos como referencia en docs/legacy/ si es necesario.

Asegúrate de que el proyecto pueda ejecutarse desde la raíz usando:

python -m rp_face_login.cli

No implementes todavía entrenamiento ni inferencia neuronal. Solo prepara estructura, imports limpios y configuración base.
```

Validación:

```bash
python -m rp_face_login.cli --help
```

Commit sugerido:

```bash
git add .
git commit -m "chore: restructure project as modular Python package"
```

---

## Prompt 2 — Centralizar configuración

```text
Implementa src/rp_face_login/config.py que cargue configs/default.yaml.

Requisitos:
1. Usar dataclasses o Pydantic, elige la opción más simple y mantenible.
2. Validar que existan camera, face_detection, preprocessing, model, decision y output.
3. Permitir sobrescribir por CLI:
   --camera-index
   --duration
   --output-dir
   --config
4. Agregar mensajes de error claros si la configuración está incompleta.
5. Agregar tests unitarios mínimos para validar carga de configuración.

No implementes aún captura ni modelo.
```

Validación:

```bash
python -m rp_face_login.cli --config configs/default.yaml --help
pytest
```

Commit:

```bash
git add .
git commit -m "feat: add YAML configuration loader"
```

---

## Prompt 3 — Refactorizar detector facial

```text
Extrae la lógica de detección facial actual hacia src/rp_face_login/vision/face_detector.py.

Requisitos:
1. Crear una clase FaceDetector.
2. Cargar Haar Cascade desde cv2.data.haarcascades.
3. Funcionar también empaquetado con PyInstaller.
4. Método detect_faces(frame) que devuelva bounding boxes.
5. Método select_largest_face(boxes) que elija la cara con mayor área.
6. Área = width * height.
7. Devolver None si no hay caras.
8. Agregar tests unitarios para select_largest_face usando cajas sintéticas.

No uses cámara todavía en esta tarea.
```

API esperada:

```python
detector = FaceDetector(config.face_detection)
boxes = detector.detect_faces(frame)
box = detector.select_largest_face(boxes)
```

Commit:

```bash
git add .
git commit -m "feat: add modular Haar face detector"
```

---

## Prompt 4 — Refactorizar preprocesamiento facial

```text
Crea src/rp_face_login/vision/preprocessing.py.

Implementa funciones puras para:
1. crop_face_with_margin(frame, box, margin)
2. resize_face(face, target_size)
3. convert_bgr_to_rgb(face)
4. normalize_pixels(face)
5. preprocess_face(frame, box, config)

Requisitos:
- crop_face_with_margin no debe salirse de los límites de la imagen.
- normalize_pixels debe convertir valores de [0,255] a [0,1].
- preprocess_face debe devolver un tensor listo para el modelo con shape (H, W, 3).
- Agregar tests unitarios para crop con margen en bordes de imagen.
```

Validación:

```bash
pytest tests/test_preprocessing.py
```

Commit:

```bash
git add .
git commit -m "feat: add face preprocessing pipeline"
```

---

## Prompt 5 — Captura temporal sin vista

```text
Reimplementa la captura sin vista previa en src/rp_face_login/acquisition/camera_capture.py.

Debe reemplazar de forma limpia la lógica actual de src/faceIdentifierNoView.py.

Requisitos:
1. Capturar durante duration_seconds.
2. No abrir ninguna ventana con cv2.imshow.
3. Leer la mayor cantidad de frames posible durante la ventana temporal.
4. Detectar rostro por frame.
5. Seleccionar la cara de mayor área.
6. Recortar ROI facial con margen.
7. Guardar únicamente un ZIP final en output_dir.
8. No dejar carpeta descomprimida.
9. No generar metadata.csv.
10. Dentro del ZIP guardar:
    faces/face_0001.jpg
    faces/face_0002.jpg
    ...
11. Guardar frames anotados solo si se activa --debug-annotated.
12. Reportar por consola:
    - frames leídos
    - frames válidos
    - ruta del ZIP
13. Manejar KeyboardInterrupt liberando la cámara.

Implementa el comando CLI:

python -m rp_face_login.cli capture --name elioth --output-dir ./capturas --duration 5
```

Validación:

```bash
python -m rp_face_login.cli capture --name elioth --output-dir ./capturas --duration 5
unzip -l capturas/*.zip
```

Commit:

```bash
git add .
git commit -m "feat: implement temporal face acquisition to zip"
```

---

## Prompt 6 — Protocolo de dataset

```text
Crea docs/dataset_protocol.md.

El documento debe explicar cómo capturar datos de entrenamiento para Elioth y Emmanuel.

Incluye:
1. Separación entre dataset de entrenamiento y batch de login.
2. Estructura:
   dataset/raw/elioth
   dataset/raw/emmanuel
   dataset/processed/train
   dataset/processed/val
   dataset/processed/test
3. Recomendaciones:
   - iluminación frontal
   - baja iluminación
   - luz lateral
   - diferentes fondos
   - diferentes distancias
   - rostro centrado
   - rostro ligeramente girado
   - con lentes/sin lentes si aplica
4. Advertencia de privacidad:
   - no subir rostros reales al repositorio
   - agregar dataset/ a .gitignore
5. Balance de clases.
6. Uso de data augmentation.
7. Diferencia entre adquisición de dataset y adquisición temporal de login.
```

Commit:

```bash
git add .
git commit -m "docs: add dataset capture protocol"
```

---

## Prompt 7 — Preparación del dataset procesado

```text
Implementa src/rp_face_login/training/dataset_loader.py.

Objetivo:
A partir de dataset/raw/elioth y dataset/raw/emmanuel, generar un dataset procesado dividido en train/val/test.

Requisitos:
1. Leer imágenes de carpetas por clase.
2. Detectar rostro.
3. Recortar ROI facial.
4. Preprocesar al tamaño objetivo.
5. Guardar imágenes procesadas en:
   dataset/processed/train/elioth
   dataset/processed/train/emmanuel
   dataset/processed/val/elioth
   dataset/processed/val/emmanuel
   dataset/processed/test/elioth
   dataset/processed/test/emmanuel
6. Permitir proporciones configurables:
   train 70%
   val 15%
   test 15%
7. Agregar seed para reproducibilidad.
8. Registrar cuántas imágenes fueron aceptadas y cuántas descartadas.
9. No subir dataset real al repo.

Agrega CLI:

python -m rp_face_login.cli prepare-dataset --raw-dir dataset/raw --output-dir dataset/processed
```

Validación:

```bash
python -m rp_face_login.cli prepare-dataset --raw-dir dataset/raw --output-dir dataset/processed
```

Commit:

```bash
git add .
git commit -m "feat: add dataset preprocessing pipeline"
```

---

## Prompt 8 — Modelo con Transfer Learning

```text
Implementa src/rp_face_login/training/train_model.py.

Usa TensorFlow/Keras y Transfer Learning.

Arquitectura:
1. Backbone CNN preentrenado: MobileNetV2 o EfficientNetB0.
2. Input shape desde config: 224x224x3.
3. Congelar inicialmente el backbone.
4. Agregar classification head:
   - GlobalAveragePooling2D
   - Dropout
   - Dense(2, activation="softmax")
5. Clases:
   - elioth
   - emmanuel
6. Pérdida:
   - categorical_crossentropy o sparse_categorical_crossentropy, según implementación.
7. Métricas:
   - accuracy
   - precision
   - recall si es razonable
8. Guardar el modelo en models/face_auth_model.keras.
9. Guardar class_indices.json para mapear índices a usuarios.
10. Guardar history.json con métricas de entrenamiento.

Agrega CLI:

python -m rp_face_login.cli train --dataset-dir dataset/processed --output models/face_auth_model.keras
```

Validación:

```bash
python -m rp_face_login.cli train --dataset-dir dataset/processed --output models/face_auth_model.keras
```

Commit:

```bash
git add .
git commit -m "feat: train face classifier with transfer learning"
```

---

## Prompt 9 — Evaluación del modelo

```text
Implementa src/rp_face_login/training/evaluate_model.py.

Debe:
1. Cargar models/face_auth_model.keras.
2. Cargar class_indices.json.
3. Evaluar sobre dataset/processed/test.
4. Generar:
   - accuracy
   - matriz de confusión
   - reporte por clase
5. Guardar resultados en reports/evaluation.json.
6. Guardar matriz de confusión como reports/confusion_matrix.png.

Agrega CLI:

python -m rp_face_login.cli evaluate --dataset-dir dataset/processed/test --model models/face_auth_model.keras
```

Validación:

```bash
python -m rp_face_login.cli evaluate --dataset-dir dataset/processed/test --model models/face_auth_model.keras
```

Commit:

```bash
git add .
git commit -m "feat: add model evaluation reports"
```

---

## Prompt 10 — Inferencia desde ZIP

```text
Implementa src/rp_face_login/inference/batch_predictor.py.

Objetivo:
Recibir un ZIP generado por la captura de login y producir predicciones softmax por frame.

Requisitos:
1. Leer imágenes desde faces/ dentro del ZIP.
2. Preprocesarlas al tamaño requerido por el modelo.
3. Cargar models/face_auth_model.keras.
4. Cargar class_indices.json.
5. Ejecutar inferencia por batch.
6. Devolver una lista de predicciones:
   [
     {"frame": "face_0001.jpg", "elioth": 0.91, "emmanuel": 0.09},
     ...
   ]
7. Guardar predictions.json si se pasa --save-json.
8. No modificar el ZIP original.

Agrega CLI:

python -m rp_face_login.cli predict-zip --zip ./capturas/test.zip --model models/face_auth_model.keras
```

Validación:

```bash
python -m rp_face_login.cli predict-zip --zip ./capturas/test.zip --model models/face_auth_model.keras --save-json reports/predictions.json
```

Commit:

```bash
git add .
git commit -m "feat: add batch inference from login zip"
```

---

## Prompt 11 — Agregación temporal

```text
Implementa src/rp_face_login/inference/temporal_aggregation.py.

Debe recibir la lista de predicciones por frame y calcular el promedio temporal por clase.

Ejemplo de entrada:
[
  {"elioth": 0.91, "emmanuel": 0.09},
  {"elioth": 0.87, "emmanuel": 0.13}
]

Salida:
{
  "avg_scores": {
    "elioth": 0.89,
    "emmanuel": 0.11
  },
  "valid_frames": 2
}

Requisitos:
1. Validar que haya predicciones.
2. Promediar por clase.
3. Ordenar las clases por score.
4. Devolver best_user, best_score, second_user, second_score y margin.
5. Agregar tests unitarios.
```

Fórmula:

```text
avg_score(c_j) = (1 / n_valid) * sum(P(c_j | x_i))
```

Commit:

```bash
git add .
git commit -m "feat: add temporal average pooling for predictions"
```

---

## Prompt 12 — Política de decisión y rechazo

```text
Implementa src/rp_face_login/decision/decision_policy.py.

La política debe aceptar o rechazar con estas condiciones:

1. valid_frames >= min_valid_frames
2. best_score >= confidence_threshold
3. margin >= margin_threshold

Si las tres se cumplen:
  selected_user = best_user

Si alguna falla:
  selected_user = guest

La salida debe incluir explicación de la decisión.

Ejemplo:
{
  "selected_user": "guest",
  "accepted": false,
  "reason": "margin_below_threshold",
  "valid_frames": 120,
  "best_user": "elioth",
  "best_score": 0.71,
  "second_user": "emmanuel",
  "second_score": 0.55,
  "margin": 0.16
}

Agrega tests para:
1. aceptación correcta de Elioth
2. aceptación correcta de Emmanuel
3. rechazo por pocos frames
4. rechazo por bajo score
5. rechazo por bajo margen
```

Pseudocódigo:

```python
if valid_frames < MIN_VALID_FRAMES:
    return guest

if best_score < CONFIDENCE_THRESHOLD:
    return guest

if margin < MARGIN_THRESHOLD:
    return guest

return best_user
```

Commit:

```bash
git add .
git commit -m "feat: add rejection-based decision policy"
```

---

## Prompt 13 — Comando de login completo en modo simulación

```text
Implementa un comando CLI llamado login-sim.

Flujo:
1. Capturar 5 segundos de cámara.
2. Generar ZIP temporal.
3. Cargar modelo.
4. Ejecutar inferencia sobre el ZIP.
5. Agregar predicciones temporalmente.
6. Aplicar política de decisión.
7. Imprimir usuario seleccionado:
   elioth | emmanuel | guest

Comando esperado:

python -m rp_face_login.cli login-sim --output-dir ./capturas --model models/face_auth_model.keras

Requisitos:
- No iniciar ninguna sesión real todavía.
- No tocar greetd.
- Guardar decision.json si se pasa --save-decision.
- Mostrar una salida clara para demo académica.
```

Validación:

```bash
python -m rp_face_login.cli login-sim --output-dir ./capturas --model models/face_auth_model.keras --save-decision reports/decision.json
```

Commit:

```bash
git add .
git commit -m "feat: add simulated facial login command"
```

---

## Prompt 14 — Dispatcher de sesión seguro

```text
Implementa src/rp_face_login/session/dispatcher.py.

Objetivo:
Crear una abstracción de despacho de sesión, pero sin modificar todavía el login real del sistema.

Debe soportar dos modos:

1. dry-run:
   Solo imprime qué sesión se despacharía.

2. command:
   Ejecuta un comando local configurado, por ejemplo:
   echo "starting session for elioth"

Requisitos:
- No hardcodear contraseñas.
- No desactivar PAM.
- No iniciar sesión real en esta fase.
- Mapear usuarios:
  elioth -> session command configurable
  emmanuel -> session command configurable
  guest -> session command configurable
- Leer el mapeo desde config.
- Agregar tests del mapeo.
```

Commit:

```bash
git add .
git commit -m "feat: add safe session dispatcher abstraction"
```

---

## Prompt 15 — Documentar integración con greetd

```text
Crea docs/greetd_integration.md.

Debe explicar la integración conceptual con greetd y KDE Plasma.

Incluye:
1. Qué es greetd dentro del proyecto.
2. Qué hace el greeter personalizado.
3. Flujo:
   systemd -> greetd -> face-login-greeter -> modelo facial -> selected_user -> session dispatch
4. Archivos relevantes:
   /etc/greetd/config.toml
   /etc/pam.d/greetd
   /usr/share/wayland-sessions/plasma.desktop
   /usr/share/xsessions/plasma.desktop
5. Ejemplo conceptual de config.toml.
6. Advertencia:
   - probar primero en VM
   - no reemplazar SDDM en máquina principal sin rollback
   - no hardcodear contraseñas
   - no saltarse PAM en un entorno real
7. Explicar que la parte de reconocimiento facial produce identidad, pero la autenticación del sistema operativo pertenece a greetd/PAM.
```

Commit:

```bash
git add .
git commit -m "docs: add greetd integration notes"
```

---

## Prompt 16 — Script experimental de greeter

```text
Crea un script experimental scripts/face-login-greeter.sh.

Este script debe:
1. Ejecutar el comando login-sim.
2. Tomar selected_user del resultado.
3. Llamar al dispatcher en modo dry-run o command.
4. Guardar logs en logs/face-login.log.

No debe modificar archivos de /etc.
No debe instalar greetd.
No debe iniciar KDE real todavía.

También crea docs/vm_test_protocol.md con un protocolo para probar en una VM.
```

Commit:

```bash
git add .
git commit -m "feat: add experimental greeter script in dry-run mode"
```

---

## Prompt 17 — Empaquetado con PyInstaller

```text
Actualiza el empaquetado con PyInstaller.

Requisitos:
1. Crear scripts/build_pyinstaller.sh.
2. Detectar dinámicamente la ruta del Haar Cascade con Python.
3. Incluir el XML con --add-data.
4. No dejar rutas absolutas del entorno local en el .spec.
5. Construir el ejecutable para capture o login-sim.
6. Documentar cómo ejecutar el binario.
7. Agregar al .gitignore:
   build/
   dist/
   *.spec si se decide generar dinámicamente
   __pycache__/
   .pytest_cache/
   dataset/
   models/
   reports/
   logs/
```

Ejemplo:

```bash
#!/usr/bin/env bash
set -e

CASCADE=$(python - <<'PY'
import cv2, os
print(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
PY
)

pyinstaller   --onefile   --clean   --name face-login   --add-data "$CASCADE:cv2/data"   main.py
```

Nota:
Si PyInstaller no puede usar directamente `python -m rp_face_login.cli`, crear un `main.py` delgado:

```python
from rp_face_login.cli import main

if __name__ == "__main__":
    main()
```

Commit:

```bash
git add .
git commit -m "build: add reproducible PyInstaller build script"
```

---

## Prompt 18 — README académico/técnico

```text
Actualiza README.md con una explicación técnica del proyecto.

Debe incluir:
1. Objetivo.
2. Pipeline general.
3. Diferencia entre entrenamiento e inferencia temporal.
4. Arquitectura de red:
   Transfer Learning + backbone CNN + Dense Softmax.
5. Regla de decisión:
   min_valid_frames
   confidence_threshold
   margin_threshold
6. Guest como mecanismo de rechazo.
7. Comandos de uso:
   - capture
   - prepare-dataset
   - train
   - evaluate
   - predict-zip
   - login-sim
8. Advertencias de seguridad y privacidad.
9. No incluir datasets reales.
10. Diagrama textual del flujo.
```

Commit:

```bash
git add .
git commit -m "docs: add technical README"
```

---

## Prompt 19 — Pruebas y robustez

```text
Revisa el proyecto completo y agrega pruebas unitarias donde falten.

Prioriza:
1. decision_policy.py
2. temporal_aggregation.py
3. preprocessing.py
4. face_detector.select_largest_face
5. zip_writer.py

Agrega también un comando:

pytest

Asegúrate de que pase sin requerir cámara ni modelo real para las pruebas unitarias básicas.
```

Commit:

```bash
git add .
git commit -m "test: add unit tests for core facial login logic"
```

---

## Prompt 20 — Revisión final de arquitectura

```text
Haz una revisión final del proyecto como si fueras un profesor experto en reconocimiento de patrones.

Evalúa:
1. Si el pipeline diferencia correctamente entrenamiento e inferencia.
2. Si Guest está implementado como rechazo y no como clase entrenada.
3. Si la agregación temporal está matemáticamente clara.
4. Si la decisión final es explicable.
5. Si el README permite defender el proyecto.
6. Si hay acoplamiento excesivo entre cámara, modelo y despacho de sesión.
7. Si hay riesgos de privacidad o seguridad.
8. Si los comandos son reproducibles.

No agregues features nuevas. Solo corrige documentación, nombres, errores y limpieza de código.
```

Commit:

```bash
git add .
git commit -m "refactor: polish architecture and documentation"
```

---

# 7. Fórmulas que deben aparecer en documentación

## Área del bounding box

```text
A = width * height
```

Selección de cara dominante:

```text
b* = argmax(width_i * height_i)
```

---

## Softmax

Para logits `z`:

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

En el proyecto:

```text
P(class_j | frame_i) = exp(logit_j) / sum_k exp(logit_k)
```

---

## Agregación temporal

```text
avg_score(class_j) = (1 / N) * sum_i P(class_j | frame_i)
```

---

## Margen de confianza

```text
margin = best_score - second_score
```

---

## Regla de aceptación

```text
accept = valid_frames >= N_min
         and best_score >= confidence_threshold
         and margin >= margin_threshold
```

---

## Regla de salida

```text
if accept:
    selected_user = best_user
else:
    selected_user = guest
```

---

# 8. Comandos finales esperados

## Capturar batch temporal de login

```bash
python -m rp_face_login.cli capture   --name elioth   --output-dir ./capturas   --duration 5
```

## Preparar dataset

```bash
python -m rp_face_login.cli prepare-dataset   --raw-dir dataset/raw   --output-dir dataset/processed
```

## Entrenar modelo

```bash
python -m rp_face_login.cli train   --dataset-dir dataset/processed   --output models/face_auth_model.keras
```

## Evaluar modelo

```bash
python -m rp_face_login.cli evaluate   --dataset-dir dataset/processed/test   --model models/face_auth_model.keras
```

## Predecir desde ZIP

```bash
python -m rp_face_login.cli predict-zip   --zip ./capturas/elioth_20260608_120000.zip   --model models/face_auth_model.keras
```

## Login simulado

```bash
python -m rp_face_login.cli login-sim   --output-dir ./capturas   --model models/face_auth_model.keras   --save-decision reports/decision.json
```

---

# 9. Orden recomendado de implementación

```text
1. Modularizar repositorio.
2. Centralizar configuración.
3. Refactorizar detector facial.
4. Refactorizar preprocesamiento.
5. Dejar captura temporal sólida.
6. Documentar protocolo de dataset.
7. Preparar dataset procesado.
8. Entrenar modelo con Transfer Learning.
9. Evaluar modelo.
10. Inferir desde ZIP.
11. Agregar Temporal Average Pooling.
12. Implementar decisión con rechazo.
13. Crear login-sim.
14. Agregar dispatcher seco.
15. Documentar greetd.
16. Empaquetar.
17. Probar en VM.
```

---

# 10. Criterios de aceptación

```text
[ ] Captura 5 segundos sin vista previa.
[ ] Genera un único ZIP de salida.
[ ] Extrae rostros válidos en faces/.
[ ] Tiene dataset separado para Elioth y Emmanuel.
[ ] Entrena modelo con Transfer Learning.
[ ] Produce softmax para dos identidades conocidas.
[ ] Promedia predicciones de múltiples frames.
[ ] Rechaza a Guest por bajo score, bajo margen o pocos frames.
[ ] Tiene comando login-sim reproducible.
[ ] Tiene documentación académica clara.
[ ] No sube rostros reales al repositorio.
[ ] No altera el login real sin VM.
```

---

# 11. Riesgos y mitigaciones

## Riesgo: confundir Guest con clase entrenada

Mitigación:

```text
Implementar Guest únicamente en decision_policy.py.
```

---

## Riesgo: sobreajuste

Mitigación:

```text
Dataset balanceado, distintas condiciones de luz, validación y test separados.
```

---

## Riesgo: decisión con pocos frames

Mitigación:

```text
min_valid_frames.
```

---

## Riesgo: empate o ambigüedad

Mitigación:

```text
margin_threshold.
```

---

## Riesgo: exceso de confianza de softmax ante usuarios desconocidos

Mitigación:

```text
confidence_threshold + margin_threshold + pruebas con personas desconocidas.
```

---

## Riesgo: romper login gráfico del sistema

Mitigación:

```text
Primero modo dry-run.
Después VM.
Nunca probar greetd directamente en la máquina principal sin rollback.
```

---

# 12. Prompt maestro para Cursor

```text
Actúa como experto en Computer Vision, Machine Learning Engineering y Linux desktop integration.

Estoy desarrollando un sistema de autenticación facial 1:N para KDE Plasma usando Python. El sistema reconoce dos identidades conocidas, Elioth y Emmanuel, y redirige a Guest mediante mecanismo de rechazo si la confianza no es suficiente.

Revisa el código completo y asegúrate de que:
1. La adquisición temporal de 5 segundos esté separada del entrenamiento.
2. El dataset de entrenamiento no se mezcle con batches de login.
3. El modelo use Transfer Learning con backbone CNN y Dense Softmax para dos clases.
4. Guest no esté implementado como clase entrenada.
5. La inferencia desde ZIP produzca scores por frame.
6. La agregación temporal use promedio de scores softmax.
7. La decisión final use min_valid_frames, confidence_threshold y margin_threshold.
8. La integración con sesión esté abstraída y segura.
9. La documentación permita defender el sistema ante un profesor experto en reconocimiento de patrones.

No agregues complejidad innecesaria. Prioriza claridad académica, reproducibilidad y separación de responsabilidades.
```

---

# 13. Entregable final esperado

Al terminar este plan, el repositorio debe permitir demostrar:

```text
1. Captura facial temporal:
   cámara -> ZIP de rostros

2. Entrenamiento:
   dataset Elioth/Emmanuel -> modelo .keras

3. Inferencia:
   ZIP -> predicciones por frame

4. Agregación:
   predicciones -> scores promedio

5. Decisión:
   scores -> Elioth/Emmanuel/Guest

6. Despacho:
   usuario seleccionado -> dry-run de sesión KDE/greetd
```

La defensa académica debe enfocarse en:

```text
- adquisición temporal
- ROI facial
- normalización
- transfer learning
- softmax
- temporal average pooling
- margen de confianza
- mecanismo de rechazo
- separación entre reconocimiento e integración de sistema operativo
```