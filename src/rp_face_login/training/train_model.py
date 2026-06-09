"""Entrenamiento del clasificador facial con Transfer Learning (Keras).

Arquitectura: backbone CNN preentrenado (MobileNetV2 por defecto, o
EfficientNetB0) congelado + cabeza de clasificación
``GlobalAveragePooling2D -> Dropout -> Dense(num_clases, softmax)``.

TensorFlow se importa de forma diferida: este módulo puede importarse sin TF
instalado (p. ej. para usar los helpers puros o el resto del paquete). El
entrenamiento real requiere ``tensorflow`` (extra ``[ml]``, Python 3.10–3.12).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..config import AppConfig

SUPPORTED_BACKBONES = ("MobileNetV2", "EfficientNetB0")


def _import_tf():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "TensorFlow no está instalado. Instálalo con 'pip install \".[ml]\"' "
            "en un entorno con Python 3.10–3.12 para entrenar/inferir."
        ) from exc


def class_indices_from_names(names: Sequence[str]) -> Dict[str, int]:
    """Mapea nombres de clase a índices (orden tal cual los entrega Keras)."""
    return {name: idx for idx, name in enumerate(names)}


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_model(
    input_shape: Sequence[int],
    num_classes: int,
    *,
    backbone: str = "MobileNetV2",
    dropout: float = 0.3,
    freeze_backbone: bool = True,
    weights: Optional[str] = "imagenet",
):
    """Construye el modelo de transfer learning con el backbone indicado."""
    tf = _import_tf()
    from tensorflow.keras import Model, layers

    input_shape = tuple(int(v) for v in input_shape)

    if backbone == "MobileNetV2":
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        base = MobileNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif backbone == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input

        base = EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    else:
        raise ValueError(
            f"Backbone no soportado: {backbone}. Usa uno de {SUPPORTED_BACKBONES}."
        )

    # Congelar el backbone inicialmente (transfer learning).
    base.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)  # espera valores en [0, 255]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name=f"face_auth_{backbone.lower()}")


def train(
    dataset_dir: str | Path,
    output_path: str | Path,
    config: AppConfig,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    seed: int = 42,
    backbone: Optional[str] = None,
) -> dict:
    """Entrena el clasificador desde ``dataset_dir/{train,val}`` y guarda artefactos.

    Guarda el modelo en ``output_path`` y, junto a él, ``class_indices.json`` y
    ``history.json``. Devuelve el historial de métricas.
    """
    tf = _import_tf()

    dataset_dir = Path(dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrenamiento: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de validación: {val_dir}")

    input_shape = tuple(int(v) for v in config.model.input_shape)
    image_size = (input_shape[0], input_shape[1])
    backbone = backbone or config.model.backbone

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    class_names: List[str] = list(train_ds.class_names)
    num_classes = len(class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(
        input_shape, num_classes, backbone=backbone, dropout=dropout, freeze_backbone=True
    )

    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    class_indices = class_indices_from_names(class_names)
    save_json(output_path.parent / "class_indices.json", class_indices)
    save_json(output_path.parent / "history.json", history.history)

    print("Entrenamiento finalizado.")
    print(f"  Backbone        : {backbone}")
    print(f"  Clases          : {class_indices}")
    print(f"  Modelo guardado : {output_path}")
    print(f"  class_indices   : {output_path.parent / 'class_indices.json'}")
    print(f"  history         : {output_path.parent / 'history.json'}")

    return history.history
