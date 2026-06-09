"""Inferencia por batch desde un ZIP de login.

Lee los rostros de ``faces/`` dentro del ZIP (en memoria, sin extraer ni
modificar el archivo), los preprocesa al tamaño del modelo y produce
predicciones softmax por frame.

TensorFlow se importa de forma diferida (extra ``[ml]``).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import AppConfig
from ..training.evaluate_model import load_class_indices, names_ordered_by_index
from ..vision.preprocessing import convert_bgr_to_rgb, resize_face

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _import_tf():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "TensorFlow no está instalado. Instálalo con 'pip install \".[ml]\"' "
            "en un entorno con Python 3.10–3.12 para inferir."
        ) from exc


def list_face_entries(zf: zipfile.ZipFile, faces_folder: str = "faces") -> List[str]:
    """Lista (ordenadas) las entradas de imagen bajo ``faces/`` dentro del ZIP."""
    prefix = faces_folder.rstrip("/") + "/"
    names = [
        n
        for n in zf.namelist()
        if n.startswith(prefix)
        and not n.endswith("/")
        and Path(n).suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(names)


def prepare_image(image_bgr: np.ndarray, image_size: Sequence[int]) -> np.ndarray:
    """Convierte un rostro BGR a RGB y lo redimensiona al tamaño del modelo.

    Mantiene el rango ``[0, 255]`` (el modelo incluye ``preprocess_input``).
    """
    rgb = convert_bgr_to_rgb(image_bgr)
    return resize_face(rgb, image_size)


def predictions_to_records(
    frame_names: Sequence[str],
    probs: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    """Convierte una matriz de probabilidades en registros por frame."""
    records: List[Dict[str, object]] = []
    for name, row in zip(frame_names, probs):
        record: Dict[str, object] = {"frame": name}
        for idx, cls in enumerate(class_names):
            record[cls] = float(row[idx])
        records.append(record)
    return records


def predict_zip(
    zip_path: str | Path,
    model_path: str | Path,
    config: AppConfig,
    *,
    class_indices_path: Optional[str | Path] = None,
    batch_size: int = 32,
    save_json: Optional[str | Path] = None,
) -> List[Dict[str, object]]:
    """Ejecuta inferencia softmax por frame sobre el ZIP de login."""
    tf = _import_tf()
    import cv2

    zip_path = Path(zip_path)
    model_path = Path(model_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"No existe el ZIP: {zip_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    if class_indices_path is None:
        class_indices_path = model_path.parent / "class_indices.json"
    class_indices = load_class_indices(class_indices_path)
    class_names = names_ordered_by_index(class_indices)

    input_shape = tuple(int(v) for v in config.model.input_shape)
    image_size = (input_shape[0], input_shape[1])
    faces_folder = config.output.zip_faces_folder

    images: List[np.ndarray] = []
    frame_names: List[str] = []
    # Solo lectura: el ZIP original no se modifica.
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in list_face_entries(zf, faces_folder):
            buffer = np.frombuffer(zf.read(name), dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                continue
            images.append(prepare_image(image, image_size))
            frame_names.append(Path(name).name)

    if not images:
        print(f"Aviso: el ZIP no contiene rostros válidos en '{faces_folder}/'.")
        records: List[Dict[str, object]] = []
    else:
        batch = np.stack(images).astype(np.float32)
        model = tf.keras.models.load_model(model_path)
        probs = model.predict(batch, batch_size=batch_size, verbose=0)
        records = predictions_to_records(frame_names, probs, class_names)

    if save_json is not None:
        save_path = Path(save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Inferencia finalizada.")
    print(f"  Frames predichos : {len(records)}")
    if save_json is not None:
        print(f"  Predicciones     : {save_json}")
    return records
