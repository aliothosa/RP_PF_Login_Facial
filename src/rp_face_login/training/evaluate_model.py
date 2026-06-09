"""Evaluación del clasificador facial sobre el conjunto de test.

Carga el modelo ``.keras`` y ``class_indices.json``, evalúa sobre
``<dataset>/test`` y produce accuracy, matriz de confusión y reporte por clase.
Resultados en ``reports/evaluation.json`` y ``reports/confusion_matrix.png``.

Las métricas se calculan con NumPy puro (testeable sin TensorFlow). TensorFlow
(carga del modelo y del dataset) y matplotlib (PNG) se importan de forma
diferida.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import AppConfig

DEFAULT_REPORTS_DIR = Path("reports")


def _import_tf():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "TensorFlow no está instalado. Instálalo con 'pip install \".[ml]\"' "
            "en un entorno con Python 3.10–3.12 para evaluar."
        ) from exc


def load_class_indices(path: str | Path) -> Dict[str, int]:
    """Carga ``class_indices.json`` (mapa nombre -> índice)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe class_indices.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"class_indices.json inválido: {path}")
    return {str(k): int(v) for k, v in data.items()}


def names_ordered_by_index(class_indices: Dict[str, int]) -> List[str]:
    """Devuelve los nombres de clase ordenados por su índice."""
    return [name for name, _ in sorted(class_indices.items(), key=lambda kv: kv[1])]


def compute_confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> np.ndarray:
    """Matriz de confusión (filas = verdadero, columnas = predicho)."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_report_from_cm(
    cm: np.ndarray, class_names: Sequence[str]
) -> Dict[str, object]:
    """Calcula accuracy, precision/recall/F1 por clase y macro-avg desde la matriz."""
    cm = np.asarray(cm)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / total if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    precisions, recalls, f1s = [], [], []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        support = int(cm[i, :].sum())
        predicted_positive = int(cm[:, i].sum())
        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_avg = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1_score": float(np.mean(f1s)) if f1s else 0.0,
    }
    return {"accuracy": accuracy, "per_class": per_class, "macro_avg": macro_avg}


def _plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4 + 0.5 * len(class_names), 4 + 0.5 * len(class_names)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Verdadero")
    ax.set_title("Matriz de confusión")

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(
    model_path: str | Path,
    dataset_dir: str | Path,
    config: AppConfig,
    *,
    class_indices_path: Optional[str | Path] = None,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Evalúa el modelo sobre ``dataset_dir`` (carpeta de test por clase)."""
    tf = _import_tf()

    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de test: {dataset_dir}")

    if class_indices_path is None:
        class_indices_path = model_path.parent / "class_indices.json"
    class_indices = load_class_indices(class_indices_path)
    class_names = names_ordered_by_index(class_indices)
    num_classes = len(class_names)

    input_shape = tuple(int(v) for v in config.model.input_shape)
    image_size = (input_shape[0], input_shape[1])

    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    model = tf.keras.models.load_model(model_path)

    y_true: List[int] = []
    y_pred: List[int] = []
    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    report = classification_report_from_cm(cm, class_names)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "evaluation.json"
    png_path = reports_dir / "confusion_matrix.png"

    results = {
        "model": str(model_path),
        "dataset": str(dataset_dir),
        "class_names": class_names,
        "num_samples": int(len(y_true)),
        "accuracy": report["accuracy"],
        "confusion_matrix": cm.tolist(),
        "per_class": report["per_class"],
        "macro_avg": report["macro_avg"],
    }
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_confusion_matrix(cm, class_names, png_path)

    _report(results, json_path, png_path)
    return results


def _report(results: dict, json_path: Path, png_path: Path) -> None:
    print("Evaluación finalizada.")
    print(f"  Muestras  : {results['num_samples']}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    for name, metrics in results["per_class"].items():
        print(
            f"  [{name}] precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1_score']:.3f} "
            f"support={metrics['support']}"
        )
    print(f"  Reporte   : {json_path}")
    print(f"  Matriz    : {png_path}")
