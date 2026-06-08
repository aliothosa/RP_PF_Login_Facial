"""Captura temporal de rostros para login, sin vista previa.

Reemplaza de forma limpia y modular a ``src/faceIdentifierNoView.py``:

- No abre ninguna ventana (``cv2.imshow``).
- Lee la mayor cantidad de frames posible durante ``duration_seconds``.
- Detecta el rostro de mayor área por frame y recorta el ROI con margen.
- Escribe los recortes **directamente dentro de un único ZIP** (en memoria), por
  lo que no deja ninguna carpeta descomprimida ni genera ``metadata.csv``.

La lógica de detección/recorte vive en :class:`FaceDetector` y en
``vision.preprocessing`` para evitar duplicación.
"""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2

from ..config import AppConfig
from ..vision.face_detector import BoundingBox, FaceDetector
from ..vision.preprocessing import crop_face_with_margin


@dataclass
class CaptureResult:
    frames_read: int
    valid_frames: int
    zip_path: Path
    interrupted: bool = False


def _write_jpg(zipf: zipfile.ZipFile, arcname: str, image) -> None:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError(f"No se pudo codificar JPEG para {arcname}")
    zipf.writestr(arcname, buf.tobytes())


def _annotate(frame, box: BoundingBox, label: str):
    """Devuelve una copia del frame con el bounding box dibujado (modo debug)."""
    annotated = frame.copy()
    x, y, w, h = box
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"{label}",
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return annotated


def build_zip_from_frames(
    frames: Iterable,
    config: AppConfig,
    name: str,
    output_dir: str | Path,
    *,
    debug_annotated: bool = False,
    detector: Optional[FaceDetector] = None,
) -> CaptureResult:
    """Procesa una secuencia de frames y los guarda en un único ZIP.

    Función desacoplada de la cámara para poder probarse con frames sintéticos.
    Dentro del ZIP: ``faces/face_0001.jpg``, ``faces/face_0002.jpg``, ... y, si
    ``debug_annotated`` está activo, ``frames_anotados/frame_0001.jpg`` ...
    """
    detector = detector or FaceDetector(config.face_detection)
    margin = int(config.face_detection.margin_pixels)
    faces_folder = config.output.zip_faces_folder
    annotated_folder = config.output.zip_annotated_folder
    flip = bool(config.camera.flip_horizontal)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"{name}_{timestamp}.zip"

    frames_read = 0
    valid_frames = 0
    interrupted = False

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        try:
            for frame in frames:
                frames_read += 1
                if frame is None:
                    continue

                if flip:
                    frame = cv2.flip(frame, 1)

                box = detector.select_largest_face(detector.detect_faces(frame))
                if box is None:
                    continue

                face = crop_face_with_margin(frame, box, margin)
                valid_frames += 1
                _write_jpg(zf, f"{faces_folder}/face_{valid_frames:04d}.jpg", face)

                if debug_annotated:
                    _write_jpg(
                        zf,
                        f"{annotated_folder}/frame_{valid_frames:04d}.jpg",
                        _annotate(frame, box, name),
                    )
        except KeyboardInterrupt:
            interrupted = True

    return CaptureResult(frames_read, valid_frames, zip_path, interrupted)


def _camera_frames(cap: "cv2.VideoCapture", duration: float):
    """Generador que entrega frames durante ``duration`` segundos.

    Entrega ``None`` cuando una lectura falla (cuenta como frame leído inválido).
    """
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        ret, frame = cap.read()
        yield frame if ret else None


def capture_to_zip(
    config: AppConfig,
    name: str,
    output_dir: str | Path,
    *,
    duration: Optional[float] = None,
    camera_index: Optional[int] = None,
    debug_annotated: bool = False,
) -> CaptureResult:
    """Abre la cámara, captura durante la ventana temporal y genera el ZIP.

    Libera siempre la cámara (incluso ante ``KeyboardInterrupt``).
    """
    cam = config.camera
    duration = cam.duration_seconds if duration is None else duration
    camera_index = cam.index if camera_index is None else camera_index

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {camera_index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)

    try:
        result = build_zip_from_frames(
            _camera_frames(cap, duration),
            config,
            name,
            output_dir,
            debug_annotated=debug_annotated,
        )
    finally:
        cap.release()

    _report(result)
    return result


def _report(result: CaptureResult) -> None:
    if result.interrupted:
        print("Captura interrumpida por el usuario (cámara liberada).")
    print("Captura finalizada.")
    print(f"  Frames leídos  : {result.frames_read}")
    print(f"  Frames válidos : {result.valid_frames}")
    print(f"  ZIP generado   : {result.zip_path}")
    if result.valid_frames == 0:
        print("  Aviso: no se detectó ningún rostro válido durante la captura.")
