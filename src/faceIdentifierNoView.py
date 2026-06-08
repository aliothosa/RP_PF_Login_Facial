import cv2
import time
import argparse
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime


def zip_folder(source_folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in source_folder.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(source_folder))


def main():
    parser = argparse.ArgumentParser(
        description="Captura rostros durante unos segundos y genera únicamente un ZIP de salida."
    )

    parser.add_argument(
        "--name",
        default="usuario",
        help="Nombre o etiqueta del usuario"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Carpeta donde se guardará el ZIP final"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de cámara: 0, 1, 2..."
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Duración de captura en segundos"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"{args.name}_{timestamp}.zip"

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("No se pudo cargar el detector Haar Cascade de OpenCV.")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved_frames = 0
    total_frames_read = 0

    print("Cámara iniciada sin vista previa.")
    print(f"Capturando rostros durante {args.duration} segundos...")
    print(f"El único output será: {zip_path}")

    start_time = time.perf_counter()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            frames_dir = temp_path / "frames_anotados"
            faces_dir = temp_path / "rostros_recortados"

            frames_dir.mkdir(parents=True, exist_ok=True)
            faces_dir.mkdir(parents=True, exist_ok=True)

            while True:
                elapsed_time = time.perf_counter() - start_time

                if elapsed_time >= args.duration:
                    break

                ret, frame = cap.read()
                total_frames_read += 1

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(100, 100)
                )

                if len(faces) == 0:
                    continue

                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]

                annotated_frame = frame.copy()

                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                center = (x + w // 2, y + h // 2)
                axes = (w // 2, h // 2)

                cv2.ellipse(
                    annotated_frame,
                    center,
                    axes,
                    0,
                    0,
                    360,
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    f"Rostro detectado: {args.name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                saved_frames += 1

                frame_name = f"frame_{saved_frames:04d}.jpg"
                face_name = f"face_{saved_frames:04d}.jpg"

                frame_path = frames_dir / frame_name
                face_path = faces_dir / face_name

                cv2.imwrite(str(frame_path), annotated_frame)

                margin = 30

                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)

                face_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(str(face_path), face_crop)

            zip_folder(temp_path, zip_path)

    except KeyboardInterrupt:
        print("\nCaptura cancelada por el usuario.")

    finally:
        cap.release()

    print()
    print("Captura finalizada.")
    print(f"Frames leídos por la cámara: {total_frames_read}")
    print(f"Frames guardados con rostro: {saved_frames}")
    print(f"ZIP generado: {zip_path}")

    if saved_frames == 0:
        print("No se detectó ningún rostro durante la captura. El ZIP quedó vacío.")


if __name__ == "__main__":
    main()