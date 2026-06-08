import cv2
import time
import argparse
from pathlib import Path


OUTPUT_DIR = Path("/tmp/face_identifier")


def limpiar_imagenes_anteriores(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    extensiones_imagen = {".jpg", ".jpeg", ".png"}

    for file in output_dir.iterdir():
        if file.is_file() and file.suffix.lower() in extensiones_imagen:
            file.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Captura rostros durante unos segundos y guarda las imágenes recortadas."
    )

    parser.add_argument(
        "--name",
        default="usuario",
        help="Nombre o etiqueta del usuario"
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

    limpiar_imagenes_anteriores(OUTPUT_DIR)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("No se pudo cargar el detector Haar Cascade de OpenCV.")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {args.camera}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    saved_faces = 0
    total_frames_read = 0


    start_time = time.perf_counter()

    try:
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

            # Toma solo el rostro más grande
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            margin = 30

            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)

            face_crop = frame[y1:y2, x1:x2]

            saved_faces += 1
            face_path = OUTPUT_DIR / f"{args.name}_{saved_faces:04d}.jpg"

            cv2.imwrite(str(face_path), face_crop)
            print("true")

    except KeyboardInterrupt:
        print("false")

    finally:
        cap.release()



    if saved_faces == 0:
        print("false")


if __name__ == "__main__":
    main()