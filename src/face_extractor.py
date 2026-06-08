import cv2
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Extrae rostros recortados desde un video y los guarda como imágenes."
    )

    parser.add_argument(
        "--name",
        default="usuario",
        help="Nombre o etiqueta del usuario"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Ruta del video de entrada"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Carpeta base donde se guardarán las caras recortadas"
    )

    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=5,
        help="Procesar un frame cada N frames"
    )

    parser.add_argument(
        "--max-saved",
        type=int,
        default=0,
        help="Máximo de imágenes a guardar. Usa 0 para no limitar"
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=30,
        help="Margen extra alrededor del rostro"
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=100,
        help="Tamaño mínimo del rostro detectado"
    )

    parser.add_argument(
        "--start-second",
        type=float,
        default=0.0,
        help="Segundo del video desde donde comenzar"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duración a procesar en segundos. Usa 0 para procesar todo el video"
    )

    parser.add_argument(
        "--flip",
        action="store_true",
        help="Voltear horizontalmente el frame"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No existe el video: {video_path}")

    if args.every_n_frames <= 0:
        raise ValueError("--every-n-frames debe ser mayor que 0")

    # Carpeta final:
    # output-dir/08-05-26_02:30:31/
    output_base_dir = Path(args.output_dir)

    timestamp = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    faces_dir = output_base_dir / timestamp

    # Evita colisiones incluso si ejecutas el script dos veces en el mismo segundo
    counter = 1
    while faces_dir.exists():
        faces_dir = output_base_dir / f"{timestamp}_{counter:02d}"
        counter += 1

    faces_dir.mkdir(parents=True, exist_ok=False)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError("No se pudo cargar el detector Haar Cascade de OpenCV.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    if args.start_second > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_second * 1000)

    start_msec = args.start_second * 1000
    end_msec = start_msec + args.duration * 1000 if args.duration > 0 else None

    total_frames_read = 0
    processed_frames = 0
    saved_faces = 0

    print(f"Video: {video_path}")
    print(f"Guardando rostros en: {faces_dir}")
    print(f"Procesando 1 de cada {args.every_n_frames} frames")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames_read += 1

            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if end_msec is not None and current_msec > end_msec:
                break

            if (total_frames_read - 1) % args.every_n_frames != 0:
                continue

            processed_frames += 1

            if args.flip:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(args.min_size, args.min_size)
            )

            if len(faces) == 0:
                continue

            # Toma solo la cara más grande
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            margin = args.margin
            y1 = max(0, y - margin)
            y2 = min(frame.shape[0], y + h + margin)
            x1 = max(0, x - margin)
            x2 = min(frame.shape[1], x + w + margin)

            face_crop = frame[y1:y2, x1:x2]

            saved_faces += 1
            face_path = faces_dir / f"{args.name}_{saved_faces:04d}.jpg"
            cv2.imwrite(str(face_path), face_crop)

            if args.max_saved > 0 and saved_faces >= args.max_saved:
                break

    except KeyboardInterrupt:
        print("\nProceso cancelado por el usuario.")

    finally:
        cap.release()

    print()
    print("Extracción finalizada.")
    print(f"Frames leídos: {total_frames_read}")
    print(f"Frames procesados: {processed_frames}")
    print(f"Rostros guardados: {saved_faces}")
    print(f"Carpeta de salida: {faces_dir}")


if __name__ == "__main__":
    main()