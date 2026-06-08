import cv2
import time
import csv
import argparse
import zipfile
from pathlib import Path
from datetime import datetime


def zip_folder(folder_path: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(folder_path.parent))


def main():
    parser = argparse.ArgumentParser(description="Captura 100 frames con rostro detectado y los guarda en ZIP.")
    parser.add_argument("--name", default="usuario", help="Nombre o etiqueta del usuario")
    parser.add_argument("--frames", type=int, default=100, help="Cantidad de frames a guardar")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara: 0, 1, 2...")
    parser.add_argument("--interval", type=float, default=0.08, help="Tiempo entre capturas en segundos")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("capturas_rostro")
    session_dir = base_dir / f"{args.name}_{timestamp}"
    frames_dir = session_dir / "frames_anotados"
    faces_dir = session_dir / "rostros_recortados"

    frames_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

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
    last_capture_time = 0
    metadata = []

    print("Cámara iniciada.")
    print("Presiona 'q' para salir antes de terminar.")
    print(f"Guardando {args.frames} frames para la etiqueta: {args.name}")

    while saved_frames < args.frames:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer frame de la cámara.")
            break

        # Efecto espejo para que sea más natural al usuario
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100)
        )

        annotated_frame = frame.copy()

        if len(faces) > 0:
            # Elegimos la cara más grande, asumiendo que es la principal
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            # Dibujar rectángulo
            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Dibujar una elipse como "contorno" facial aproximado
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

            current_time = time.time()

            if current_time - last_capture_time >= args.interval:
                saved_frames += 1
                last_capture_time = current_time

                frame_name = f"frame_{saved_frames:03d}.jpg"
                face_name = f"face_{saved_frames:03d}.jpg"

                frame_path = frames_dir / frame_name
                face_path = faces_dir / face_name

                # Guardar frame completo anotado
                cv2.imwrite(str(frame_path), annotated_frame)

                # Recortar rostro con pequeño margen
                margin = 30
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)

                face_crop = frame[y1:y2, x1:x2]
                cv2.imwrite(str(face_path), face_crop)

                metadata.append({
                    "frame": frame_name,
                    "face_crop": face_name,
                    "label": args.name,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })

        else:
            cv2.putText(
                annotated_frame,
                "No se detecta rostro",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.putText(
            annotated_frame,
            f"Frames guardados: {saved_frames}/{args.frames}",
            (30, annotated_frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Captura de rostro", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Captura interrumpida por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar metadata
    metadata_path = session_dir / "metadata.csv"

    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["frame", "face_crop", "label", "x", "y", "width", "height"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    # Crear ZIP
    zip_path = base_dir / f"{args.name}_{timestamp}.zip"
    zip_folder(session_dir, zip_path)

    print()
    print("Captura finalizada.")
    print(f"Frames guardados: {saved_frames}")
    print(f"Carpeta generada: {session_dir}")
    print(f"ZIP generado: {zip_path}")


if __name__ == "__main__":
    main()