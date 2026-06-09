# Scripts legacy (referencia, no usar en producción)

Estos scripts monolíticos son la **implementación original** y se conservan solo
como referencia mientras se migra la lógica al paquete `src/rp_face_login/`.
No deben ejecutarse como parte del flujo final.

| Archivo | Rol original | Destino modular previsto |
|---|---|---|
| `faceIdentifierNoView.py` | Captura temporal de login sin vista, vuelca a `/tmp/face_identifier`, imprime `true`/`false`. | `rp_face_login/acquisition/camera_capture.py` |
| `faceIdentifierView.py` | Captura de dataset con vista previa, genera frames anotados + `metadata.csv` + ZIP. (Restaurado desde el historial git, commit `068e5f8`.) | `rp_face_login/acquisition/zip_writer.py` + `vision/` |
| `face_extractor.py` | Extrae rostros recortados desde un video de entrada. | `rp_face_login/training/dataset_loader.py` |

## Lógica reutilizable a rescatar

- Carga del Haar Cascade vía `cv2.data.haarcascades` (compatible con PyInstaller).
- Detección con `detectMultiScale` y selección de la cara de **mayor área** (`w*h`).
- Recorte de ROI facial con margen acotado a los bordes de la imagen.
- Generación de ZIP + `metadata.csv` (de `faceIdentifierView.py`).
- Extracción por video con `--every-n-frames` y ventana temporal (de `face_extractor.py`).

> Nota: el empaquetado PyInstaller se genera con `scripts/build_pyinstaller.sh`
> (sin `.spec` versionado). El antiguo `faceIdentifierNoView.spec` con rutas
> absolutas fue eliminado.
