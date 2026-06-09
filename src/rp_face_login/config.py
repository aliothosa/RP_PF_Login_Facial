"""Carga, validación y override de la configuración del proyecto.

Lee ``configs/default.yaml`` y lo expone como dataclasses tipadas. Permite
sobrescribir valores puntuales desde la CLI (``--camera-index``, ``--duration``,
``--output-dir``) sin tocar el archivo de configuración.

Se eligió **dataclasses** (biblioteca estándar) frente a Pydantic por ser la
opción más simple y mantenible para este alcance, sin dependencias extra.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Secciones obligatorias que debe contener el archivo de configuración.
REQUIRED_SECTIONS = (
    "camera",
    "face_detection",
    "preprocessing",
    "model",
    "decision",
    "output",
)

def default_config_path() -> Path:
    """Ruta al YAML por defecto; en binarios PyInstaller usa el bundle embebido."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled = Path(sys._MEIPASS) / "configs" / "default.yaml"  # type: ignore[attr-defined]
        if bundled.exists():
            return bundled
    return Path("configs/default.yaml")


DEFAULT_CONFIG_PATH = default_config_path()


class ConfigError(RuntimeError):
    """Error de carga o validación de configuración."""


@dataclass
class CameraConfig:
    index: int = 0
    duration_seconds: float = 5.0
    width: int = 1280
    height: int = 720
    flip_horizontal: bool = True


@dataclass
class FaceDetectionConfig:
    method: str = "haar"
    haar_cascade: str = "haarcascade_frontalface_default.xml"
    scale_factor: float = 1.1
    min_neighbors: int = 6
    min_size: list[int] = field(default_factory=lambda: [100, 100])
    margin_pixels: int = 30


@dataclass
class PreprocessingConfig:
    target_size: list[int] = field(default_factory=lambda: [224, 224])
    color_format: str = "RGB"
    normalize_pixels: bool = True


@dataclass
class ModelConfig:
    backbone: str = "MobileNetV2"
    input_shape: list[int] = field(default_factory=lambda: [224, 224, 3])
    classes: list[str] = field(default_factory=lambda: ["elioth", "emmanuel"])
    model_path: str = "models/face_auth_model.keras"


@dataclass
class DecisionConfig:
    min_valid_frames: int = 30
    confidence_threshold: float = 0.80
    margin_threshold: float = 0.25
    fallback_user: str = "guest"


@dataclass
class OutputConfig:
    zip_faces_folder: str = "faces"
    zip_annotated_folder: str = "frames_anotados"
    output_dir: str = "./capturas"


@dataclass
class CLIOverrides:
    """Valores que la CLI puede sobreponer al config. ``None`` = no sobrescribir."""

    camera_index: int | None = None
    duration: float | None = None
    output_dir: str | None = None


# Mapea cada sección obligatoria a su dataclass.
_SECTION_TYPES: dict[str, type] = {
    "camera": CameraConfig,
    "face_detection": FaceDetectionConfig,
    "preprocessing": PreprocessingConfig,
    "model": ModelConfig,
    "decision": DecisionConfig,
    "output": OutputConfig,
}


@dataclass
class AppConfig:
    camera: CameraConfig
    face_detection: FaceDetectionConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    decision: DecisionConfig
    output: OutputConfig
    session_dispatch: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(
            camera=_instantiate(CameraConfig, data["camera"], "camera"),
            face_detection=_instantiate(
                FaceDetectionConfig, data["face_detection"], "face_detection"
            ),
            preprocessing=_instantiate(
                PreprocessingConfig, data["preprocessing"], "preprocessing"
            ),
            model=_instantiate(ModelConfig, data["model"], "model"),
            decision=_instantiate(DecisionConfig, data["decision"], "decision"),
            output=_instantiate(OutputConfig, data["output"], "output"),
            session_dispatch=data.get("session_dispatch", {}) or {},
        )

    def apply_overrides(self, overrides: CLIOverrides | None) -> "AppConfig":
        """Devuelve una copia con los overrides de CLI aplicados."""
        if overrides is None:
            return self
        if overrides.camera_index is not None:
            self.camera.index = overrides.camera_index
        if overrides.duration is not None:
            if overrides.duration <= 0:
                raise ConfigError("--duration debe ser un número mayor que 0.")
            self.camera.duration_seconds = overrides.duration
        if overrides.output_dir is not None:
            self.output.output_dir = overrides.output_dir
        return self


def _instantiate(cls: type, data: Any, section: str):
    """Crea una dataclass de sección validando tipo y claves desconocidas."""
    if not isinstance(data, dict):
        raise ConfigError(
            f"La sección '{section}' debe ser un mapeo (clave: valor), "
            f"se obtuvo {type(data).__name__}."
        )
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid_fields)
    if unknown:
        raise ConfigError(
            f"Claves no reconocidas en la sección '{section}': {', '.join(unknown)}. "
            f"Claves válidas: {', '.join(sorted(valid_fields))}."
        )
    try:
        return cls(**data)
    except TypeError as exc:  # pragma: no cover - salvaguarda
        raise ConfigError(f"Error al construir la sección '{section}': {exc}") from exc


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: CLIOverrides | None = None,
) -> AppConfig:
    """Carga el YAML de configuración y devuelve un :class:`AppConfig` validado.

    Importa ``yaml`` de forma diferida para que comandos como ``--help`` no
    requieran PyYAML instalado. Aplica overrides de CLI si se proporcionan.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(
            f"No existe el archivo de configuración: {config_path}. "
            f"Usa --config para indicar una ruta válida."
        )

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ConfigError(
            "PyYAML no está instalado. Instala las dependencias con "
            "'pip install -r requirements.txt'."
        ) from exc

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"El archivo YAML está mal formado ({config_path}): {exc}") from exc

    if raw is None:
        raise ConfigError(f"El archivo de configuración está vacío: {config_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"El YAML de configuración debe ser un mapeo en la raíz: {config_path}"
        )

    missing = [section for section in REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise ConfigError(
            "Faltan secciones obligatorias en la configuración: "
            f"{', '.join(missing)}. Secciones requeridas: {', '.join(REQUIRED_SECTIONS)}."
        )

    config = AppConfig.from_dict(raw)
    return config.apply_overrides(overrides)
