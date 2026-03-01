import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnxruntime as ort

# Project root directory
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "public" / "models"
MANIFEST_PATH = MODEL_DIR / "manifest.json"


@dataclass
class ModelInfo:
    """Model metadata."""

    name: str
    md5: str
    size_mb: float
    format: str
    updated_at: str
    path: Path = field(init=False)
    _session: ort.InferenceSession | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.path = MODEL_DIR / self.name

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load(self, providers: list[str] | None = None) -> ort.InferenceSession:
        """Load model into session."""
        if self._session is None:
            if providers is None:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.path, providers=providers)
        return self._session

    def unload(self) -> None:
        """Unload model to free memory."""
        self._session = None

    def verify(self) -> bool:
        """Verify model file integrity."""
        if not self.path.exists():
            return False
        return compute_md5(self.path) == self.md5


@dataclass
class ModelRegistry:
    """Model registry with manifest-based management."""

    models: dict[str, ModelInfo] = field(default_factory=dict)
    version: str = "1.0.0"
    updated_at: str = ""
    _aliases: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_manifest(cls, manifest_path: Path | None = None) -> "ModelRegistry":
        """Load registry from manifest.json."""
        if manifest_path is None:
            manifest_path = MANIFEST_PATH

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        registry = cls(
            version=data.get("version", "1.0.0"),
            updated_at=data.get("updated_at", ""),
        )

        for m in data.get("models", []):
            info = ModelInfo(
                name=m["name"],
                md5=m["md5"],
                size_mb=m["size_mb"],
                format=m.get("format", "onnx"),
                updated_at=m.get("updated_at", ""),
            )
            # Full filename as primary key (e.g., "face_detection_yunet_2023mar")
            full_key = info.name.rsplit(".", 1)[0]
            registry.models[full_key] = info

            # Create alias from base model name (e.g., "face_detection_yunet")
            base_key = cls._extract_base_name(full_key)
            if base_key != full_key:
                registry._aliases[base_key] = full_key

        return registry

    @staticmethod
    def _extract_base_name(full_name: str) -> str:
        """Extract base model name by removing date suffix."""
        # e.g., "face_detection_yunet_2023mar" -> "face_detection_yunet"
        parts = full_name.rsplit("_", 1)
        if (
            len(parts) == 2
            and len(parts[1]) == 7
            and parts[1][-3:]
            in ("mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
        ):
            return parts[0]
        return full_name

    def get(self, name: str) -> ModelInfo:
        """Get model info by name (supports aliases)."""
        # Check direct match first
        if name in self.models:
            return self.models[name]
        # Check alias
        if name in self._aliases:
            return self.models[self._aliases[name]]
        available = list(self.models.keys()) + list(self._aliases.keys())
        raise KeyError(f"Unknown model: '{name}', available: {available}")

    def load(
        self, name: str, providers: list[str] | None = None
    ) -> ort.InferenceSession:
        """Load model and return session."""
        info = self.get(name)
        return info.load(providers)

    def run(self, name: str, input_data: Any) -> list:
        """Run inference on model."""
        session = self.load(name)
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: input_data})

    def list_models(self) -> list[str]:
        """List all available model names."""
        return list(self.models.keys())

    def get_model_info(self, name: str) -> dict:
        """Get model metadata as dict."""
        info = self.get(name)
        return {
            "name": info.name,
            "size_mb": info.size_mb,
            "format": info.format,
            "loaded": info.is_loaded,
            "verified": info.verify(),
        }


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get or create global registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry.from_manifest()
    return _registry


def load_model(name: str) -> ort.InferenceSession:
    """Load model by name (backward compatible API)."""
    return get_registry().load(name)


def run_model(name: str, input_data: Any) -> list:
    """Run inference (backward compatible API)."""
    return get_registry().run(name, input_data)
