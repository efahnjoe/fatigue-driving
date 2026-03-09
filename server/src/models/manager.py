from pathlib import Path
from typing import Any, Optional

import onnxruntime as ort

BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "public" / "models"

# Default execution providers (ordered by priority)
AVAILABLE_PROVIDERS = ort.get_available_providers()
DEFAULT_PROVIDERS = [
    p
    for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if p in AVAILABLE_PROVIDERS
]

if not DEFAULT_PROVIDERS:
    DEFAULT_PROVIDERS = ["CPUExecutionProvider"]


class ModelManager:
    _instance: Optional["ModelManager"] = None
    _sessions: dict[str, ort.InferenceSession]

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sessions = {}
        return cls._instance

    def load(
        self, model_name: str, providers: Optional[list[str]] = None
    ) -> ort.InferenceSession:
        if model_name in self._sessions:
            return self._sessions[model_name]

        model_path = MODELS_DIR / model_name

        if not model_path.exists():
            raise FileNotFoundError(f"Cannot find model in: {model_path}\n")

        session = ort.InferenceSession(
            str(model_path), providers=providers or DEFAULT_PROVIDERS
        )
        self._sessions[model_name] = session
        return session

    def run(self, model_name: str, input_data: Any) -> list:
        session = self.load(model_name)
        input_name = session.get_inputs()[0].name
        return session.run(None, {input_name: input_data})

    def clear_cache(self):
        self._sessions.clear()


def load_model(
    model_name: str, providers: Optional[list[str]] = None
) -> ort.InferenceSession:
    return ModelManager().load(model_name, providers)


def run_model(model_name: str, input_data: Any) -> list:
    return ModelManager().run(model_name, input_data)


if __name__ == "__main__":
    try:
        session = load_model("face_detection_yunet_2023mar.onnx")
        print("Model loaded successfully!")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
