#!/usr/bin/env python3
"""Generate manifest.json for ONNX models."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str
    md5: str
    size_mb: float = Field(..., description="Size in MB")
    format: str = "onnx"
    updated_at: str


class Manifest(BaseModel):
    """Model manifest."""

    version: str
    updated_at: str
    models: list[ModelInfo]


MODEL_DIR = Path(__file__).parent.parent / "public" / "models"
MANIFEST_VERSION = "1.0.0"


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def scan_models(model_dir: Path) -> list[ModelInfo]:
    """Scan directory for ONNX models and extract metadata."""
    models = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for path in sorted(model_dir.glob("*.onnx")):
        size_mb = round(path.stat().st_size / 1024 / 1024, 2)
        models.append(
            ModelInfo(
                name=path.name,
                md5=compute_md5(path),
                size_mb=size_mb,
                updated_at=timestamp,
            )
        )

    return models


def generate_manifest(model_dir: Path, output_path: Path) -> Manifest:
    """Generate and save manifest file."""
    models = scan_models(model_dir)
    timestamp = datetime.now(timezone.utc).isoformat()

    manifest = Manifest(
        version=MANIFEST_VERSION,
        updated_at=timestamp,
        models=models,
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2, ensure_ascii=False))

    return manifest


def main() -> None:
    """Main entry point."""
    manifest_path = MODEL_DIR / "manifest.json"
    manifest = generate_manifest(MODEL_DIR, manifest_path)
    print(f"Generated manifest.json with {len(manifest.models)} models")


if __name__ == "__main__":
    main()
