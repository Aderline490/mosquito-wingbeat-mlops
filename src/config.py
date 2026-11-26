"""Paths, settings and the pointer to the model version currently in service."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
BASE_MODEL_DIR = Path(os.getenv("BASE_MODEL_DIR", ROOT / "models"))
BASE_MODEL = BASE_MODEL_DIR / "wingbeat_model.keras"
BASE_METRICS = BASE_MODEL_DIR / "metrics.json"

# Shared between containers through a Docker volume.
STATE_DIR = Path(os.getenv("STATE_DIR", ROOT / "storage"))
DB_PATH = STATE_DIR / "app.db"
UPLOAD_DIR = STATE_DIR / "uploads"
VERSIONS_DIR = STATE_DIR / "versions"
LOG_DIR = STATE_DIR / "logs"
ACTIVE_POINTER = STATE_DIR / "active_model.json"

RETRAIN_MIN_FILES = int(os.getenv("RETRAIN_MIN_FILES", "6"))
RETRAIN_AUTO_THRESHOLD = int(os.getenv("RETRAIN_AUTO_THRESHOLD", "20"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() == "true"
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
STALE_JOB_MINUTES = int(os.getenv("STALE_JOB_MINUTES", "30"))
MAX_FILES_PER_UPLOAD = 200
LOW_CONFIDENCE = 0.60


def ensure_dirs() -> None:
    for folder in (STATE_DIR, UPLOAD_DIR, VERSIONS_DIR, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def active_model() -> dict:
    """Return the version in service, falling back to the original model."""
    if ACTIVE_POINTER.exists():
        return json.loads(ACTIVE_POINTER.read_text())
    return {"version": 1, "path": str(BASE_MODEL), "metrics_path": str(BASE_METRICS)}


def set_active_model(version: int, path: Path, metrics_path: Path) -> None:
    """Switch versions atomically so a container never reads a half-written pointer."""
    temporary = ACTIVE_POINTER.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"version": version, "path": str(path), "metrics_path": str(metrics_path)}))
    os.replace(temporary, ACTIVE_POINTER)


def active_metrics() -> dict:
    path = Path(active_model()["metrics_path"])
    return json.loads(path.read_text()) if path.exists() else {}
