"""FastAPI service. Run from the project root: uvicorn src.api:app --port 8000"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src.config import (
    AUTO_RETRAIN,
    BASE_MODEL_DIR,
    LOG_DIR,
    MAX_FILES_PER_UPLOAD,
    RETRAIN_AUTO_THRESHOLD,
    RETRAIN_MIN_FILES,
    ROOT,
    UPLOAD_DIR,
    active_metrics,
    active_model,
)
from src.prediction import Predictor
from src.preprocessing import ALLOWED_EXTENSIONS, CLASSES, MAX_UPLOAD_BYTES, SAMPLE_RATE, AudioError, load_audio, split_windows

predictor = Predictor()
started_at = datetime.now(timezone.utc)
started_clock = time.monotonic()
stats_lock = threading.Lock()
stats = {"requests": 0, "errors": 0, "predictions": 0}


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    try:
        predictor.load()
    except Exception as exc:  # stay up so /health can report the problem
        print(f"Could not load model at startup: {exc}", file=sys.stderr, flush=True)
    yield


app = FastAPI(
    title="Mosquito Wingbeat API",
    description="Identify malaria-carrying mosquitoes from the sound of their wingbeat, upload new recordings and retrain the model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        with stats_lock:
            stats["requests"] += 1
            stats["errors"] += 1
        raise
    with stats_lock:
        stats["requests"] += 1
        stats["errors"] += int(response.status_code >= 500)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    response.headers["X-Container-Id"] = socket.gethostname()
    return response


def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Protect upload and retrain when ADMIN_TOKEN is set on a public deployment."""
    token = os.getenv("ADMIN_TOKEN")
    if token and not secrets.compare_digest((x_api_key or "").encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def read_limited(file: UploadFile) -> bytes:
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise AudioError("Recording is larger than the 20 MB limit.")
    return content


def start_retraining(trigger: str, pending: int) -> int:
    """Create a job and run it in a separate process so predictions keep flowing."""
    job_id = db.create_job(trigger, pending, int(active_model()["version"]))
    try:
        log = open(LOG_DIR / f"retrain_job_{job_id}.log", "ab")
        try:
            subprocess.Popen(
                [sys.executable, "-m", "src.retrain", "--job-id", str(job_id)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()
    except OSError as exc:
        db.update_job(job_id, status="failed", stage="failed", message=f"Could not start worker: {exc}")
        raise HTTPException(status_code=500, detail="Could not start retraining.") from exc
    return job_id


@app.get("/health", tags=["Monitoring"])
def health() -> dict:
    predictor.refresh()
    with stats_lock:
        counters = dict(stats)
    return {
        "status": "ok" if predictor.loaded else "degraded",
        "model_loaded": predictor.loaded,
        "model_version": predictor.version,
        "uptime_seconds": round(time.monotonic() - started_clock, 1),
        "started_at": started_at.isoformat(timespec="seconds"),
        "container": socket.gethostname(),
        **counters,
    }


@app.get("/classes", tags=["Model"])
def classes() -> dict:
    return {"classes": CLASSES}


@app.get("/model/metrics", tags=["Model"])
def model_metrics() -> dict:
    metrics = active_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for the active model.")
    return metrics


@app.get("/insights", tags=["Model"])
def insights() -> dict:
    path = BASE_MODEL_DIR / "insights.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Insights have not been generated.")
    return json.loads(path.read_text())


@app.post("/predict", tags=["Model"])
def predict(file: UploadFile = File(...)) -> dict:
    try:
        result = predictor.predict(read_limited(file))
    except AudioError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with stats_lock:
        stats["predictions"] += 1
    return result


@app.post("/upload", tags=["Retraining"], dependencies=[Depends(require_admin)])
def upload(label: str = Form(...), files: List[UploadFile] = File(...)) -> dict:
    """Validate a bulk upload for one class and save it for retraining."""
    if label not in CLASSES:
        raise HTTPException(status_code=422, detail=f"Label must be one of {CLASSES}.")
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(status_code=413, detail=f"Upload at most {MAX_FILES_PER_UPLOAD} files at a time.")

    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    saved, duplicates, rejected = 0, 0, []
    for file in files:
        name = Path(file.filename or "recording").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            rejected.append({"file": name, "reason": "Only WAV, FLAC and OGG recordings are accepted."})
            continue
        try:
            content = read_limited(file)
            signal = load_audio(content)
            duration, windows = signal.size / SAMPLE_RATE, len(split_windows(signal))
        except AudioError as exc:
            rejected.append({"file": name, "reason": str(exc)})
            continue

        digest = hashlib.sha256(content).hexdigest()
        target = folder / f"{digest}{suffix}"
        temporary = target.with_suffix(f"{suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        if db.add_upload(name, str(target), label, digest, duration, windows):
            saved += 1
        else:
            duplicates += 1

    summary = db.upload_summary()
    auto_job = None
    if AUTO_RETRAIN and saved and summary["pending"] >= RETRAIN_AUTO_THRESHOLD:
        try:
            auto_job = start_retraining("auto", summary["pending"])
        except db.JobAlreadyActive:
            auto_job = None
    return {
        "saved": saved,
        "duplicates": duplicates,
        "rejected": rejected,
        "pending_total": summary["pending"],
        "auto_retrain_job": auto_job,
    }


@app.get("/uploads/summary", tags=["Retraining"])
def uploads_summary() -> dict:
    summary = db.upload_summary()
    return {
        **summary,
        "minimum_to_retrain": RETRAIN_MIN_FILES,
        "auto_retrain_enabled": AUTO_RETRAIN,
        "auto_retrain_threshold": RETRAIN_AUTO_THRESHOLD,
        "retrain_recommended": summary["pending"] >= RETRAIN_AUTO_THRESHOLD,
        "api_key_required": bool(os.getenv("ADMIN_TOKEN")),
    }


@app.post("/retrain", status_code=202, tags=["Retraining"], dependencies=[Depends(require_admin)])
def retrain() -> dict:
    pending = db.upload_summary()["pending"]
    if pending < RETRAIN_MIN_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload at least {RETRAIN_MIN_FILES} new recordings first. Currently pending: {pending}.",
        )
    try:
        job_id = start_retraining("manual", pending)
    except db.JobAlreadyActive as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued", "new_recordings": pending}


@app.get("/retrain/jobs", tags=["Retraining"])
def retrain_jobs(limit: int = 10) -> dict:
    return {"jobs": db.recent_jobs(max(1, min(limit, 50)))}


@app.get("/retrain/{job_id}", tags=["Retraining"])
def retrain_status(job_id: int) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "ui" / "index.html")


app.mount("/ui", StaticFiles(directory=ROOT / "ui"), name="ui")
