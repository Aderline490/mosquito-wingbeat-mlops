"""Retrain the model in service on the original data plus uploaded recordings.

The current model is the pre-trained starting point. The new version only goes
into service if it does at least as well on the validation set.

Run by the API, or by hand: python -m src.retrain --job-id 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src import database as db
from src.config import DATA_DIR, RETRAIN_EPOCHS, UPLOAD_DIR, VERSIONS_DIR, active_model, ensure_dirs, set_active_model
from src.model import SEED, compile_model, evaluate_model, list_clips, make_dataset, training_callbacks
from src.preprocessing import CLASS_TO_INDEX, CLASSES, SAMPLE_RATE, AudioError, load_audio, split_windows

RETRAIN_LEARNING_RATE = 1e-4
WINDOW_DIR = UPLOAD_DIR / "windows"


class Heartbeat(tf.keras.callbacks.Callback):
    """Records progress so the UI can show it and stale jobs can be detected."""

    def __init__(self, job_id: int, epochs: int) -> None:
        super().__init__()
        self.job_id, self.epochs = job_id, epochs

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss", float("nan"))
        db.update_job(self.job_id, stage=f"training epoch {epoch + 1} of {self.epochs} (val loss {val_loss:.4f})")


def holdout_hashes() -> set:
    """Hashes of validation and test clips, which must never be trained on."""
    hashes = set()
    for split in ("validation", "test"):
        for path in list_clips(DATA_DIR / split)[0]:
            hashes.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return hashes


def prepare_uploads(rows) -> tuple:
    """Validate uploads and cut each one into 2 s training clips.

    Rejects unreadable files and any recording copied from the validation or
    test set, so evaluation stays honest.
    """
    holdout = holdout_hashes()
    paths, labels, ids, rejected = [], [], [], []
    for row in rows:
        if row["sha256"] in holdout:
            rejected.append((row["id"], "Recording is in the validation or test set."))
            continue
        if row["label"] not in CLASS_TO_INDEX:
            rejected.append((row["id"], "Unknown class label."))
            continue
        try:
            windows = split_windows(load_audio(row["stored_path"]))
        except (AudioError, FileNotFoundError) as exc:
            rejected.append((row["id"], str(exc)))
            continue
        folder = WINDOW_DIR / row["label"]
        folder.mkdir(parents=True, exist_ok=True)
        for k, window in enumerate(windows):
            clip = folder / f"{row['sha256'][:16]}_{k:02d}.wav"
            sf.write(clip, window, SAMPLE_RATE, subtype="PCM_16")
            paths.append(str(clip))
            labels.append(CLASS_TO_INDEX[row["label"]])
        ids.append(row["id"])
    return paths, labels, ids, rejected


def is_better(candidate: dict, current: dict) -> bool:
    """Prefer higher macro F1, using lower loss to break ties."""
    if candidate["f1_macro"] != current["f1_macro"]:
        return candidate["f1_macro"] > current["f1_macro"]
    return candidate["loss"] <= current["loss"]


def run(job_id: int) -> None:
    tf.keras.utils.set_random_seed(SEED)
    ensure_dirs()
    active = active_model()
    db.update_job(job_id, status="running", stage="preprocessing uploaded recordings", from_version=active["version"])

    rows = db.pending_uploads()
    new_paths, new_labels, new_ids, rejected = prepare_uploads(rows)
    if rejected:
        db.mark_rejected(rejected)
    if not new_paths:
        db.update_job(job_id, status="failed", stage="failed", message="No usable uploaded recordings.", promoted=0)
        return

    base_paths, base_labels = list_clips(DATA_DIR / "train")
    val_paths, val_labels = list_clips(DATA_DIR / "validation")
    test_paths, test_labels = list_clips(DATA_DIR / "test")
    train_paths, train_labels = base_paths + new_paths, base_labels + new_labels

    classes = np.arange(len(CLASSES))
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(train_labels))
    class_weight = {int(index): float(value) for index, value in zip(classes, weights)}

    db.update_job(job_id, stage="loading current model as pre-trained base")
    current = tf.keras.models.load_model(active["path"])
    current_val = evaluate_model(current, val_paths, val_labels)
    current_test = evaluate_model(current, test_paths, test_labels)

    candidate_path = VERSIONS_DIR / f"candidate_job{job_id}.keras"
    compile_model(current, RETRAIN_LEARNING_RATE)
    callbacks = training_callbacks(candidate_path, patience=3) + [Heartbeat(job_id, RETRAIN_EPOCHS)]
    history = current.fit(
        make_dataset(train_paths, train_labels, training=True),
        validation_data=make_dataset(val_paths, val_labels),
        epochs=RETRAIN_EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    db.update_job(job_id, stage="evaluating new model against current model")
    candidate = tf.keras.models.load_model(candidate_path)
    candidate_val = evaluate_model(candidate, val_paths, val_labels)
    candidate_test = evaluate_model(candidate, test_paths, test_labels)
    promoted = is_better(candidate_val, current_val)

    result = {
        "recordings_used": len(new_ids),
        "clips_added": len(new_paths),
        "recordings_rejected": len(rejected),
        "epochs_run": len(history.history["loss"]),
        "current": {"validation": current_val, "test": current_test},
        "candidate": {"validation": candidate_val, "test": candidate_test},
    }

    if promoted:
        version = int(active["version"]) + 1
        model_path = VERSIONS_DIR / f"model_v{version}.keras"
        metrics_path = VERSIONS_DIR / f"metrics_v{version}.json"
        os.replace(candidate_path, model_path)
        metrics_path.write_text(
            json.dumps({"version": version, "validation": candidate_val, "test": candidate_test}, indent=2)
        )
        set_active_model(version, model_path, metrics_path)
        db.mark_used(new_ids, version)
        message = f"Version {version} is now in service."
    else:
        version = int(active["version"])
        candidate_path.unlink(missing_ok=True)
        message = f"New model did not beat version {version} on validation, so version {version} stays in service."

    db.update_job(
        job_id, status="completed", stage="done", promoted=int(promoted), to_version=version,
        message=message, result=result,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    job_id = parser.parse_args().job_id
    try:
        run(job_id)
    except Exception as exc:
        traceback.print_exc()
        db.update_job(job_id, status="failed", stage="failed", message=f"{type(exc).__name__}: {exc}", promoted=0)
        raise


if __name__ == "__main__":
    main()
