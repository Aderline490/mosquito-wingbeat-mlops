"""Train the baseline and the first model version. Run from the project root: python -m src.train"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.model import SEED, build_model, evaluate_model, fit, list_clips, make_dataset
from src.preprocessing import CLASSES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "wingbeat_model.keras"
BASELINE_PATH = MODELS / "baseline_model.keras"


def summary(name: str, metrics: dict) -> str:
    return f"{name}: accuracy {metrics['accuracy']:.4f}  f1 {metrics['f1_macro']:.4f}  loss {metrics['loss']:.4f}"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    splits = {name: list_clips(DATA / name) for name in ("train", "validation", "test")}
    print("clips: " + "  ".join(f"{k}={len(v[0])}" for k, v in splits.items()), flush=True)

    classes = np.arange(len(CLASSES))
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(splits["train"][1]))
    class_weight = {int(i): float(w) for i, w in zip(classes, weights)}
    val_ds = make_dataset(*splits["validation"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["baseline", "final"], help="train just one of the two models")
    only = parser.parse_args().only

    if only != "final":
        train_baseline(splits, val_ds, class_weight)
    if only != "baseline":
        train_final(splits, val_ds, class_weight)


def train_baseline(splits: dict, val_ds, class_weight: dict) -> None:
    # Baseline: same network and shuffled data, but no augmentation, no dropout, no L2.
    tf.keras.utils.set_random_seed(SEED)
    started = time.time()
    baseline = build_model(dropout=0.0, l2=0.0)
    baseline_history = fit(
        baseline, make_dataset(*splits["train"], shuffle=True), val_ds, BASELINE_PATH, epochs=30, class_weight=class_weight
    )
    baseline = tf.keras.models.load_model(BASELINE_PATH)
    baseline_metrics = {
        "training_seconds": round(time.time() - started, 1),
        "epochs_run": len(baseline_history["loss"]),
        "validation": evaluate_model(baseline, *splits["validation"]),
        "test": evaluate_model(baseline, *splits["test"]),
        "history": baseline_history,
    }
    (MODELS / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2))
    print(summary("baseline test", baseline_metrics["test"]), flush=True)


def train_final(splits: dict, val_ds, class_weight: dict) -> None:
    # Optimised model: augmentation, dropout, L2, learning-rate decay, early stopping.
    tf.keras.utils.set_random_seed(SEED)
    started = time.time()
    model = build_model()
    history = fit(
        model, make_dataset(*splits["train"], training=True), val_ds, MODEL_PATH, epochs=60, class_weight=class_weight
    )
    model = tf.keras.models.load_model(MODEL_PATH)
    metrics = {
        "version": 1,
        "training_seconds": round(time.time() - started, 1),
        "epochs_run": len(history["loss"]),
        "validation": evaluate_model(model, *splits["validation"]),
        "test": evaluate_model(model, *splits["test"]),
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "history.json").write_text(json.dumps(history, indent=2))
    print(summary("final validation", metrics["validation"]))
    print(summary("final test", metrics["test"]))


if __name__ == "__main__":
    main()
