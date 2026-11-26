"""Load the model in service and predict one image.

Command line use: python -m src.prediction path/to/recording.wav
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from src.config import ACTIVE_POINTER, LOW_CONFIDENCE, active_model
from src.preprocessing import CLASSES, INPUT_SHAPE, SAMPLE_RATE, AudioSource, load_audio, log_spectrogram, split_windows, wingbeat_frequency

RELOAD_CHECK_SECONDS = 2.0

DISPLAY_NAMES = {
    "an_arabiensis": "Anopheles arabiensis",
    "an_funestus": "Anopheles funestus",
    "culex_pipiens": "Culex pipiens complex",
    "background": "No mosquito",
}
MALARIA_VECTORS = {"an_arabiensis", "an_funestus"}
SPECTROGRAM_POOL = 4


def spectrogram_preview(spectrogram: np.ndarray) -> list:
    """First window's spectrogram, averaged over 4 frames, low frequencies first."""
    image = spectrogram[..., 0]
    usable = image.shape[1] - image.shape[1] % SPECTROGRAM_POOL
    pooled = image[:, :usable].reshape(image.shape[0], -1, SPECTROGRAM_POOL).mean(axis=2)
    return np.round(pooled, 2).tolist()


class Predictor:
    """Serves one model and picks up a newly promoted version automatically."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reloading = threading.Event()
        self._model: Optional[tf.keras.Model] = None
        self._infer = None
        self.version: Optional[int] = None
        self._pointer_mtime: Optional[int] = None
        self._next_check = 0.0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _pointer_state(self) -> Optional[int]:
        try:
            return ACTIVE_POINTER.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def load(self) -> None:
        """Load the active version, keeping the old model if loading fails."""
        with self._lock:
            pointer_mtime = self._pointer_state()
            active = active_model()
            model = tf.keras.models.load_model(active["path"], compile=False)

            @tf.function(input_signature=[tf.TensorSpec((None, *INPUT_SHAPE), tf.float32)])
            def infer(batch):
                return model(batch, training=False)

            infer(tf.zeros((1, *INPUT_SHAPE)))
            self._model, self._infer = model, infer
            self.version = int(active["version"])
            self._pointer_mtime = pointer_mtime

    def refresh(self) -> None:
        current = time.monotonic()
        if current < self._next_check:
            return
        self._next_check = current + RELOAD_CHECK_SECONDS
        if (not self.loaded or self._pointer_state() != self._pointer_mtime) and not self._reloading.is_set():
            self._reloading.set()
            threading.Thread(target=self._reload, daemon=True).start()

    def _reload(self) -> None:
        """Load a new version in the background while the old one keeps serving."""
        try:
            self.load()
        except Exception as exc:
            print(f"Model reload failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self._reloading.clear()

    def predict(self, source: AudioSource) -> dict:
        """Score every 2 s window of the recording and average the probabilities."""
        self.refresh()
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        signal = load_audio(source)
        windows = split_windows(signal)
        batch = np.stack([log_spectrogram(window) for window in windows])
        version = self.version
        started = time.perf_counter()
        per_window = self._infer(tf.constant(batch)).numpy()
        inference_ms = (time.perf_counter() - started) * 1000
        probabilities = per_window.mean(axis=0)

        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        return {
            "label": CLASSES[index],
            "display_name": DISPLAY_NAMES[CLASSES[index]],
            "confidence": round(confidence, 4),
            "low_confidence": confidence < LOW_CONFIDENCE,
            "malaria_vector": CLASSES[index] in MALARIA_VECTORS,
            "probabilities": {name: round(float(p), 4) for name, p in zip(CLASSES, probabilities)},
            "duration_seconds": round(signal.size / SAMPLE_RATE, 2),
            "windows": len(windows),
            "window_labels": [CLASSES[i] for i in per_window.argmax(axis=1)],
            "wingbeat_hz": round(wingbeat_frequency(signal), 1),
            "spectrogram": spectrogram_preview(batch[0]),
            "model_version": version,
            "inference_ms": round(inference_ms, 2),
        }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.prediction path/to/recording.wav")
    predictor = Predictor()
    predictor.load()
    result = predictor.predict(Path(sys.argv[1]))
    result.pop("spectrogram")
    print(json.dumps(result, indent=2))
