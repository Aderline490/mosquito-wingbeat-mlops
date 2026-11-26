"""Model creation, training and evaluation on denoised log spectrograms."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing import ALLOWED_EXTENSIONS, CLASS_TO_INDEX, CLASSES, INPUT_SHAPE, preprocess_window

BATCH_SIZE = 32
SEED = 42


def list_clips(split_dir: Path) -> Tuple[List[str], List[int]]:
    """List audio clip paths and integer labels from class-named folders."""
    paths, labels = [], []
    for name in CLASSES:
        folder = Path(split_dir) / name
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(str(path))
                labels.append(CLASS_TO_INDEX[name])
    return paths, labels


def augment(spectrogram: tf.Tensor) -> tf.Tensor:
    """Random time shift plus SpecAugment style frequency and time masking."""
    spectrogram = tf.roll(spectrogram, shift=tf.random.uniform([], 0, INPUT_SHAPE[1], dtype=tf.int32), axis=1)
    rows, frames = INPUT_SHAPE[0], INPUT_SHAPE[1]
    for size, length, axis in ((8, rows, 0), (30, frames, 1)):
        width = tf.random.uniform([], 0, size + 1, dtype=tf.int32)
        start = tf.random.uniform([], 0, length - width + 1, dtype=tf.int32)
        positions = tf.range(length)
        keep = tf.logical_or(positions < start, positions >= start + width)
        shape = [length, 1, 1] if axis == 0 else [1, length, 1]
        spectrogram = spectrogram * tf.reshape(tf.cast(keep, tf.float32), shape)
    return spectrogram


def make_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    training: bool = False,
    batch_size: int = BATCH_SIZE,
    shuffle: Optional[bool] = None,
) -> tf.data.Dataset:
    """tf.data pipeline that uses the shared preprocessing code.

    training=True shuffles and augments. shuffle=True on its own shuffles
    without augmenting, which is what the baseline uses. Clips are stored
    class by class, so training without shuffling gives one-class batches.
    """
    shuffle = training if shuffle is None else shuffle

    def decode(path: bytes) -> np.ndarray:
        return preprocess_window(path.decode("utf-8"))

    def load(path, label):
        spectrogram = tf.numpy_function(decode, [path], tf.float32)
        spectrogram.set_shape(INPUT_SHAPE)
        return spectrogram, label

    dataset = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels, dtype="int32")))
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    if training:
        dataset = dataset.map(lambda x, y: (augment(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(dropout: float = 0.3, l2: float = 1e-4, filters: Sequence[int] = (32, 64, 128)) -> tf.keras.Model:
    """Small CNN over the spectrogram. Pass dropout=0 and l2=0 for the baseline."""
    regularizer = tf.keras.regularizers.l2(l2) if l2 else None
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="spectrogram")
    x = inputs
    for block, count in enumerate(filters, start=1):
        for conv in (1, 2):
            x = tf.keras.layers.Conv2D(
                count, 3, padding="same", use_bias=False, kernel_regularizer=regularizer, name=f"block{block}_conv{conv}"
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"block{block}_bn{conv}")(x)
            x = tf.keras.layers.ReLU(name=f"block{block}_relu{conv}")(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="pool")(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", kernel_regularizer=regularizer, name="classifier")(x)
    return tf.keras.Model(inputs, outputs, name="wingbeat_classifier")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def training_callbacks(checkpoint: Path, patience: int = 8, best_so_far: Optional[float] = None) -> list:
    """Early stopping, learning-rate decay and a checkpoint of the best epoch."""
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, initial_value_threshold=best_so_far
        ),
    ]


def fit(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    checkpoint: Path,
    epochs: int = 60,
    learning_rate: float = 1e-3,
    class_weight: Optional[Dict[int, float]] = None,
    extra_callbacks: Optional[list] = None,
) -> Dict[str, list]:
    compile_model(model, learning_rate)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks(checkpoint) + list(extra_callbacks or []),
        class_weight=class_weight,
        verbose=2,
    )
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def evaluate_probabilities(y_true: Sequence[int], probabilities: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = probabilities.argmax(axis=1)
    class_ids = list(range(len(CLASSES)))
    return {
        "samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, probabilities, labels=class_ids)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, probabilities, multi_class="ovr", labels=class_ids)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_ids).tolist(),
        "per_class": classification_report(
            y_true, y_pred, labels=class_ids, target_names=CLASSES, output_dict=True, zero_division=0
        ),
    }


def evaluate_model(model: tf.keras.Model, paths: Sequence[str], labels: Sequence[int]) -> dict:
    """Evaluation metrics on a labelled set of 2 s clips."""
    probabilities = model.predict(make_dataset(paths, labels), verbose=0)
    return evaluate_probabilities(labels, probabilities)
