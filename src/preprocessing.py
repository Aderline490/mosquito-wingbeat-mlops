"""Audio preprocessing shared by training, prediction and retraining.

Every recording goes through the same steps so the model sees identical input
in the notebook, in the API and during retraining:

1. decode the file (WAV, FLAC or OGG) and mix it down to mono
2. resample to 8 kHz, which keeps the wingbeat and its harmonics (below 4 kHz)
3. cut it into 2 second windows (a short final piece is zero padded)
4. turn each window into a fine log spectrogram (1024 point FFT, so each row
   is 7.8 Hz wide) covering 150 to 2000 Hz, where the wingbeat tone and its
   first harmonics sit. Species differ by only 30 to 60 Hz in wingbeat, so a
   coarse mel scale (about 30 Hz per band here) blurs them together
5. subtract each frequency's median over the window. Steady noise from the
   recorder, the cup or the room stays at the same frequency the whole time,
   so this removes it and leaves what changes, the buzzing insect
6. standardise to zero mean and unit variance, so the model listens to the
   shape of the sound and not to how loud the recorder was
"""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 8000
WINDOW_SECONDS = 2.0
WINDOW = int(SAMPLE_RATE * WINDOW_SECONDS)
N_FFT = 1024
HOP = 128
F_MIN, F_MAX = 150.0, 2000.0
FREQUENCIES = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
BAND = (FREQUENCIES >= F_MIN) & (FREQUENCIES <= F_MAX)
N_BINS = int(BAND.sum())
FRAMES = 1 + (WINDOW - N_FFT) // HOP
INPUT_SHAPE = (N_BINS, FRAMES, 1)

MIN_SECONDS = 1.0
MAX_SECONDS = 120.0
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg"}

CLASSES = ["an_arabiensis", "an_funestus", "culex_pipiens", "background"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASSES)}

AudioSource = Union[str, Path, bytes]


class AudioError(ValueError):
    """Raised when a file cannot be used as a mosquito recording."""


HANN = np.hanning(N_FFT).astype("float32")


def load_audio(source: AudioSource) -> np.ndarray:
    """Decode a file path or raw bytes into a mono 8 kHz float signal."""
    if isinstance(source, (bytes, bytearray)) and len(source) > MAX_UPLOAD_BYTES:
        raise AudioError("Recording is larger than the 20 MB limit.")
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source)
    try:
        signal, rate = sf.read(handle, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError, ValueError, TypeError) as exc:
        raise AudioError(f"Could not read audio. Use WAV, FLAC or OGG. ({exc})") from exc

    signal = signal.mean(axis=1)
    if signal.size == 0 or rate <= 0:
        raise AudioError("The recording is empty.")
    seconds = signal.size / rate
    if seconds < MIN_SECONDS:
        raise AudioError(f"Recording is {seconds:.1f} s long. It must be at least {MIN_SECONDS:.0f} s.")
    signal = signal[: int(MAX_SECONDS * rate)]
    if rate != SAMPLE_RATE:
        divisor = gcd(int(rate), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // divisor, int(rate) // divisor).astype("float32")
    if not np.isfinite(signal).all() or np.abs(signal).max() < 1e-5:
        raise AudioError("The recording is silent or damaged.")
    return signal - signal.mean()


def split_windows(signal: np.ndarray) -> List[np.ndarray]:
    """Cut into 2 s windows. A final piece of at least 1 s is zero padded."""
    windows = []
    for start in range(0, signal.size, WINDOW):
        piece = signal[start : start + WINDOW]
        if piece.size < WINDOW:
            if piece.size < MIN_SECONDS * SAMPLE_RATE and windows:
                break
            piece = np.pad(piece, (0, WINDOW - piece.size))
        windows.append(piece.astype("float32"))
    return windows


def log_spectrogram(window: np.ndarray) -> np.ndarray:
    """Denoised, standardised log spectrogram of one 2 s window, shape INPUT_SHAPE."""
    frames = np.lib.stride_tricks.sliding_window_view(window, N_FFT)[::HOP][:FRAMES]
    power = np.abs(np.fft.rfft(frames * HANN, axis=1)) ** 2
    spec = np.log(power[:, BAND].T + 1e-10)
    spec = spec - np.median(spec, axis=1, keepdims=True)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec[..., np.newaxis].astype("float32")


def preprocess(source: AudioSource) -> np.ndarray:
    """Spectrograms for every window of one recording, shape (windows, *INPUT_SHAPE)."""
    return np.stack([log_spectrogram(window) for window in split_windows(load_audio(source))])


def preprocess_window(source: AudioSource) -> np.ndarray:
    """Spectrogram of the first window only. Used for 2 s clips in the dataset."""
    return log_spectrogram(split_windows(load_audio(source))[0])


def wingbeat_frequency(signal: np.ndarray, low: float = 250.0, high: float = 900.0) -> float:
    """Wingbeat fundamental in Hz, found with a harmonic product spectrum.

    A buzzing wing produces a tone plus harmonics at 2x and 3x. Scoring each
    candidate by the energy at f, 2f and 3f together stops the estimate from
    jumping to a harmonic or to a stray noise peak.
    """
    spectrum = np.log(np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2 + 1e-12)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / SAMPLE_RATE)
    candidates = np.arange(low, high, 1.0)
    score = sum(np.interp(multiple * candidates, freqs, spectrum) for multiple in (1, 2, 3))
    return float(candidates[score.argmax()])


def preprocess_many(paths: Iterable[Path]) -> Tuple[List[Path], List[dict]]:
    """Check several files, returning the usable paths and the skipped ones."""
    kept, skipped = [], []
    for path in paths:
        try:
            load_audio(path)
            kept.append(Path(path))
        except AudioError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
    return kept, skipped
