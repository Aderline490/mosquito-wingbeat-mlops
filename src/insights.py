"""Measure acoustic features of the training clips for the dashboard.

* wingbeat frequency: the wingbeat fundamental between 250 and 900 Hz, found
  with a harmonic product spectrum so harmonics are not mistaken for it
* brightness: the spectral centroid, the "centre of mass" of the sound
* tonal strength: share of sound energy that sits on the wingbeat tone and its
  first two harmonics. A buzzing mosquito is tonal, room noise is not

Energy below 150 Hz (mains and room hum) is ignored for brightness and tonal
strength, so they describe the insect and not the building.

Run from the project root: python -m src.insights
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import numpy as np

from src.config import BASE_MODEL_DIR, DATA_DIR
from src.preprocessing import ALLOWED_EXTENSIONS, CLASSES, SAMPLE_RATE, load_audio, wingbeat_frequency

OUTPUT = BASE_MODEL_DIR / "insights.json"
FEATURES = {
    "wingbeat_hz": {"title": "Wingbeat frequency", "unit": "Hz", "min": 250, "max": 900},
    "centroid_hz": {"title": "Brightness (spectral centroid)", "unit": "Hz", "min": 300, "max": 1500},
    "tonal_pct": {"title": "Tonal strength", "unit": "% of energy on the wingbeat and its harmonics", "min": 0, "max": 60},
}
BINS = 12
HARMONIC_HALF_WIDTH = 15.0
HUM_CUTOFF = 150.0


def measure(path: str) -> Dict[str, float]:
    signal = load_audio(path)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / SAMPLE_RATE)
    audible = (freqs >= HUM_CUTOFF) & (freqs <= 4000)
    total = spectrum[audible].sum() + 1e-12

    fundamental = wingbeat_frequency(signal)
    harmonic = np.zeros_like(freqs, dtype=bool)
    for multiple in (1, 2, 3):
        harmonic |= np.abs(freqs - multiple * fundamental) <= HARMONIC_HALF_WIDTH
    return {
        "wingbeat_hz": fundamental,
        "centroid_hz": float((freqs[audible] * spectrum[audible]).sum() / total),
        "tonal_pct": float(100 * spectrum[harmonic & audible].sum() / total),
    }


def add_stories(features: dict) -> None:
    """Interpretations, written from the measured medians."""
    names = {"an_arabiensis": "An. arabiensis", "an_funestus": "An. funestus", "culex_pipiens": "Culex"}

    def med(key: str) -> Dict[str, float]:
        return {name: features[key]["classes"][name]["median"] for name in CLASSES}

    w, c, t = med("wingbeat_hz"), med("centroid_hz"), med("tonal_pct")
    species = sorted(names, key=lambda name: -w[name])
    order = ", then ".join(f"{names[name]} ({w[name]:.0f} Hz)" for name in species)
    features["wingbeat_hz"]["story"] = (
        f"The typical wingbeat tone is highest for {order}. "
        "The ranges overlap a lot, so pitch alone cannot name the species. The model also has to use the "
        "harmonics and how the sound changes over the 2 seconds, which is what the spectrogram CNN learns."
    )
    mosquito_c = [c[name] for name in names]
    features["centroid_hz"]["story"] = (
        f"Mosquito clips are brighter, with a typical centre of {min(mosquito_c):.0f} to {max(mosquito_c):.0f} Hz, "
        f"against {c['background']:.0f} Hz for background noise. The three species are almost identical here, "
        "so brightness helps answer whether a mosquito is present, not which one."
    )
    mosquito_t = [t[name] for name in names]
    features["tonal_pct"]["story"] = (
        f"In a typical mosquito clip {min(mosquito_t):.0f} to {max(mosquito_t):.0f}% of the energy sits on the "
        f"wingbeat and its harmonics, about twice the {t['background']:.0f}% of background noise. A buzz is a "
        "clean tone with overtones, while room noise is spread out. Like brightness, it separates mosquito from "
        "no mosquito better than it separates species."
    )


def summarise(values_by_class: Dict[str, np.ndarray], low: float, high: float) -> dict:
    edges = np.linspace(low, high, BINS + 1)
    output = {"edges": [round(float(edge), 2) for edge in edges], "classes": {}}
    for name, values in values_by_class.items():
        counts, _ = np.histogram(np.clip(values, low, high), bins=edges)
        output["classes"][name] = {
            "n": int(values.size),
            "median": round(float(np.median(values)), 2),
            "q1": round(float(np.percentile(values, 25)), 2),
            "q3": round(float(np.percentile(values, 75)), 2),
            "share_pct": [round(100 * float(count) / values.size, 2) for count in counts],
        }
    return output


def main() -> None:
    paths, labels = [], []
    for name in CLASSES:
        for path in sorted((DATA_DIR / "train" / name).iterdir()):
            if path.suffix.lower() in ALLOWED_EXTENSIONS:
                paths.append(str(path))
                labels.append(name)

    with ProcessPoolExecutor() as pool:
        measured = list(pool.map(measure, paths, chunksize=32))

    output = {
        "clips_total": len(paths),
        "clip_seconds": 2,
        "class_counts": {name: labels.count(name) for name in CLASSES},
        "features": {},
    }
    for key, meta in FEATURES.items():
        values = {name: np.array([row[key] for label, row in zip(labels, measured) if label == name]) for name in CLASSES}
        output["features"][key] = {**meta, **summarise(values, meta["min"], meta["max"])}
    add_stories(output["features"])
    OUTPUT.write_text(json.dumps(output, indent=2))

    for key in FEATURES:
        print(key)
        for name in CLASSES:
            values = np.array([row[key] for label, row in zip(labels, measured) if label == name])
            print(f"  {name:15} median={np.median(values):8.1f} q1={np.percentile(values, 25):8.1f} "
                  f"q3={np.percentile(values, 75):8.1f} p1={np.percentile(values, 1):8.1f} p99={np.percentile(values, 99):8.1f}")


if __name__ == "__main__":
    main()
