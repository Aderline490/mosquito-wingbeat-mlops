"""Pick one held-out test recording per class for the demo and the load test.

Only recordings from the test split are considered, so the model never saw
them. Each candidate is cut to its first 10 seconds, saved as 16 bit WAV at
its original sample rate, and kept only if the model labels it correctly
with every window agreeing. The most confident one per class is written to
demo/predict/demo_{class}.wav.

It also writes a few labelled training-split recordings to demo/retrain/{class}/
for showing the upload and retrain flow. They come from training nights only,
so the retraining test never touches validation or test data.

Run from the project root: python -m scripts.pick_demos
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from scripts.prepare_data import member_index
from src.prediction import Predictor
from src.preprocessing import CLASSES

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "humbugdb"
OUT = ROOT / "demo" / "predict"
RETRAIN_OUT = ROOT / "demo" / "retrain"
RETRAIN_PER_CLASS = 4
SECONDS = 10
CANDIDATES = 20


def clip(index: dict, rec: str) -> bytes:
    zip_path, member = index[rec]
    with zipfile.ZipFile(zip_path) as archive:
        signal, rate = sf.read(io.BytesIO(archive.read(member)), dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, signal[: SECONDS * rate], rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def retrain_samples(manifest: pd.DataFrame, index: dict) -> None:
    train = manifest[manifest.split == "train"].drop_duplicates("recording_id")
    for name in CLASSES:
        folder = RETRAIN_OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        ids = train[train.label == name].recording_id.astype(str).tolist()
        for rec in ids[:: max(1, len(ids) // RETRAIN_PER_CLASS)][:RETRAIN_PER_CLASS]:
            (folder / f"{name}_{rec}.wav").write_bytes(clip(index, rec))
        print(f"retrain samples for {name}: {len(list(folder.glob('*.wav')))}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    test = manifest[manifest.split == "test"].drop_duplicates("recording_id")
    index = member_index(sorted(CACHE.glob("humbugdb_neurips_2021_*.zip")))
    retrain_samples(manifest, index)
    if "--retrain-only" in sys.argv:
        return
    predictor = Predictor()
    predictor.load()

    for name in CLASSES:
        ids = test[test.label == name].recording_id.astype(str).tolist()[:: max(1, len(test[test.label == name]) // CANDIDATES)]
        best = None
        for rec in ids:
            data = clip(index, rec)
            result = predictor.predict(data)
            agree = all(label == name for label in result["window_labels"])
            print(f"{name:14} {rec:>8} -> {result['label']:14} {result['confidence']:.3f} windows {result['windows']} agree={agree}")
            if result["label"] == name and agree and result["windows"] >= 3:
                if best is None or result["confidence"] > best[0]:
                    best = (result["confidence"], rec, data)
        if best is None:
            print(f"!! no correct candidate for {name}")
            continue
        (OUT / f"demo_{name}.wav").write_bytes(best[2])
        print(f"chose {best[1]} for {name} ({best[0]:.3f})")


if __name__ == "__main__":
    main()
