"""Build data/train, data/validation and data/test from the HumBugDB download.

Only recordings made the same way are used (Tanzania, cup recordings, Tascam
recorder, one mosquito at a time) so the model has to learn the species and
not the device. Only female mosquitoes are kept, because only females bite.

Recordings are split by recording date, so all clips from one night end up in
the same split and the test set holds nights the model never heard.

Missing files are downloaded from Zenodo first (about 4 GB, record 4904800).

Usage: python scripts/prepare_data.py [download folder, default .cache/humbugdb]
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import SAMPLE_RATE, WINDOW, AudioError, load_audio  # noqa: E402

SEED = 42
SPLITS = {"train": 0.70, "validation": 0.15, "test": 0.15}
TARGET = {"train": 900, "validation": 200, "test": 200}
MAX_WINDOWS = {"background": 20}
DEFAULT_MAX_WINDOWS = 6
SPECIES = {
    "an arabiensis": "an_arabiensis",
    "an funestus ss": "an_funestus",
    "culex pipiens complex": "culex_pipiens",
}


def select_recordings(meta: pd.DataFrame) -> pd.DataFrame:
    same_setup = (meta.country == "Tanzania") & (meta.location_type == "cup") & (meta.device_type == "tascam")
    mosquito = same_setup & (meta.sound_type == "mosquito") & meta.species.isin(SPECIES) & (meta.gender == "Female")
    background = same_setup & (meta.sound_type == "background")
    chosen = meta[mosquito | background].copy()
    chosen["label"] = chosen.species.map(SPECIES).fillna("background")
    chosen["session"] = chosen.record_datetime.str.slice(0, 8)
    return chosen


def assign_splits(chosen: pd.DataFrame) -> pd.DataFrame:
    """Give whole sessions to splits, filling each split towards its share of audio."""
    rng = np.random.default_rng(SEED)
    chosen = chosen.copy()
    chosen["split"] = ""
    for label, group in chosen.groupby("label"):
        seconds = group.groupby("session").length.sum()
        sessions = list(seconds.index)
        rng.shuffle(sessions)
        total = seconds.sum()
        filled = dict.fromkeys(SPLITS, 0.0)
        for session in sessions:
            split = max(SPLITS, key=lambda name: SPLITS[name] - filled[name] / total)
            filled[split] += seconds[session]
            chosen.loc[(chosen.label == label) & (chosen.session == session), "split"] = split
    return chosen


def member_index(zip_paths) -> dict:
    index = {}
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".wav"):
                    index[Path(name).stem] = (zip_path, name)
    return index


def evenly_spaced(count: int, limit: int) -> list:
    if count <= limit:
        return list(range(count))
    return sorted({int(round(i)) for i in np.linspace(0, count - 1, limit)})


ZENODO = "https://zenodo.org/records/4904800/files/{}?download=1"
FILES = {"meta.csv": "neurips_2021_zenodo_0_0_1.csv",
         **{f"humbugdb_neurips_2021_{i}.zip": f"humbugdb_neurips_2021_{i}.zip" for i in range(1, 5)}}


def download(download_dir: Path) -> None:
    """Fetch the label file and the four audio zips unless they are already there."""
    download_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        target = download_dir / local
        if target.exists():
            continue
        print(f"downloading {remote} ...", flush=True)
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(ZENODO.format(remote), partial)
        partial.rename(target)


def main(download_dir: Path) -> None:
    download(download_dir)
    meta = pd.read_csv(download_dir / "meta.csv")
    chosen = assign_splits(select_recordings(meta))
    index = member_index(sorted(download_dir.glob("humbugdb_neurips_2021_*.zip")))

    clips = defaultdict(list)
    missing, unreadable = 0, 0
    open_archives = {}
    for row in chosen.itertuples():
        found = index.get(str(row.id))
        if found is None:
            missing += 1
            continue
        zip_path, member = found
        archive = open_archives.setdefault(zip_path, zipfile.ZipFile(zip_path))
        try:
            signal = load_audio(archive.read(member))
        except AudioError:
            unreadable += 1
            continue
        windows = signal.size // WINDOW
        limit = MAX_WINDOWS.get(row.label, DEFAULT_MAX_WINDOWS)
        for k in evenly_spaced(windows, limit):
            clips[(row.split, row.label)].append((row, k, signal[k * WINDOW : (k + 1) * WINDOW]))

    rng = np.random.default_rng(SEED)
    records = []
    for (split, label), items in sorted(clips.items(), key=lambda item: item[0]):
        keep = rng.permutation(len(items))[: TARGET[split]]
        folder = ROOT / "data" / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in sorted(keep):
            row, k, window = items[i]
            name = f"{row.id}_{k:02d}.wav"
            sf.write(folder / name, window, SAMPLE_RATE, subtype="PCM_16")
            records.append({
                "file": f"{split}/{label}/{name}", "split": split, "label": label, "recording_id": row.id,
                "session": row.session, "window": k, "species": row.species if isinstance(row.species, str) else "",
            })

    manifest = pd.DataFrame(records)
    manifest.to_csv(ROOT / "data" / "manifest.csv", index=False)
    print(f"recordings selected: {len(chosen)}  missing from zips: {missing}  unreadable: {unreadable}")
    print(manifest.pivot_table(index="label", columns="split", values="file", aggfunc="count")[list(SPLITS)])
    overlap = manifest.groupby(["label", "session"]).split.nunique().max()
    print(f"sessions shared between splits: {'none' if overlap == 1 else 'FOUND'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache" / "humbugdb")
