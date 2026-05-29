"""
Group-based train/val/test split.

Parses artist name from filename format:
    <song_name> --- <band_name> --- <hash>.mid

Sorts artist groups alphabetically, then shuffles with a fixed seed to ensure
reproducibility when the JSON files need to be regenerated. The JSON files
themselves are the source of truth across training platforms.

Output: data/splits/train.json, val.json, test.json
"""

import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"

SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test gets the remainder (0.1)


def parse_artist(filename: str) -> str:
    """Extract artist name from '<song> --- <artist> --- <hash>.mid'."""
    stem = filename.removesuffix(".mid")
    parts = stem.split(" --- ")
    if len(parts) >= 3:
        return parts[-2].strip()
    # Fallback: whole stem as artist so the file still lands somewhere
    return stem.strip()


def main():
    mid_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mid")]
    if not mid_files:
        raise FileNotFoundError(f"No .mid files found in {RAW_DIR}")

    # Group filenames by artist
    artist_to_files: dict[str, list[str]] = defaultdict(list)
    for fname in mid_files:
        artist = parse_artist(fname)
        artist_to_files[artist].append(fname)

    # Sort artist list for deterministic ordering before shuffling
    artists = sorted(artist_to_files.keys())

    rng = random.Random(SEED)
    rng.shuffle(artists)

    n = len(artists)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_artists = artists[:n_train]
    val_artists = artists[n_train : n_train + n_val]
    test_artists = artists[n_train + n_val :]

    train_files = [f for a in train_artists for f in artist_to_files[a]]
    val_files = [f for a in val_artists for f in artist_to_files[a]]
    test_files = [f for a in test_artists for f in artist_to_files[a]]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        out = SPLITS_DIR / f"{name}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(sorted(files), fh, indent=2, ensure_ascii=False)
        print(f"{name}: {len(files)} files ({len(set(a for f in files for a in [parse_artist(f)]))} artists)")

    total = len(train_files) + len(val_files) + len(test_files)
    print(f"\nTotal: {total} files across {n} artists (seed={SEED})")


if __name__ == "__main__":
    main()