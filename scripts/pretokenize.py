"""
Pre-tokenization script.

Run once per tokenizer AFTER that tokenizer is implemented and verified:

    python pretokenize.py --tokenizer event
    python pretokenize.py --tokenizer remi

Reads each .mid from data/raw/, tokenizes it, saves the token ids as a
PyTorch tensor (.pt) to:
    data/tokenized/event/<hash>.pt
    data/tokenized/remi/<hash>.pt

The hash is parsed from the filename (last field before .mid extension).
Files that fail to tokenize are logged and skipped.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import torch

# Allow imports from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_hash(filename: str) -> str:
    """Extract the hash segment from '<song> --- <band> --- <hash>.mid'."""
    stem = Path(filename).stem
    parts = stem.split(" --- ")
    if len(parts) >= 1:
        return parts[-1].strip()
    return stem  # fallback: whole stem


def load_tokenizer(name: str):
    if name == "event":
        from tokenizers.event_tokenizer import EventTokenizer
        return EventTokenizer()
    elif name == "remi":
        from tokenizers.remi_wrapper import REMIWrapper
        return REMIWrapper()
    else:
        raise ValueError(f"Unknown tokenizer '{name}'. Choose 'event' or 'remi'.")


def pretokenize(tokenizer_name: str, overwrite: bool = False):
    out_dir = TOKENIZED_DIR / tokenizer_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(tokenizer_name)
    log.info(f"Loaded tokenizer '{tokenizer_name}' (vocab_size={tok.vocab_size})")

    mid_files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mid"))
    log.info(f"Found {len(mid_files)} .mid files in {RAW_DIR}")

    n_ok = 0
    n_skip = 0
    n_fail = 0
    failed: list[str] = []

    for fname in mid_files:
        hash_id = parse_hash(fname)
        out_path = out_dir / f"{hash_id}.pt"

        if out_path.exists() and not overwrite:
            n_skip += 1
            continue

        midi_path = RAW_DIR / fname
        try:
            ids = tok.encode(midi_path)
            if not ids:
                raise ValueError("empty token sequence")
            tensor = torch.tensor(ids, dtype=torch.long)
            torch.save(tensor, out_path)
            n_ok += 1
        except Exception as exc:
            log.warning(f"FAILED {fname}: {exc}")
            failed.append(fname)
            n_fail += 1

    log.info(
        f"\nDone. tokenized={n_ok}  skipped(existing)={n_skip}  failed={n_fail}"
    )
    if failed:
        log.warning("Failed files:")
        for f in failed:
            log.warning(f"  {f}")


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize MIDI files.")
    parser.add_argument(
        "--tokenizer",
        choices=["event", "remi"],
        required=True,
        help="Which tokenizer to use.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-tokenize files that already have a .pt output.",
    )
    args = parser.parse_args()
    pretokenize(args.tokenizer, overwrite=args.overwrite)


if __name__ == "__main__":
    main()