"""
Pre-tokenization script.

Run once per tokenizer AFTER that tokenizer is implemented and verified:

    python pretokenize.py --tokenizer event --workers 4
    python pretokenize.py --tokenizer remi  --workers 4

Reads each .mid from data/raw/, tokenizes it, saves the token ids as a
PyTorch tensor (.pt) to:
    data/tokenized/event/<hash>.pt
    data/tokenized/remi/<hash>.pt

The hash is parsed from the filename (last field before .mid extension).
Files that fail to tokenize are logged and skipped.

--workers controls the number of parallel worker processes.  A value of 4
uses ~50% of an 8-core CPU; 6 uses ~75%.  Defaults to 1 (single-threaded).
"""

import argparse
import logging
import multiprocessing as mp
import os
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_hash(filename: str) -> str:
    """Extract the hash segment from '<song> --- <band> --- <hash>.mid'."""
    stem = Path(filename).stem
    parts = stem.split(" --- ")
    if len(parts) >= 1:
        return parts[-1].strip()
    return stem


def load_tokenizer(name: str):
    if name == "event":
        from tokenizers.event_tokenizer import EventTokenizer
        return EventTokenizer()
    elif name == "remi":
        from tokenizers.remi_wrapper import REMIWrapper
        return REMIWrapper()
    else:
        raise ValueError(f"Unknown tokenizer '{name}'. Choose 'event' or 'remi'.")


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

# Module-level tokenizer instance, initialised once per worker process.
_worker_tok = None
_worker_tok_name: str = ""


def _worker_init(tokenizer_name: str) -> None:
    """Called once in each worker process to build the tokenizer."""
    global _worker_tok, _worker_tok_name
    _worker_tok_name = tokenizer_name
    _worker_tok = load_tokenizer(tokenizer_name)


def _process_one(task: tuple) -> tuple:
    """
    Tokenize a single MIDI file.

    Args:
        task: (fname, out_path_str, overwrite)

    Returns:
        (fname, status, error_or_None)
        where status is 'ok' | 'skip' | 'fail'
    """
    fname, out_path_str, overwrite = task
    out_path = Path(out_path_str)

    if out_path.exists() and not overwrite:
        return fname, "skip", None

    midi_path = RAW_DIR / fname
    try:
        ids = _worker_tok.encode(midi_path)
        if not ids:
            raise ValueError("empty token sequence")
        tensor = torch.tensor(ids, dtype=torch.long)
        torch.save(tensor, out_path)
        return fname, "ok", None
    except Exception as exc:
        return fname, "fail", str(exc)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def pretokenize(tokenizer_name: str, overwrite: bool = False, n_workers: int = 1) -> None:
    out_dir = TOKENIZED_DIR / tokenizer_name
    out_dir.mkdir(parents=True, exist_ok=True)

    mid_files = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".mid"))
    log.info(f"Found {len(mid_files)} .mid files in {RAW_DIR}")
    log.info(f"Tokenizer: {tokenizer_name} | workers: {n_workers} | overwrite: {overwrite}")

    tasks = [
        (fname, str(out_dir / f"{parse_hash(fname)}.pt"), overwrite)
        for fname in mid_files
    ]

    n_ok = 0
    n_skip = 0
    n_fail = 0
    failed: list[str] = []

    if n_workers <= 1:
        # Single-threaded path — avoids multiprocessing overhead
        _worker_init(tokenizer_name)
        results = (_process_one(t) for t in tasks)
    else:
        pool = mp.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(tokenizer_name,),
        )
        results = pool.imap_unordered(_process_one, tasks, chunksize=8)

    try:
        for i, (fname, status, err) in enumerate(results, 1):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                failed.append(fname)
                log.warning(f"FAILED {fname}: {err}")

            if i % 2000 == 0 or i == len(tasks):
                log.info(
                    f"Progress: {i}/{len(tasks)} | ok={n_ok} skip={n_skip} fail={n_fail}"
                )
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    log.info(f"\nDone. tokenized={n_ok}  skipped(existing)={n_skip}  failed={n_fail}")
    if failed:
        log.warning("Failed files:")
        for f in failed:
            log.warning(f"  {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). "
             "4 workers ≈ 50%% CPU on an 8-core machine.",
    )
    args = parser.parse_args()
    pretokenize(args.tokenizer, overwrite=args.overwrite, n_workers=args.workers)


if __name__ == "__main__":
    # Required on Windows so spawned workers don't re-run main()
    mp.freeze_support()
    main()
