"""
Smoke test — run before pretokenizing or training.

Covers every step that can catch bugs cheaply before they corrupt
hours of GPU time:

  1. split_dataset.py   — JSON files, no artist leakage
  2. event_tokenizer    — encode 5 files, validate token ids
  3. encode → decode    — round-trip; decoded MIDI is non-empty
  4. validate_decode    — no structural errors
  5. duration_seconds   — returns a positive float
  6. pretokenize (5 files) — .pt files written correctly
  7. MusicTokenDataset  — correct shape, no PAD leaking into x
  8. LSTM forward pass  — logits shape, finite loss near log(V)
  9. Transformer forward — same checks
 10. One backward pass  — gradients flow, no NaN

Usage:
    python tests/smoke_test.py
    python tests/smoke_test.py --midi path/to/file.mid   # specific file
    python tests/smoke_test.py --n 10                    # more files
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIR  = ROOT / "data" / "raw"
SPLITS_DIR = ROOT / "data" / "splits"

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
SKIP = "\033[33m SKIP\033[0m"

_failures: list[str] = []


def section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def ok(label: str, detail: str = ""):
    msg = f"[{PASS}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def fail(label: str, detail: str = ""):
    msg = f"[{FAIL}] {label}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)
    _failures.append(label)


def skip(label: str, reason: str = ""):
    print(f"[{SKIP}] {label}  — {reason}")


def pick_midi_files(n: int = 5, specific: str | None = None) -> list[Path]:
    if specific:
        p = Path(specific)
        return [p] if p.exists() else []
    files = list(RAW_DIR.glob("*.mid"))
    if not files:
        return []
    rng = random.Random(42)
    return rng.sample(files, min(n, len(files)))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — split_dataset
# ─────────────────────────────────────────────────────────────────────────────

def test_split(run_split: bool):
    section("Step 1 — split_dataset.py")

    if run_split:
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "split_dataset.py")],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            fail("split_dataset ran without error", r.stderr[-500:])
            return
        ok("split_dataset.py exited 0")
    else:
        skip("split_dataset.py", "skipped (pass --run-split to execute)")

    # Validate JSON files if they exist
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}.json"
        if not p.exists():
            skip(f"{split}.json exists", "file not found — run split_dataset.py first")
            continue
        with open(p, encoding="utf-8") as fh:
            files = json.load(fh)
        if not isinstance(files, list) or len(files) == 0:
            fail(f"{split}.json is a non-empty list")
            continue
        ok(f"{split}.json", f"{len(files)} files")

    # No artist leakage between splits
    if all((SPLITS_DIR / f"{s}.json").exists() for s in ("train", "val", "test")):
        def parse_artist(fname):
            parts = Path(fname).stem.split(" --- ")
            return parts[-2].strip() if len(parts) >= 3 else fname

        train_artists, val_artists, test_artists = set(), set(), set()
        for split, dest in [("train", train_artists), ("val", val_artists), ("test", test_artists)]:
            with open(SPLITS_DIR / f"{split}.json", encoding="utf-8") as fh:
                for f in json.load(fh):
                    dest.add(parse_artist(f))

        overlap_tv = train_artists & val_artists
        overlap_tt = train_artists & test_artists
        overlap_vt = val_artists & test_artists
        if overlap_tv or overlap_tt or overlap_vt:
            fail("no artist leakage between splits",
                 f"overlaps: train∩val={len(overlap_tv)}, train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}")
        else:
            ok("no artist leakage between splits")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2–5 — event_tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def test_event_tokenizer(midi_files: list[Path]):
    section("Steps 2–5 — event_tokenizer (encode / decode / validate)")

    try:
        from tokenizers.event_tokenizer import EventTokenizer, VOCAB
        tok = EventTokenizer()
        ok("EventTokenizer imported")
    except Exception as e:
        fail("EventTokenizer import", str(e))
        return

    # Vocab sanity
    if len(VOCAB) == 532:
        ok("vocab size == 532")
    else:
        fail("vocab size", f"expected 532, got {len(VOCAB)}")

    if VOCAB.get("PAD") == 529:
        ok("PAD id == 529")
    else:
        fail("PAD id", f"got {VOCAB.get('PAD')}")

    if not midi_files:
        skip("encode/decode tests", f"no .mid files found in {RAW_DIR}")
        return

    n_ok = 0
    for midi_path in midi_files:
        label = midi_path.name[:50]
        try:
            ids = tok.encode(midi_path)

            # Token ids in range
            bad = [x for x in ids if x < 0 or x > 531]
            if bad:
                fail(f"encode range [{label}]", f"{len(bad)} out-of-range ids: {bad[:5]}")
                continue

            # BOS at start, EOS somewhere
            if ids[0] != VOCAB["BOS"]:
                fail(f"BOS at start [{label}]", f"first token is {ids[0]}")
                continue
            if VOCAB["EOS"] not in ids:
                fail(f"EOS present [{label}]")
                continue

            # Duration
            dur = tok.duration_seconds(ids)
            if dur <= 0:
                fail(f"duration_seconds > 0 [{label}]", f"got {dur}")
                continue

            # Decode → non-empty MIDI
            mid = tok.decode(ids)
            notes = sum(
                1 for msg in mid.tracks[0] if hasattr(msg, "type") and msg.type == "note_on"
            )
            if notes == 0:
                fail(f"decoded MIDI has notes [{label}]", "0 note_on messages")
                continue

            # validate_decode
            valid, reason = tok.validate_decode(ids)
            if not valid:
                fail(f"validate_decode [{label}]", reason)
                continue

            ok(f"encode→decode→validate [{label}]",
               f"{len(ids)} tokens, {dur:.1f}s, {notes} note_on events")
            n_ok += 1

        except Exception:
            fail(f"encode/decode [{label}]", traceback.format_exc(limit=3))

    if n_ok == len(midi_files):
        ok(f"all {n_ok} files passed encode/decode")
    else:
        fail(f"encode/decode success rate", f"{n_ok}/{len(midi_files)}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — pretokenize (event, 5 files only)
# ─────────────────────────────────────────────────────────────────────────────

def test_pretokenize(midi_files: list[Path]):
    section("Step 6 — pretokenize (event tokenizer, sample files)")

    if not midi_files:
        skip("pretokenize", "no .mid files")
        return

    try:
        import torch
        from tokenizers.event_tokenizer import EventTokenizer
    except ImportError as e:
        fail("imports for pretokenize", str(e))
        return

    tok = EventTokenizer()
    tmp = Path(tempfile.mkdtemp(prefix="smoke_pt_"))

    try:
        for midi_path in midi_files:
            stem = Path(midi_path).stem.split(" --- ")
            hash_id = stem[-1].strip() if len(stem) >= 1 else Path(midi_path).stem
            out_path = tmp / f"{hash_id}.pt"
            label = midi_path.name[:50]

            try:
                ids = tok.encode(midi_path)
                tensor = torch.tensor(ids, dtype=torch.long)
                torch.save(tensor, out_path)

                loaded = torch.load(out_path, weights_only=True)
                if loaded.dtype != torch.long:
                    fail(f"pt dtype is long [{label}]", str(loaded.dtype))
                elif list(loaded.shape) != [len(ids)]:
                    fail(f"pt shape [{label}]", f"{loaded.shape} vs expected ({len(ids)},)")
                else:
                    ok(f"pretokenize [{label}]", f"shape {tuple(loaded.shape)}")
            except Exception:
                fail(f"pretokenize [{label}]", traceback.format_exc(limit=3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — MusicTokenDataset
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset():
    section("Step 7 — MusicTokenDataset")

    train_json = SPLITS_DIR / "train.json"
    event_dir  = ROOT / "data" / "tokenized" / "event"

    if not train_json.exists():
        skip("MusicTokenDataset", "train.json not found — run split_dataset.py first")
        return
    if not event_dir.exists() or not list(event_dir.glob("*.pt")):
        skip("MusicTokenDataset", "no .pt files in data/tokenized/event — run pretokenize.py first")
        return

    try:
        import torch
        from dataset import MusicTokenDataset

        ds = MusicTokenDataset(
            split_json=train_json,
            tokenized_dir=event_dir,
            context_len=512,
        )

        if len(ds) == 0:
            fail("dataset has windows", "0 windows found")
            return
        ok("dataset length", f"{len(ds)} windows")

        x, y = ds[0]
        if tuple(x.shape) != (512,):
            fail("x shape", f"expected (512,), got {tuple(x.shape)}")
        elif tuple(y.shape) != (512,):
            fail("y shape", f"expected (512,), got {tuple(y.shape)}")
        else:
            ok("window shapes", f"x={tuple(x.shape)}, y={tuple(y.shape)}")

        # y is x shifted by 1: check on non-padded suffix
        pad_id = 529
        non_pad = (x != pad_id).nonzero(as_tuple=True)[0]
        if len(non_pad) > 1:
            last = non_pad[-1].item()
            if y[last - 1].item() != x[last].item():
                fail("y == x shifted by 1")
            else:
                ok("y == x shifted by 1")

        # PAD in y only at positions where x is also PAD
        x_pad = (x == pad_id)
        y_pad = (y == pad_id)
        if (y_pad & ~x_pad).any():
            fail("PAD only at padded positions", "y has PAD where x does not")
        else:
            ok("PAD positions consistent between x and y")

    except Exception:
        fail("MusicTokenDataset", traceback.format_exc(limit=5))


# ─────────────────────────────────────────────────────────────────────────────
# Steps 8–10 — model forward + backward
# ─────────────────────────────────────────────────────────────────────────────

def test_models():
    section("Steps 8–10 — model forward pass + one backward step")

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        fail("torch import")
        return

    VOCAB_SIZE = 532
    PAD_ID = 529
    B, T = 2, 64   # small batch/seq for speed

    x = torch.randint(0, VOCAB_SIZE - 3, (B, T))  # avoid PAD/BOS/EOS
    y = torch.randint(0, VOCAB_SIZE - 3, (B, T))
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    expected_init_loss = math.log(VOCAB_SIZE)  # ≈ 6.27

    # ── LSTM ─────────────────────────────────────────────────────────────────
    try:
        from models.lstm_model import build_lstm
        lstm = build_lstm(vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
        lstm.train()

        logits, hidden = lstm(x)

        if tuple(logits.shape) != (B, T, VOCAB_SIZE):
            fail("LSTM logits shape", f"got {tuple(logits.shape)}")
        else:
            ok("LSTM logits shape", str(tuple(logits.shape)))

        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not math.isfinite(loss.item()):
            fail("LSTM loss is finite", str(loss.item()))
        else:
            ok("LSTM loss is finite", f"{loss.item():.3f}  (expected ~{expected_init_loss:.2f} at init)")

        loss.backward()
        has_grad = all(
            p.grad is not None for p in lstm.parameters() if p.requires_grad
        )
        nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in lstm.parameters() if p.requires_grad
        )
        if not has_grad:
            fail("LSTM gradients exist")
        elif nan_grad:
            fail("LSTM no NaN gradients")
        else:
            ok("LSTM backward pass (grads exist, no NaN)")

        n = lstm.count_parameters()
        ok("LSTM param count", f"{n:,}")

    except Exception:
        fail("LSTM tests", traceback.format_exc(limit=5))

    # ── Transformer ───────────────────────────────────────────────────────────
    try:
        from models.transformer_model import build_transformer
        tf = build_transformer(vocab_size=VOCAB_SIZE, pad_id=PAD_ID)
        tf.train()

        logits = tf(x)

        if tuple(logits.shape) != (B, T, VOCAB_SIZE):
            fail("Transformer logits shape", f"got {tuple(logits.shape)}")
        else:
            ok("Transformer logits shape", str(tuple(logits.shape)))

        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        if not math.isfinite(loss.item()):
            fail("Transformer loss is finite", str(loss.item()))
        else:
            ok("Transformer loss is finite", f"{loss.item():.3f}  (expected ~{expected_init_loss:.2f} at init)")

        loss.backward()
        has_grad = all(
            p.grad is not None for p in tf.parameters() if p.requires_grad
        )
        nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in tf.parameters() if p.requires_grad
        )
        if not has_grad:
            fail("Transformer gradients exist")
        elif nan_grad:
            fail("Transformer no NaN gradients")
        else:
            ok("Transformer backward pass (grads exist, no NaN)")

        n = tf.count_parameters()
        ok("Transformer param count", f"{n:,}")

        # ±10% rule
        from models.lstm_model import build_lstm as _bl
        n_lstm = _bl(vocab_size=VOCAB_SIZE, pad_id=PAD_ID).count_parameters()
        ratio = n / n_lstm
        if abs(ratio - 1) > 0.10:
            fail("Transformer within ±10% of LSTM",
                 f"ratio={ratio:.3f}  (LSTM={n_lstm:,}, Transformer={n:,})")
        else:
            ok("Transformer within ±10% of LSTM",
               f"ratio={ratio:.3f}  (LSTM={n_lstm:,}, Transformer={n:,})")

    except Exception:
        fail("Transformer tests", traceback.format_exc(limit=5))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smoke test for the music modeling project.")
    parser.add_argument("--midi", default=None, help="Path to a specific .mid file to test with.")
    parser.add_argument("--n", type=int, default=5, help="Number of random .mid files to test (default 5).")
    parser.add_argument("--run-split", action="store_true", help="Actually execute split_dataset.py.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SMOKE TEST - LSTM vs Transformer music modeling project")
    print("=" * 60)

    midi_files = pick_midi_files(n=args.n, specific=args.midi)
    print(f"\nUsing {len(midi_files)} MIDI file(s) for encode/decode tests.")

    test_split(run_split=args.run_split)
    test_event_tokenizer(midi_files)
    test_pretokenize(midi_files)
    test_dataset()
    test_models()

    print("\n" + "=" * 60)
    if _failures:
        print(f"\033[31m  {len(_failures)} FAILURE(S):\033[0m")
        for f in _failures:
            print(f"    • {f}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\033[32m  ALL CHECKS PASSED\033[0m")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
