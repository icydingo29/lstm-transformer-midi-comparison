"""
Evaluation script.

Computes for a trained model checkpoint:
  - Cross-Entropy loss
  - Perplexity
  - Top-1 and Top-5 accuracy
  - Per-position loss (4 buckets: 0-127, 128-255, 256-383, 384-511)
  - Bits per second of musical time (cross-tokenizer comparison metric)
  - Decode Success Rate (DSR) — sanity check for the custom tokenizer

Usage:
    python evaluate.py --checkpoint checkpoints/lstm_event_best.pt \
                       --model lstm --tokenizer event
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MusicTokenDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POSITION_BUCKETS = [(0, 128), (128, 256), (256, 384), (384, 512)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str | Path, model_name: str, vocab_size: int, pad_id: int, device):
    if model_name == "lstm":
        from models.lstm_model import build_lstm
        model = build_lstm(vocab_size=vocab_size, pad_id=pad_id)
    else:
        from models.transformer_model import build_transformer
        model = build_transformer(vocab_size=vocab_size, pad_id=pad_id)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_tokenizer(name: str):
    if name == "event":
        from tokenizers.event_tokenizer import EventTokenizer
        return EventTokenizer()
    elif name == "remi":
        from tokenizers.remi_wrapper import REMIWrapper
        return REMIWrapper()
    raise ValueError(f"Unknown tokenizer '{name}'")


def get_tokenizer_meta(name: str) -> tuple[int, int]:
    """Return (vocab_size, pad_id)."""
    tok = get_tokenizer(name)
    return tok.vocab_size, tok.pad_id


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    vocab_size, pad_id = get_tokenizer_meta(args.tokenizer)
    tokenizer = get_tokenizer(args.tokenizer)

    model = load_model(args.checkpoint, args.model, vocab_size, pad_id, device)

    splits_dir = Path(args.splits_dir) if args.splits_dir else PROJECT_ROOT / "data" / "splits"
    tokenized_dir = Path(args.tokenized_dir) if args.tokenized_dir else PROJECT_ROOT / "data" / "tokenized" / args.tokenizer

    test_ds = MusicTokenDataset(
        split_json=splits_dir / "test.json",
        tokenized_dir=tokenized_dir,
        context_len=args.context_len,
        pad_id=pad_id,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    log.info(f"Test windows: {len(test_ds)}")

    # Accumulators
    total_loss = 0.0
    total_tokens = 0
    top1_correct = 0
    top5_correct = 0
    # Per-position buckets: [sum_loss, count]
    pos_stats = {bucket: [0.0, 0] for bucket in POSITION_BUCKETS}
    # Bits/second
    total_bits = 0.0
    total_duration_s = 0.0
    # DSR
    n_valid = 0
    n_total = 0

    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.to(device)  # (B, T)
        y = y.to(device)  # (B, T)

        if args.model == "lstm":
            logits, _ = model(x)
        else:
            logits = model(x)
        # logits: (B, T, V)

        B, T, V = logits.shape
        mask = (y != pad_id)  # (B, T) — True for real (non-PAD) tokens

        # --- Per-token NLL (vectorized) ---
        log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
        target_lp = log_probs.gather(2, y.unsqueeze(2)).squeeze(2)  # (B, T)
        nll = -target_lp                                   # (B, T), positive
        nll_masked = nll * mask.float()                    # zero PAD positions

        total_loss   += nll_masked.sum().item()
        total_tokens += mask.sum().item()

        # --- Top-1 / Top-5 (vectorized) ---
        top5_preds = logits.topk(5, dim=-1).indices        # (B, T, 5)
        y_exp = y.unsqueeze(2)                             # (B, T, 1)
        top1_correct += ((top5_preds[:, :, :1] == y_exp).squeeze(2) & mask).sum().item()
        top5_correct += ((top5_preds == y_exp).any(dim=-1) & mask).sum().item()

        # --- Per-position buckets (vectorized) ---
        for bucket in POSITION_BUCKETS:
            lo, hi = bucket
            pos_stats[bucket][0] += nll_masked[:, lo:hi].sum().item()
            pos_stats[bucket][1] += mask[:, lo:hi].sum().item()

        # --- Bits (log2) ---
        total_bits += (nll_masked.sum() / math.log(2)).item()

        # --- Duration: must stay per-sequence (tokenizer is Python) ---
        for b in range(B):
            total_duration_s += tokenizer.duration_seconds(x[b].tolist())

        # --- DSR: validate model's greedy (teacher-forced) predictions ---
        pred_ids_batch = logits.argmax(dim=-1)  # (B, T)
        for b in range(B):
            n_total += 1
            is_valid, _ = tokenizer.validate_decode(pred_ids_batch[b].tolist())
            if is_valid:
                n_valid += 1

        if (batch_idx + 1) % 50 == 0:
            log.info(f"  Batch {batch_idx+1}/{len(test_loader)}")

    # --- Aggregate ---
    ce_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(ce_loss)
    top1_acc = top1_correct / max(total_tokens, 1)
    top5_acc = top5_correct / max(total_tokens, 1)
    bits_per_sec = total_bits / max(total_duration_s, 1e-9)
    dsr = n_valid / max(n_total, 1)

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"Model:     {args.model}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Checkpoint: {args.checkpoint}")
    print("-" * 60)
    print(f"Test tokens (non-PAD): {total_tokens:,}")
    print(f"Cross-Entropy Loss:    {ce_loss:.4f}")
    print(f"Perplexity:            {perplexity:.2f}")
    print(f"Top-1 Accuracy:        {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy:        {top5_acc*100:.2f}%")
    print(f"Bits/second:           {bits_per_sec:.4f}")
    print(f"Decode Success Rate:   {dsr*100:.2f}%  ({n_valid}/{n_total})")
    print("-" * 60)
    print("Per-position loss (token position in context window):")
    for (lo, hi), (loss_sum, count) in sorted(pos_stats.items()):
        if count > 0:
            pos_ce = loss_sum / count
            pos_ppl = math.exp(pos_ce)
            print(f"  [{lo:3d}-{hi:3d}]  CE={pos_ce:.4f}  PPL={pos_ppl:.2f}  ({count:,} tokens)")
        else:
            print(f"  [{lo:3d}-{hi:3d}]  (no tokens)")
    print("=" * 60)

    return {
        "ce_loss": ce_loss,
        "perplexity": perplexity,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "bits_per_sec": bits_per_sec,
        "dsr": dsr,
        "pos_stats": {
            f"{lo}-{hi}": {
                "ce": pos_stats[(lo, hi)][0] / max(pos_stats[(lo, hi)][1], 1),
                "count": pos_stats[(lo, hi)][1],
            }
            for (lo, hi) in POSITION_BUCKETS
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    p.add_argument("--model", choices=["lstm", "transformer"], required=True)
    p.add_argument("--tokenizer", choices=["event", "remi"], required=True)
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tokenized_dir", default=None,
                   help="Path to tokenized .pt files. Defaults to data/tokenized/<tokenizer>/.")
    p.add_argument("--splits_dir", default=None,
                   help="Path to split JSON files. Defaults to data/splits/.")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())