"""
Shared training script for all four model/tokenizer combinations.

Usage:
    python train.py --model lstm   --tokenizer event
    python train.py --model lstm   --tokenizer remi
    python train.py --model transformer --tokenizer event
    python train.py --model transformer --tokenizer remi

Supports checkpoint save/resume so training can span multiple sessions.
Both model AND optimizer state are saved so the learning rate schedule is
not reset on resume.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MusicTokenDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (override via CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    context_len=512,
    batch_size=32,
    num_epochs=30,
    lr=3e-4,
    warmup_steps=1000,
    grad_clip=1.0,
    num_workers=0,
    seed=42,
    checkpoint_every=1,     # save checkpoint every N epochs
)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, vocab_size: int, pad_id: int):
    if model_name == "lstm":
        from models.lstm_model import build_lstm
        return build_lstm(vocab_size=vocab_size, pad_id=pad_id)
    elif model_name == "transformer":
        from models.transformer_model import build_transformer
        return build_transformer(vocab_size=vocab_size, pad_id=pad_id)
    else:
        raise ValueError(f"Unknown model '{model_name}'")


def get_tokenizer_info(tokenizer_name: str) -> tuple[int, int]:
    """Return (vocab_size, pad_id) for the given tokenizer."""
    if tokenizer_name == "event":
        from tokenizers.event_tokenizer import VOCAB
        return len(VOCAB), VOCAB["PAD"]
    elif tokenizer_name == "remi":
        from tokenizers.remi_wrapper import REMIWrapper
        tok = REMIWrapper()
        return tok.vocab_size, tok.pad_id
    else:
        raise ValueError(f"Unknown tokenizer '{tokenizer_name}'")


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return max(min_lr_ratio, cosine)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    vocab_size, pad_id = get_tokenizer_info(args.tokenizer)
    log.info(f"Tokenizer: {args.tokenizer} | vocab_size={vocab_size} | pad_id={pad_id}")

    # Data
    splits_dir = PROJECT_ROOT / "data" / "splits"
    tokenized_dir = PROJECT_ROOT / "data" / "tokenized" / args.tokenizer

    train_ds = MusicTokenDataset(
        split_json=splits_dir / "train.json",
        tokenized_dir=tokenized_dir,
        context_len=args.context_len,
        pad_id=pad_id,
    )
    val_ds = MusicTokenDataset(
        split_json=splits_dir / "val.json",
        tokenized_dir=tokenized_dir,
        context_len=args.context_len,
        pad_id=pad_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    log.info(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows")

    # Model
    model = build_model(args.model, vocab_size=vocab_size, pad_id=pad_id).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_lambda(step, args.warmup_steps, total_steps),
    )

    # Checkpoint setup
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    run_name = f"{args.model}_{args.tokenizer}"
    ckpt_path = ckpt_dir / f"{run_name}_latest.pt"

    start_epoch = 0
    best_val_loss = float("inf")

    if ckpt_path.exists() and not args.reset:
        log.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Restore scheduler step count
        trained_steps = start_epoch * len(train_loader)
        for _ in range(trained_steps):
            scheduler.step()
        log.info(f"Resumed at epoch {start_epoch} | best_val_loss={best_val_loss:.4f}")

    # Training
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if args.model == "lstm":
                logits, _ = model(x)
            else:
                logits = model(x)

            # logits: (B, T, V)  y: (B, T)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    f"Epoch {epoch+1}/{args.num_epochs} "
                    f"[{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} lr={lr_now:.2e}"
                )

        train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if args.model == "lstm":
                    logits, _ = model(x)
                else:
                    logits = model(x)
                val_loss += criterion(logits.reshape(-1, vocab_size), y.reshape(-1)).item()
        val_loss /= len(val_loader)
        val_ppl = math.exp(val_loss)

        elapsed = time.time() - t0
        log.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | "
            f"time={elapsed:.0f}s"
        )

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }
            torch.save(state, ckpt_path)
            log.info(f"Checkpoint saved → {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / f"{run_name}_best.pt"
            torch.save(state, best_path)
            log.info(f"New best val_loss={best_val_loss:.4f} → {best_path}")

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LSTM or Transformer on music tokens.")
    p.add_argument("--model", choices=["lstm", "transformer"], required=True)
    p.add_argument("--tokenizer", choices=["event", "remi"], required=True)
    p.add_argument("--context_len", type=int, default=DEFAULTS["context_len"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num_epochs", type=int, default=DEFAULTS["num_epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--warmup_steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--grad_clip", type=float, default=DEFAULTS["grad_clip"])
    p.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--checkpoint_every", type=int, default=DEFAULTS["checkpoint_every"])
    p.add_argument("--reset", action="store_true", help="Ignore existing checkpoint and start fresh.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())