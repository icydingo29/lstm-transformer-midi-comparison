"""
PyTorch Dataset for next-token prediction over pre-tokenized MIDI files.

Loads .pt token tensors, slides a window of context_len tokens,
and yields (input, target) pairs where target = input shifted by 1.

Short windows (< context_len) are right-padded with PAD; the loss
must be configured with ignore_index=PAD so padding is not trained on.

Default stride = context_len (non-overlapping windows). Change via stride= arg;
note that stride < context_len increases dataset size and changes the A/B comparison.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset


class MusicTokenDataset(Dataset):
    def __init__(
        self,
        split_json: str | Path,
        tokenized_dir: str | Path,
        context_len: int = 512,
        stride: Optional[int] = None,
        pad_id: int = 529,
    ):
        """
        Args:
            split_json:     Path to train.json / val.json / test.json
                            (list of filenames in data/raw/)
            tokenized_dir:  Directory containing .pt files named by hash
                            (e.g. data/tokenized/event/)
            context_len:    Number of input tokens per window
            stride:         Step between windows (default = context_len → non-overlapping)
            pad_id:         Token id used for padding (must be ignored by the loss)
        """
        self.context_len = context_len
        self.stride = stride if stride is not None else context_len
        self.pad_id = pad_id
        self.tokenized_dir = Path(tokenized_dir)

        with open(split_json, encoding="utf-8") as fh:
            filenames: List[str] = json.load(fh)

        # Build list of (pt_path, tensor_length) for all files in split
        # Windows are indexed lazily to avoid loading everything into RAM
        self._windows: List[tuple[Path, int, int]] = []  # (path, start, seq_len)
        self._missing: List[str] = []

        for fname in filenames:
            hash_id = self._parse_hash(fname)
            pt_path = self.tokenized_dir / f"{hash_id}.pt"
            if not pt_path.exists():
                self._missing.append(fname)
                continue
            n = self._tensor_length(pt_path)
            # Need at least 2 tokens to form one (x, y) pair
            if n < 2:
                continue
            for start in range(0, max(1, n - 1), self.stride):
                self._windows.append((pt_path, start, n))

        if self._missing:
            import logging
            logging.getLogger(__name__).warning(
                f"{len(self._missing)} files missing from {self.tokenized_dir} "
                f"(run pretokenize.py first)"
            )

    @staticmethod
    def _parse_hash(filename: str) -> str:
        stem = Path(filename).stem
        parts = stem.split(" --- ")
        return parts[-1].strip() if len(parts) >= 1 else stem

    @staticmethod
    def _tensor_length(pt_path: Path) -> int:
        """Read tensor length without loading all data (uses storage metadata)."""
        t = torch.load(pt_path, weights_only=True)
        return len(t)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        pt_path, start, seq_len = self._windows[idx]
        tokens: torch.Tensor = torch.load(pt_path, weights_only=True)

        chunk = tokens[start : start + self.context_len + 1]
        x = chunk[:-1]
        y = chunk[1:]

        pad_needed = self.context_len - len(x)
        if pad_needed > 0:
            pad_t = torch.full((pad_needed,), self.pad_id, dtype=torch.long)
            x = torch.cat([x, pad_t])
            y = torch.cat([y, pad_t])

        return x.long(), y.long()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    splits_dir: str | Path,
    tokenized_dir: str | Path,
    context_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    pad_id: int = 529,
):
    """Return (train_loader, val_loader, test_loader)."""
    from torch.utils.data import DataLoader

    splits_dir = Path(splits_dir)

    def _loader(split_name, shuffle):
        ds = MusicTokenDataset(
            split_json=splits_dir / f"{split_name}.json",
            tokenized_dir=tokenized_dir,
            context_len=context_len,
            pad_id=pad_id,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    train = _loader("train", shuffle=True)
    val = _loader("val", shuffle=False)
    test = _loader("test", shuffle=False)
    return train, val, test