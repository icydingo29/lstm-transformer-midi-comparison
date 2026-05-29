"""
LSTM decoder for next-token prediction over music token sequences.

Architecture:
    Embedding(vocab_size, 256) → LSTM(256→512, 2 layers) → Linear(512, vocab_size)

Trainable parameter count is computed exactly and printed at construction;
it is the target that the Transformer must match within ±10%.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 529,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:      (batch, seq_len) token ids
            hidden: optional LSTM hidden state from previous call

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: new LSTM hidden state (h, c)
        """
        emb = self.dropout(self.embedding(x))           # (B, T, embed_dim)
        out, hidden = self.lstm(emb, hidden)            # (B, T, hidden_dim)
        out = self.dropout(out)
        logits = self.head(out)                         # (B, T, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return a zeroed initial hidden state."""
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm(vocab_size: int, pad_id: int = 529) -> LSTMModel:
    model = LSTMModel(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        pad_id=pad_id,
    )
    n = model.count_parameters()
    print(f"[LSTMModel] trainable parameters: {n:,}")
    return model


if __name__ == "__main__":
    m = build_lstm(vocab_size=532)
    x = torch.randint(0, 532, (2, 512))
    logits, _ = m(x)
    print(f"Output shape: {logits.shape}")   # (2, 512, 532)