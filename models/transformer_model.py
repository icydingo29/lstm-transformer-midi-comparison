"""
Causal (decoder-only) Transformer for next-token prediction.

Architecture:
    Embedding(vocab_size, 256) + sinusoidal positional encoding
    → 4 × [CausalSelfAttention(4 heads) + FFN(ffn_dim)]
    → Linear(256, vocab_size)

FFN dimension is tuned to match the LSTM parameter count (≈4M) within ±10%.
The default ffn_dim=1024 gives ≈3.8M parameters with vocab_size=532 — within
±10% of the LSTM's ≈4M. Adjust ffn_dim in build_transformer() if needed.

Exact parameter counts are printed at construction so they can be compared
and reported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, embed_dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Causal self-attention block
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, T, T)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 1024,
        context_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 529,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.context_len = context_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=context_len + 1, dropout=dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
        # Weight tying: share embedding and head weights
        self.head.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token ids

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        emb = self.embedding(x)          # (B, T, embed_dim)
        h = self.pos_enc(emb)            # (B, T, embed_dim)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)                 # (B, T, embed_dim)
        logits = self.head(h)            # (B, T, vocab_size)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_transformer(vocab_size: int, pad_id: int = 529, ffn_dim: int = 1280) -> TransformerModel:
    """
    Build a Transformer matched (within ±10%) to the LSTM's ~4.09M parameter count.

    Measured counts with vocab_size=532, embed_dim=256, 4 layers, 4 heads:
      ffn_dim=1024 → 3,292,672  (ratio 0.806 — FAILS ±10%)
      ffn_dim=1280 → 3,817,984  (ratio 0.934 — passes ±10%)  ← default
      ffn_dim=1536 → 4,343,296  (ratio 1.063 — passes ±10%)

    Weight tying (head.weight = embedding.weight) means the output projection
    adds no extra parameters. Adjust ffn_dim here if the LSTM param count changes.
    """
    model = TransformerModel(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        ffn_dim=ffn_dim,
        context_len=512,
        dropout=0.1,
        pad_id=pad_id,
    )
    n = model.count_parameters()
    print(f"[TransformerModel] trainable parameters: {n:,}  (ffn_dim={ffn_dim})")
    return model


if __name__ == "__main__":
    # Quick sanity check and parameter count
    for ffn in [512, 1024, 1280, 1536]:
        m = build_transformer(vocab_size=532, ffn_dim=ffn)
    x = torch.randint(0, 532, (2, 512))
    logits = m(x)
    print(f"Output shape: {logits.shape}")   # (2, 512, 532)