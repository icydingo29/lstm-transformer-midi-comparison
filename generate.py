"""
Autoregressive generation script — produces MIDI files for listening examples.

Usage:
    python generate.py --checkpoint checkpoints/lstm_event_best.pt \
                       --model lstm --tokenizer event \
                       --prompt data/raw/<some_file>.mid \
                       --out generated/output.mid \
                       --max_tokens 512 \
                       --temperature 1.0 \
                       --top_k 40
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


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
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """
    Sample the next token from logits with temperature scaling and top-k filtering.

    Args:
        logits:      (vocab_size,) raw logits for the next position
        temperature: >1 → more random, <1 → more peaked, =1 → as-is
        top_k:       keep only the top-k tokens before sampling (0 = all)

    Returns:
        sampled token id
    """
    if temperature == 0.0:
        return int(logits.argmax().item())

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    tokenizer = get_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id

    model = load_model(args.checkpoint, args.model, vocab_size, pad_id, device)

    # Build prompt token sequence
    if args.prompt:
        log.info(f"Encoding prompt: {args.prompt}")
        prompt_ids = tokenizer.encode(args.prompt)
        # Use at most context_len // 2 tokens as prompt (leave room for generation)
        max_prompt = args.context_len // 2
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[:max_prompt]
    else:
        # Start from BOS only
        prompt_ids = [bos_id]

    log.info(f"Prompt length: {len(prompt_ids)} tokens")

    generated = list(prompt_ids)

    # LSTM needs hidden state carried between steps
    hidden = None
    if args.model == "lstm":
        # Warm up hidden state on the prompt
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = model(prompt_tensor)

    for step in range(args.max_tokens):
        # For Transformer, feed the last context_len tokens
        if args.model == "transformer":
            ctx = generated[-args.context_len:]
            x = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(x)  # (1, T, V)
            next_logits = logits[0, -1, :]  # (V,)
        else:
            # LSTM: feed only the last token, carry hidden state
            last_token = torch.tensor([[generated[-1]]], dtype=torch.long).to(device)
            logits, hidden = model(last_token, hidden)
            next_logits = logits[0, -1, :]  # (V,)

        # Mask PAD so we never generate it
        next_logits[pad_id] = float("-inf")

        next_id = sample_next_token(next_logits, args.temperature, args.top_k)
        generated.append(next_id)

        if next_id == eos_id:
            log.info(f"EOS reached at step {step+1}")
            break

    log.info(f"Generated {len(generated)} tokens total")

    # Decode to MIDI
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.tokenizer == "event":
        midi = tokenizer.decode(generated)
        midi.save(str(out_path))
    else:
        score = tokenizer.decode(generated)
        score.dump_midi(str(out_path))

    log.info(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate music with a trained model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["lstm", "transformer"], required=True)
    p.add_argument("--tokenizer", choices=["event", "remi"], required=True)
    p.add_argument("--prompt", default=None, help="Path to a .mid file used as prompt.")
    p.add_argument("--out", default="generated/output.mid")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=40, help="0 = no top-k filtering.")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())