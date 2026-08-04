# LSTM vs Transformer for Symbolic Music Generation

> **Does self-attention's growing context access give the Transformer an increasing advantage over LSTM as sequence position increases?**

A controlled comparison of two neural architectures on next-token prediction over polyphonic MIDI sequences, using two tokenization schemes and matched parameter counts. The core finding: **the Transformer's advantage grows ~8.5× across the context window**, confirming that direct access to all previous tokens matters more as dependencies get longer.
 
**Institution:** Faculty of Mathematics and Informatics, Sofia University "St. Kliment Ohridski"  
**Paper:** [`3MI3400841_9MI3400791_LSTM_versus_Transformer_for_Symbolic_Music_anonymous.pdf`](3MI3400841_9MI3400791_LSTM_versus_Transformer_for_Symbolic_Music_anonymous.pdf)

---

## Experimental Design

2 architectures × 2 tokenizers = **4 trained combinations**, with parameter counts matched within ±10%:

| Label | Architecture | Tokenizer |
|:---:|---|---|
| **A** | LSTM | Event-based |
| **B** | Transformer | Event-based |
| **C** | LSTM | REMI+ |
| **D** | Transformer | REMI+ |

Training order: A → B → C → D, trained on the same data and split. Platforms: Kaggle and Lightning AI.

---

## Architectures

Both models share: **embedding dim 256**, **context length 512 tokens**, **Adam + cosine LR decay**, **Cross-Entropy loss** (PAD ignored).

### LSTM (~4.09M parameters)
```
Embedding(532, 256) → LSTM(256→512, 2 layers, dropout 0.1) → Linear(512, 532)
```

### Decoder-only Transformer (~3.82M parameters)
```
Embedding(532, 256) + sinusoidal PE
→ 4 × [LayerNorm → CausalSelfAttention(4 heads) → LayerNorm → FFN(256→1280→256)]
→ LayerNorm → Linear(256, 532)
```
Uses `F.scaled_dot_product_attention(is_causal=True)` (flash attention). Output projection shares weights with the embedding (weight tying). FFN dim is tuned to 1280 specifically to keep the parameter ratio within ±10% of the LSTM baseline.

---

## Tokenizers

### Event-based (custom, `tokenizers/event_tokenizer.py`)
Chronological stream of MIDI events. Vocab size **532**: NOTE_ON/OFF, DRUM_ON, TIME_SHIFT (100 × 10ms bins), VELOCITY (16 bins), PROGRAM (16 bins), TEMPO (12 bins), BAR, PAD, BOS, EOS. Compact and dense — fewer tokens per second of music.

### REMI+ (via MidiTok, `tokenizers/remi_wrapper.py`)
Hierarchical structure around bars. Vocab size **~502**: Bar, Position, Pitch, Velocity, Duration, Tempo, Program, TimeSig. Encodes explicit metric structure; requires more tokens per second than the event tokenizer.

> Raw Cross-Entropy and Perplexity cannot be compared across tokenizers (different vocab sizes and token densities). **Bits/second of musical time** is the only valid cross-tokenizer metric.

---

## Results

### Bits / second of musical time *(lower = better compression → better model)*

| | Event tokenizer | REMI+ tokenizer |
|---|:---:|:---:|
| **LSTM** | A = 133.67 | C = 94.49 |
| **Transformer** | B = 121.00 | D = 71.30 |
| *Transformer advantage* | *−9.5%* | *−24.5%* |

The Transformer benefits substantially more from REMI+'s explicit metric structure. Both architectures achieve lower bits/sec on REMI+, but the gap between them widens.

### Key finding: advantage grows with context length

Per-position Cross-Entropy loss was measured in four 128-token buckets across the 512-token context window (event tokenizer):

| Position bucket | LSTM–Transformer gap |
|---|:---:|
| [0 – 128) | smallest |
| [128 – 256) | ↑ |
| [256 – 384) | ↑↑ |
| [384 – 512) | **~8.5× larger than at position 0** |

This confirms the hypothesis: LSTM's fixed-size hidden state increasingly loses long-range information, while self-attention retains direct access to all prior tokens.

### Decode Success Rate (DSR)

| | LSTM | Transformer |
|---|:---:|:---:|
| Event | — | — |
| REMI+ | C = 100.00% | D = 99.97% |

---

## Listening Examples

> The output of `generate.py` is a MIDI file. To hear it, open it in any MIDI player or DAW, or render to audio with a soundfont (e.g. [FluidSynth](https://www.fluidsynth.org/) + [GeneralUser GS](http://www.schristiancollins.com/generaluser.php)).

<!--
Once you have MP3 samples, commit them to samples/ and uncomment:

| Prompt | LSTM + Event | Transformer + Event | LSTM + REMI+ | Transformer + REMI+ |
|---|---|---|---|---|
| Aerials | [▶](samples/aerials_A.mp3) | [▶](samples/aerials_B.mp3) | [▶](samples/aerials_C.mp3) | [▶](samples/aerials_D.mp3) |
| Till I Collapse | [▶](samples/tic_A.mp3) | [▶](samples/tic_B.mp3) | [▶](samples/tic_C.mp3) | [▶](samples/tic_D.mp3) |
-->

---

## Project Layout

```
dataset.py          PyTorch Dataset — fixed windows, next-token shift
train.py            Training entry point
evaluate.py         Test-set metrics
generate.py         Autoregressive MIDI generation
models/             LSTM and decoder-only Transformer definitions
tokenizers/         Event-based tokenizer + REMI+ wrapper (MidiTok)
scripts/            split_dataset.py, pretokenize.py
tests/smoke_test.py End-to-end sanity checks (10 stages)
data/splits/        Artist-stratified train/val/test file lists (provided)
```

---

## Quick Start

### 1. Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

pip install -U pip # because of --extra-index-url
pip install -r requirements.txt
```

> `requirements.txt` pins PyTorch 2.6.0 + CUDA 12.4. For CPU-only: remove `+cu124` and the `--extra-index-url` line.

### 2. Download pretrained models

Get any of the 4 checkpoints from HuggingFace into a `checkpoints/` folder:

```
https://huggingface.co/icydingo29/lstm-vs-transformer-symbolic-music
```

The filename tells you which flags to use:
```
transformer_remi_best.pt   →  --model transformer --tokenizer remi
lstm_remi_best.pt          →  --model lstm        --tokenizer remi
transformer_event_best.pt  →  --model transformer --tokenizer event
lstm_event_best.pt         →  --model lstm        --tokenizer event
```

### 3. Generate music

```bash
python generate.py \
  --checkpoint checkpoints/transformer_remi_best.pt \
  --model transformer --tokenizer remi \
  --prompt path/to/song.mid \
  --out generated/output.mid \
  --max_tokens 4096 --temperature 1.0 --top_k 40 --suppress_eos
```

Best results: `--temperature 1.0 --top_k 40`. Greedy and temperature 0.5 collapse into repeated notes.

### 4. Evaluate on the test set

```bash
python evaluate.py \
  --checkpoint checkpoints/transformer_remi_best.pt \
  --model transformer --tokenizer remi
```

Reports CE loss, perplexity, Top-1/Top-5 accuracy, per-position bucket loss, bits/second, and DSR.

### 5. Retrain from scratch *(optional)*

Requires the dataset and pre-tokenization first (see below).

```bash
python train.py --model transformer --tokenizer remi
```

### 6. Data pipeline *(only needed to retrain or re-evaluate)*

```bash
# a) Download dataset into data/raw/
#    https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset

# b) The train/val/test split is already in data/splits/ — to regenerate:
python scripts/split_dataset.py

# c) Pre-tokenize once per tokenizer:
python scripts/pretokenize.py --tokenizer event --workers 4
python scripts/pretokenize.py --tokenizer remi  --workers 4
```

### 7. Sanity check

```bash
python tests/smoke_test.py
```

---

## Dataset & Models

| Resource | Link |
|---|---|
| Pretrained checkpoints (4 × `.pt`) | [HuggingFace — icydingo29/lstm-vs-transformer-symbolic-music](https://huggingface.co/icydingo29/lstm-vs-transformer-symbolic-music) |
| Training data (Lyrics-MIDI) | [HuggingFace — asigalov61/Lyrics-MIDI-Dataset](https://huggingface.co/datasets/asigalov61/Lyrics-MIDI-Dataset) |

Dataset: ~38K MIDI files after artist-stratified 80/10/10 split. Artists are kept together — no artist appears in both train and test — to prevent style memorization from inflating test scores.

---

## Acknowledgements
**[Bifrost19](https://github.com/Bifrost19)** co-author; contributed to model training on
Kaggle and Lightning AI and to the academic paper.
**[asigalov61](https://huggingface.co/asigalov61)** — creator of the Lyrics-MIDI Dataset used for training.

---

## Reproducibility

Seed 42 throughout (data split, training, smoke tests). Rerunning any training command with the same checkpoint produces identical results.
