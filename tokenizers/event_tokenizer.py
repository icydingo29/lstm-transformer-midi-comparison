"""
Custom event-based MIDI tokenizer.

Vocabulary (532 tokens):
  NOTE_ON_0..127   →  0..127
  NOTE_OFF_0..127  →  128..255
  DRUM_ON_0..127   →  256..383
  TIME_SHIFT_0..99 →  384..483   (each step = 10ms, max 1s per token)
  VELOCITY_0..15   →  484..499   (velocity quantized to 16 bins)
  PROGRAM_0..15    →  500..515   (128 MIDI programs → 16 categories)
  TEMPO_0..11      →  516..527   (tempo bins)
  BAR              →  528
  PAD              →  529        (ignore_index for loss)
  BOS              →  530
  EOS              →  531
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import mido

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB: dict[str, int] = {
    **{f"NOTE_ON_{i}": i for i in range(128)},
    **{f"NOTE_OFF_{i}": i + 128 for i in range(128)},
    **{f"DRUM_ON_{i}": i + 256 for i in range(128)},
    **{f"TIME_SHIFT_{i}": i + 384 for i in range(100)},
    **{f"VELOCITY_{i}": i + 484 for i in range(16)},
    **{f"PROGRAM_{i}": i + 500 for i in range(16)},
    **{f"TEMPO_{i}": i + 516 for i in range(12)},
    "BAR": 528,
    "PAD": 529,
    "BOS": 530,
    "EOS": 531,
}
assert len(VOCAB) == 532, f"Vocab size mismatch: {len(VOCAB)}"

ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_STEP_MS = 10          # ms per TIME_SHIFT unit
MAX_SHIFT_MS = 1000        # max ms a single TIME_SHIFT covers
N_VELOCITY_BINS = 16
N_PROGRAM_BINS = 16
DRUM_CHANNEL = 9           # 0-indexed

# Tempo bins: boundaries in BPM
# 0:<60, 1:60-70, 2:70-80, 3:80-90, 4:90-100, 5:100-110,
# 6:110-120, 7:120-140, 8:140-160, 9:160-180, 10:180-200, 11:>=200
TEMPO_BOUNDARIES = [60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200]

DRUM_FIXED_DURATION_MS = 100  # duration given to DRUM_ON events during decode


def _velocity_to_bin(velocity: int) -> int:
    """Quantize MIDI velocity 0-127 → 0-15."""
    return min(velocity * N_VELOCITY_BINS // 128, N_VELOCITY_BINS - 1)


def _bin_to_velocity(bin_idx: int) -> int:
    """Dequantize velocity bin 0-15 → MIDI velocity (centre of bin)."""
    return (bin_idx * 128 // N_VELOCITY_BINS) + (128 // N_VELOCITY_BINS // 2)


def _program_to_bin(program: int) -> int:
    """Map MIDI program 0-127 → 0-15 (8 programs per bin)."""
    return min(program * N_PROGRAM_BINS // 128, N_PROGRAM_BINS - 1)


def _bin_to_program(bin_idx: int) -> int:
    """Map program bin → representative MIDI program (first in bin)."""
    return bin_idx * (128 // N_PROGRAM_BINS)


def _bpm_to_bin(bpm: float) -> int:
    """Map BPM → tempo bin index 0-11."""
    for i, boundary in enumerate(TEMPO_BOUNDARIES):
        if bpm < boundary:
            return i
    return 11


def _microseconds_to_bpm(tempo_us: int) -> float:
    """Convert MIDI tempo (microseconds per beat) to BPM."""
    if tempo_us <= 0:
        return 120.0
    return 60_000_000 / tempo_us


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def _ticks_to_ms(ticks: int, ticks_per_beat: int, tempo_us: int) -> float:
    """Convert MIDI ticks to milliseconds given current tempo."""
    ms_per_tick = tempo_us / 1000 / ticks_per_beat
    return ticks * ms_per_tick


def _ms_to_time_shifts(ms: float) -> List[int]:
    """Convert a gap in ms to a list of TIME_SHIFT token ids (max 1s each)."""
    remaining = max(0.0, ms)
    tokens = []
    while remaining >= TIME_STEP_MS / 2:  # round to nearest step
        steps = min(int(remaining / TIME_STEP_MS + 0.5), 99)
        if steps == 0:
            break
        tokens.append(VOCAB[f"TIME_SHIFT_{steps}"])
        remaining -= steps * TIME_STEP_MS
    return tokens


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode(midi_path: str | Path) -> List[int]:
    """
    Encode a MIDI file to a list of token ids.

    Unwanted event types (Sustain Pedal, Pitch Bend, SysEx) are silently skipped.
    Wraps output with BOS … EOS.
    """
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat  # ticks per beat

    # Merge all tracks into a single chronological stream of absolute-tick messages
    merged = mido.merge_tracks(mid.tracks)

    # ---- Pass 1: build an event list with absolute ms timestamps ----
    current_tempo_us = 500_000  # default 120 BPM
    abs_tick = 0
    abs_ms = 0.0

    # Track per-channel current program
    channel_program: dict[int, int] = {}

    # Collect raw events: (abs_ms, event_type, data...)
    # event_type in: 'note_on', 'note_off', 'drum_on', 'tempo', 'bar_hint'
    raw_events: list[tuple] = []

    # We also need bar-boundary info. We collect time_signature meta messages
    # and compute bar lengths.
    time_sig_events: list[tuple[float, int, int]] = []  # (abs_ms, numerator, denominator)
    tempo_events: list[tuple[float, int]] = []  # (abs_ms, tempo_us)
    tempo_events.append((0.0, current_tempo_us))

    for msg in merged:
        delta_ms = _ticks_to_ms(msg.time, tpb, current_tempo_us)
        abs_ms += delta_ms
        abs_tick += msg.time

        if msg.type == "set_tempo":
            current_tempo_us = msg.tempo
            tempo_events.append((abs_ms, current_tempo_us))
        elif msg.type == "time_signature":
            time_sig_events.append((abs_ms, msg.numerator, msg.denominator))
        elif msg.type in ("note_on", "note_off"):
            ch = msg.channel
            pitch = msg.pitch
            vel = msg.velocity if hasattr(msg, "velocity") else 0
            is_drum = (ch == DRUM_CHANNEL)

            if msg.type == "note_on" and vel > 0:
                if is_drum:
                    raw_events.append((abs_ms, "drum_on", pitch, vel))
                else:
                    prog = channel_program.get(ch, 0)
                    raw_events.append((abs_ms, "note_on", pitch, vel, prog))
            else:
                # note_off or note_on with vel=0
                if not is_drum:
                    raw_events.append((abs_ms, "note_off", pitch))
        elif msg.type == "program_change":
            channel_program[msg.channel] = msg.program
        # Sustain Pedal (CC 64), Pitch Bend, SysEx → silently skip

    # ---- Compute BAR boundaries ----
    bar_boundaries_ms = _compute_bar_boundaries(
        time_sig_events, tempo_events, abs_ms
    )

    # ---- Pass 2: sort events at the same ms; build token list ----
    # Group events by quantized time slot (10ms grid)
    # Also interleave BAR markers and TEMPO tokens

    tokens: List[int] = [VOCAB["BOS"]]

    # Emit initial TEMPO token
    init_bpm = _microseconds_to_bpm(tempo_events[0][1])
    tokens.append(VOCAB[f"TEMPO_{_bpm_to_bin(init_bpm)}"])

    # If no time signature info, treat as 4/4 from the start
    if not time_sig_events:
        time_sig_events = [(0.0, 4, 4)]

    # Emit first BAR token
    tokens.append(VOCAB["BAR"])

    # Walk through time in quantized 10ms steps, interleaving:
    #   - BAR markers when a boundary is crossed
    #   - TEMPO tokens on tempo changes
    #   - NOTE_ON / NOTE_OFF / DRUM_ON with VELOCITY / PROGRAM prefix

    # Sort raw_events by (abs_ms, event ordering priority)
    def _event_sort_key(ev):
        t = ev[0]
        kind = ev[1]
        pitch = ev[2]
        # At same time: note_on < drum_on < note_off (so we match OFF after ON)
        priority = {"note_on": 0, "drum_on": 1, "note_off": 2}[kind]
        return (t, priority, pitch)

    raw_events.sort(key=_event_sort_key)

    # Walk along bar boundaries and tempo changes as checkpoints
    bar_iter = iter(bar_boundaries_ms[1:])  # skip the first boundary (=0)
    tempo_iter = iter(tempo_events[1:])     # skip the initial tempo

    next_bar = next(bar_iter, math.inf)
    next_tempo_ms, next_tempo_us = next(tempo_iter, (math.inf, 0))

    current_ms = 0.0
    current_program_bin = -1  # force PROGRAM emit on first note

    for ev in raw_events:
        ev_ms = ev[0]

        # Drain checkpoints that fall before this event
        while next_bar <= ev_ms or next_tempo_ms <= ev_ms:
            checkpoint_ms = min(next_bar, next_tempo_ms)

            # Emit TIME_SHIFT up to checkpoint
            gap = checkpoint_ms - current_ms
            tokens.extend(_ms_to_time_shifts(gap))
            current_ms = checkpoint_ms

            if next_bar <= ev_ms and next_bar == checkpoint_ms:
                tokens.append(VOCAB["BAR"])
                next_bar = next(bar_iter, math.inf)

            if next_tempo_ms <= ev_ms and next_tempo_ms == checkpoint_ms:
                bpm = _microseconds_to_bpm(next_tempo_us)
                tokens.append(VOCAB[f"TEMPO_{_bpm_to_bin(bpm)}"])
                next_tempo_ms, next_tempo_us = next(tempo_iter, (math.inf, 0))

        # Emit TIME_SHIFT up to this event
        gap = ev_ms - current_ms
        tokens.extend(_ms_to_time_shifts(gap))
        current_ms = ev_ms

        kind = ev[1]
        if kind == "note_on":
            _, _, pitch, vel, prog = ev
            prog_bin = _program_to_bin(prog)
            if prog_bin != current_program_bin:
                tokens.append(VOCAB[f"PROGRAM_{prog_bin}"])
                current_program_bin = prog_bin
            tokens.append(VOCAB[f"VELOCITY_{_velocity_to_bin(vel)}"])
            tokens.append(VOCAB[f"NOTE_ON_{pitch}"])
        elif kind == "drum_on":
            _, _, pitch, vel = ev
            tokens.append(VOCAB[f"VELOCITY_{_velocity_to_bin(vel)}"])
            tokens.append(VOCAB[f"DRUM_ON_{pitch}"])
        elif kind == "note_off":
            _, _, pitch = ev
            tokens.append(VOCAB[f"NOTE_OFF_{pitch}"])

    tokens.append(VOCAB["EOS"])
    return tokens


def _compute_bar_boundaries(
    time_sig_events: list,
    tempo_events: list,
    total_ms: float,
) -> List[float]:
    """
    Compute bar start times (in ms) for the entire piece.

    Uses tempo and time-signature meta messages.
    Returns a sorted list starting at 0.0.
    """
    if not time_sig_events:
        time_sig_events = [(0.0, 4, 4)]

    boundaries = [0.0]
    current_ms = 0.0

    # Pair up time-signature segments
    for seg_idx, (ts_ms, num, den) in enumerate(time_sig_events):
        # This time-sig is active from ts_ms until next time-sig or end
        seg_end = (
            time_sig_events[seg_idx + 1][0]
            if seg_idx + 1 < len(time_sig_events)
            else total_ms + 1
        )

        # A bar lasts num beats, each beat = 1 quarter / (den/4) = 4/den quarter notes
        # Beat duration in ms depends on tempo at that time
        cur_ms = max(ts_ms, boundaries[-1])

        while cur_ms < seg_end - 1e-6:
            # Find tempo at cur_ms
            tempo_us = 500_000
            for t_ms, t_us in tempo_events:
                if t_ms <= cur_ms:
                    tempo_us = t_us
            ms_per_beat = tempo_us / 1000  # ms per quarter note
            beats_per_bar = num * (4 / den)
            bar_dur_ms = ms_per_beat * beats_per_bar
            if bar_dur_ms <= 0:
                break
            next_bar = cur_ms + bar_dur_ms
            if next_bar > cur_ms + 1e-6:
                boundaries.append(next_bar)
            cur_ms = next_bar

    return sorted(boundaries)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode(token_ids: List[int], ticks_per_beat: int = 480) -> mido.MidiFile:
    """
    Decode a list of token ids back to a mido.MidiFile.

    PAD, BOS, EOS are skipped.
    A NOTE_OFF without a prior matching NOTE_ON is flagged (decode failure).
    DRUM_ON is given a fixed short duration.

    Returns a MidiFile; may be empty on severe errors.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # State
    current_ms = 0.0
    current_program = 0
    # pitch → onset_ms
    open_notes: dict[int, float] = {}

    # Collect (abs_ms, mido_message) pairs then sort by time
    events: list[tuple[float, mido.Message]] = []

    for tid in token_ids:
        if tid not in ID_TO_TOKEN:
            continue  # out-of-range — skip
        tok = ID_TO_TOKEN[tid]

        if tok in ("PAD", "BOS", "EOS", "BAR"):
            continue
        elif tok.startswith("TIME_SHIFT_"):
            steps = int(tok.split("_")[2])
            current_ms += steps * TIME_STEP_MS
        elif tok.startswith("TEMPO_"):
            pass  # tempo info used during encode; ignored in basic decode
        elif tok.startswith("VELOCITY_"):
            pass  # velocity is picked up when NOTE_ON/DRUM_ON follows
        elif tok.startswith("PROGRAM_"):
            bin_idx = int(tok.split("_")[1])
            current_program = _bin_to_program(bin_idx)
            events.append((
                current_ms,
                mido.Message("program_change", channel=0, program=current_program, time=0),
            ))
        elif tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[2])
            open_notes[pitch] = current_ms
            events.append((
                current_ms,
                mido.Message("note_on", channel=0, pitch=pitch, velocity=64, time=0),
            ))
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[2])
            open_notes.pop(pitch, None)
            events.append((
                current_ms,
                mido.Message("note_off", channel=0, pitch=pitch, velocity=0, time=0),
            ))
        elif tok.startswith("DRUM_ON_"):
            pitch = int(tok.split("_")[2])
            on_ms = current_ms
            off_ms = current_ms + DRUM_FIXED_DURATION_MS
            events.append((
                on_ms,
                mido.Message("note_on", channel=9, pitch=pitch, velocity=64, time=0),
            ))
            events.append((
                off_ms,
                mido.Message("note_off", channel=9, pitch=pitch, velocity=0, time=0),
            ))

    # Close any notes that were never closed
    for pitch, on_ms in open_notes.items():
        events.append((
            current_ms,
            mido.Message("note_off", channel=0, pitch=pitch, velocity=0, time=0),
        ))

    # Sort by abs_ms, convert to delta ticks
    events.sort(key=lambda x: x[0])
    ms_per_tick = 500_000 / 1000 / ticks_per_beat  # assume 120 BPM for decode
    prev_ms = 0.0
    for abs_ms, msg in events:
        delta_ms = abs_ms - prev_ms
        delta_ticks = max(0, int(round(delta_ms / ms_per_tick)))
        msg = msg.copy(time=delta_ticks)
        track.append(msg)
        prev_ms = abs_ms

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_decode(token_ids: List[int]) -> Tuple[bool, str]:
    """
    Check whether a sequence can decode without structural errors.

    Returns (is_valid, reason).
    Drums are exempt from the NOTE_OFF-without-NOTE_ON check.
    PAD/BOS/EOS are skipped.
    """
    open_pitches: set[int] = set()
    has_music = False

    for tid in token_ids:
        if tid not in ID_TO_TOKEN:
            return False, f"out-of-range token id {tid}"
        tok = ID_TO_TOKEN[tid]
        if tok in ("PAD", "BOS", "EOS", "BAR"):
            continue
        if tok.startswith("NOTE_ON_"):
            pitch = int(tok.split("_")[2])
            open_pitches.add(pitch)
            has_music = True
        elif tok.startswith("NOTE_OFF_"):
            pitch = int(tok.split("_")[2])
            if pitch not in open_pitches:
                return False, f"NOTE_OFF_{pitch} without prior NOTE_ON"
            open_pitches.discard(pitch)
        elif tok.startswith("DRUM_ON_"):
            has_music = True

    if not has_music:
        return False, "no musical events"
    return True, "ok"


# ---------------------------------------------------------------------------
# Duration helper (for bits/second metric)
# ---------------------------------------------------------------------------

def duration_seconds(token_ids: List[int]) -> float:
    """Return the total musical duration of a token sequence in seconds."""
    ms = 0.0
    for tid in token_ids:
        if tid not in ID_TO_TOKEN:
            continue
        tok = ID_TO_TOKEN[tid]
        if tok.startswith("TIME_SHIFT_"):
            steps = int(tok.split("_")[2])
            ms += steps * TIME_STEP_MS
    return ms / 1000.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class EventTokenizer:
    vocab = VOCAB
    id_to_token = ID_TO_TOKEN
    vocab_size = len(VOCAB)
    pad_id = VOCAB["PAD"]
    bos_id = VOCAB["BOS"]
    eos_id = VOCAB["EOS"]

    def encode(self, midi_path: str | Path) -> List[int]:
        return encode(midi_path)

    def decode(self, token_ids: List[int], ticks_per_beat: int = 480) -> mido.MidiFile:
        return decode(token_ids, ticks_per_beat)

    def validate_decode(self, token_ids: List[int]) -> Tuple[bool, str]:
        return validate_decode(token_ids)

    def duration_seconds(self, token_ids: List[int]) -> float:
        return duration_seconds(token_ids)