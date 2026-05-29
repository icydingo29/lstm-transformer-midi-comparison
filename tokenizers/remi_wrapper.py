"""
Thin wrapper around MidiTok's REMI+ tokenizer.

MidiTok ignores Sustain Pedal, Pitch Bend, and SysEx automatically
(it has no tokens for them), so no extra filtering is needed.

Usage:
    tok = REMIWrapper()
    ids = tok.encode(midi_path)   # List[int]
    mid = tok.decode(ids)         # symusic.Score (MidiTok ≥ 3.x) or mido.MidiFile
    secs = tok.duration_seconds(ids)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from miditok import REMI, TokenizerConfig
from miditok.constants import SCORE_LOADING_EXCEPTION
import symusic


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def _make_config() -> TokenizerConfig:
    return TokenizerConfig(
        pitch_range=(21, 109),        # 88-key piano range
        beat_res={(0, 4): 8, (4, 12): 4},  # 8 positions/beat for 0-4 beats, 4 for 4-12
        num_velocities=16,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_chords=False,
        use_rests=False,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        num_tempos=32,
        tempo_range=(50, 200),
    )


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class REMIWrapper:
    """Wraps MidiTok REMI+ with encode/decode/duration_seconds helpers."""

    def __init__(self, config: TokenizerConfig | None = None):
        cfg = config if config is not None else _make_config()
        self._tok = REMI(cfg)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._tok)

    @property
    def pad_id(self) -> int:
        return self._tok["PAD_None"]

    @property
    def bos_id(self) -> int:
        return self._tok["BOS_None"]

    @property
    def eos_id(self) -> int:
        return self._tok["EOS_None"]

    @property
    def vocab(self):
        return self._tok.vocab

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, midi_path: str | Path) -> List[int]:
        """
        Tokenize a MIDI file and return a flat list of token ids.

        Returns an empty list if the file cannot be loaded.
        """
        try:
            score = symusic.Score(str(midi_path))
        except Exception:
            return []

        try:
            tokseq = self._tok(score)
        except Exception:
            return []

        # MidiTok returns a TokSequence or list of TokSequence (one per track)
        if isinstance(tokseq, list):
            # Merge all tracks' ids into one flat sequence
            ids: List[int] = []
            for seq in tokseq:
                ids.extend(seq.ids)
            return ids
        return list(tokseq.ids)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int]) -> symusic.Score:
        """
        Decode a list of token ids back to a symusic.Score (the MidiTok 3.x return type).

        Returns an empty Score on failure.
        """
        from miditok import TokSequence
        seq = TokSequence(ids=token_ids, are_ids_encoded=True)
        try:
            score = self._tok(seq)
        except Exception:
            score = symusic.Score()
        return score

    # ------------------------------------------------------------------
    # Duration helper (for bits/second metric)
    # ------------------------------------------------------------------

    def duration_seconds(self, token_ids: List[int]) -> float:
        """
        Estimate the musical duration covered by the token sequence in seconds.

        Decodes to symusic.Score and reads its end time; falls back to summing
        Position tokens scaled to the beat resolution if decoding fails.
        """
        try:
            score = self.decode(token_ids)
            if score.tracks:
                end_tick = score.end()
                tpq = score.ticks_per_quarter
                # Use 120 BPM as default if no tempo track
                if score.tempos:
                    tempo_us = score.tempos[0].qpm  # MidiTok uses qpm (quarter per minute)
                    secs_per_tick = 60.0 / (tempo_us * tpq)
                else:
                    secs_per_tick = 60.0 / (120 * tpq)
                return end_tick * secs_per_tick
        except Exception:
            pass

        # Fallback: sum Position tokens' beat fractions and convert at 120 BPM
        return self._duration_from_tokens(token_ids)

    def _duration_from_tokens(self, token_ids: List[int]) -> float:
        """Rough duration estimate by counting Bar tokens at 120 BPM / 4/4."""
        id_to_str = {v: k for k, v in self._tok.vocab.items()}
        bar_count = sum(
            1 for tid in token_ids
            if id_to_str.get(tid, "").startswith("Bar_")
        )
        # Assume 4/4 at 120 BPM → 2 seconds per bar
        return max(bar_count * 2.0, 0.0)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def validate_decode(self, token_ids: List[int]) -> Tuple[bool, str]:
        """
        REMI+ almost always produces valid MIDI (MidiTok handles structure internally).
        Returns (True, 'ok') in nearly all cases; use only as a sanity/bug check.
        """
        try:
            score = self.decode(token_ids)
            has_music = any(len(t.notes) > 0 for t in score.tracks)
            if not has_music:
                return False, "no notes in decoded score"
            return True, "ok"
        except Exception as exc:
            return False, str(exc)