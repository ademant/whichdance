"""Auxiliary, non-model metrics computed alongside a dance prediction.

These are cheap signal-processing estimates (not learned) that are useful
to show a user next to the model's guess — tempo in particular is a big
part of *why* a dance gets predicted, since many traditional dances have
fairly characteristic BPM ranges.
"""

from __future__ import annotations

import numpy as np
import librosa


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate tempo in beats per minute via librosa's beat tracker."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa returns a 0-d array in some versions, a float in others.
    return float(np.atleast_1d(tempo)[0])


def estimate_duration(y: np.ndarray, sr: int) -> float:
    """Duration in seconds of the (already loaded) audio."""
    return len(y) / sr
