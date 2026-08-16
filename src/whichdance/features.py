"""Audio loading and feature extraction.

Turns an audio file into a fixed-shape log-mel spectrogram tensor of shape
(1, N_MELS, N_FRAMES), suitable as a single-channel "image" input to a CNN.
"""

from __future__ import annotations

import numpy as np
import librosa
import torch

from whichdance import config


def load_audio(path: str) -> np.ndarray:
    """Load an audio file as mono float32 at config.SAMPLE_RATE."""
    y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
    return y


def fix_length(y: np.ndarray) -> np.ndarray:
    """Clip or pad audio to exactly CLIP_SECONDS.

    For clips longer than CLIP_SECONDS, take the middle section (intros/
    outros are often less representative of the dance rhythm than the body
    of the tune). Shorter clips are zero-padded.
    """
    target_len = int(config.CLIP_SECONDS * config.SAMPLE_RATE)
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start : start + target_len]
    pad = target_len - len(y)
    left = pad // 2
    right = pad - left
    return np.pad(y, (left, right))


def extract_logmel(y: np.ndarray) -> torch.Tensor:
    """Compute a log-mel spectrogram, shape (1, N_MELS, N_FRAMES)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Per-sample normalization to zero mean / unit variance.
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    tensor = torch.from_numpy(log_mel).float().unsqueeze(0)

    # Guard against off-by-one frame counts from librosa; pad/trim to exact.
    n_frames = tensor.shape[-1]
    if n_frames < config.N_FRAMES:
        tensor = torch.nn.functional.pad(tensor, (0, config.N_FRAMES - n_frames))
    elif n_frames > config.N_FRAMES:
        tensor = tensor[..., : config.N_FRAMES]
    return tensor


def audio_file_to_features(path: str) -> torch.Tensor:
    """End-to-end: audio file path -> model-ready feature tensor."""
    y = load_audio(path)
    y = fix_length(y)
    return extract_logmel(y)
