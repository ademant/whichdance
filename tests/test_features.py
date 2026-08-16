import numpy as np

from whichdance import config
from whichdance.features import extract_logmel, fix_length


def test_fix_length_pads_short_audio():
    y = np.zeros(1000, dtype=np.float32)
    out = fix_length(y)
    assert len(out) == int(config.CLIP_SECONDS * config.SAMPLE_RATE)


def test_fix_length_trims_long_audio():
    target = int(config.CLIP_SECONDS * config.SAMPLE_RATE)
    y = np.zeros(target * 2, dtype=np.float32)
    out = fix_length(y)
    assert len(out) == target


def test_extract_logmel_shape():
    y = np.random.randn(int(config.CLIP_SECONDS * config.SAMPLE_RATE)).astype(np.float32)
    features = extract_logmel(y)
    assert features.shape == (1, config.N_MELS, config.N_FRAMES)
