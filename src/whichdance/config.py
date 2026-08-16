"""Central hyperparameters and paths.

Kept as plain module-level constants (not a framework config object) so
they're easy to read, override, and pass around without extra dependencies.
"""

from pathlib import Path

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

# --- Audio / feature extraction ---
SAMPLE_RATE = 22050
CLIP_SECONDS = 30.0  # tunes are clipped/padded to this length
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Derived: number of time frames in a spectrogram of CLIP_SECONDS
N_FRAMES = int(CLIP_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1

# --- Training ---
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
RANDOM_SEED = 42
