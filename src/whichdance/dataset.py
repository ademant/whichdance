"""PyTorch Dataset over (audio filepath, dance label) rows.

Features are cached to disk on first access (data/processed/<hash>.pt) so
repeated epochs don't re-run librosa's feature extraction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from whichdance import config
from whichdance.features import audio_file_to_features


class LabelEncoder:
    """Minimal, dependency-free string <-> int label encoder."""

    def __init__(self, labels: list[str]):
        self.classes_ = sorted(set(labels))
        self._to_idx = {label: i for i, label in enumerate(self.classes_)}

    def encode(self, label: str) -> int:
        return self._to_idx[label]

    def decode(self, idx: int) -> str:
        return self.classes_[idx]

    def __len__(self) -> int:
        return len(self.classes_)

    def to_dict(self) -> dict:
        return {"classes": self.classes_}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelEncoder":
        enc = cls.__new__(cls)
        enc.classes_ = d["classes"]
        enc._to_idx = {label: i for i, label in enumerate(enc.classes_)}
        return enc


class TuneDataset(Dataset):
    """Expects a CSV with columns: filepath,label.

    `filepath` is resolved relative to `root` (defaults to data/).
    """

    def __init__(
        self,
        csv_path: str,
        label_encoder: LabelEncoder,
        root: Path = config.DATA_DIR,
        cache_dir: Path = config.PROCESSED_DIR,
    ):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.df)

    def _cache_path(self, filepath: str) -> Path:
        digest = hashlib.sha1(filepath.encode()).hexdigest()[:16]
        return self.cache_dir / f"{digest}.pt"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        cache_path = self._cache_path(filepath)

        if cache_path.exists():
            features = torch.load(cache_path, weights_only=True)
        else:
            audio_path = self.root / filepath
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"No cached features and no local audio for {filepath!r}. "
                    "If this row came from whichdance.funkwhale_import with "
                    "raw audio discarded, re-run the import to rebuild the cache."
                )
            features = audio_file_to_features(str(audio_path))
            torch.save(features, cache_path)

        label = self.label_encoder.encode(row["label"])
        return features, label

    @staticmethod
    def build_label_encoder(*csv_paths: str) -> LabelEncoder:
        labels: list[str] = []
        for path in csv_paths:
            labels.extend(pd.read_csv(path)["label"].tolist())
        return LabelEncoder(labels)
