"""Run inference on a single audio file.

Usage:
    python -m whichdance.predict --audio path/to/tune.mp3 \
        --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from whichdance import config
from whichdance.analysis import estimate_bpm, estimate_duration
from whichdance.dataset import LabelEncoder
from whichdance.features import extract_logmel, fix_length, load_audio
from whichdance.model import DanceCNN


@dataclass
class Prediction:
    top_labels: list[tuple[str, float]]  # (label, probability), sorted desc
    bpm: float
    duration_seconds: float


def load_model(checkpoint_dir: str) -> tuple[DanceCNN, LabelEncoder, torch.device]:
    ckpt_dir = Path(checkpoint_dir).parent if Path(checkpoint_dir).is_file() else Path(checkpoint_dir)
    checkpoint_path = Path(checkpoint_dir) if Path(checkpoint_dir).is_file() else ckpt_dir / "best.pt"

    with open(ckpt_dir / "label_encoder.json") as f:
        label_encoder = LabelEncoder.from_dict(json.load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DanceCNN(num_classes=len(label_encoder)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    return model, label_encoder, device


def predict_with_model(
    audio_path: str,
    model: DanceCNN,
    label_encoder: LabelEncoder,
    device: torch.device,
    top_k: int = 3,
) -> Prediction:
    """Run inference using an already-loaded model.

    Prefer this over `predict()` for a long-running process (e.g. the
    FastAPI service) so the checkpoint isn't reloaded from disk per request.
    """
    y = load_audio(audio_path)
    bpm = estimate_bpm(y, config.SAMPLE_RATE)
    duration = estimate_duration(y, config.SAMPLE_RATE)

    clipped = fix_length(y)
    features = extract_logmel(clipped).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(features), dim=1).squeeze(0)

    top = torch.topk(probs, k=min(top_k, len(label_encoder)))
    top_labels = [
        (label_encoder.decode(idx.item()), prob.item())
        for prob, idx in zip(top.values, top.indices)
    ]
    return Prediction(top_labels=top_labels, bpm=bpm, duration_seconds=duration)


def predict(audio_path: str, checkpoint_dir: str, top_k: int = 3) -> Prediction:
    """Convenience one-shot: load the model fresh, then run inference.

    Fine for the CLI; a long-running server should call load_model() once
    and reuse it via predict_with_model() instead.
    """
    model, label_encoder, device = load_model(checkpoint_dir)
    return predict_with_model(audio_path, model, label_encoder, device, top_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best.pt"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    result = predict(args.audio, args.checkpoint, args.top_k)
    print(f"tempo:    {result.bpm:.0f} bpm")
    print(f"duration: {result.duration_seconds:.0f}s")
    print()
    for label, prob in result.top_labels:
        print(f"{label:20s} {prob:.3f}")


if __name__ == "__main__":
    main()
