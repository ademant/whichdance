"""Train DanceCNN on train.csv / val.csv splits.

Usage:
    python -m whichdance.train --splits data/splits --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from whichdance import config
from whichdance.dataset import TuneDataset
from whichdance.model import DanceCNN


def compute_class_weights(dataset: TuneDataset) -> torch.Tensor:
    """Inverse-frequency weights, for a weighted cross-entropy loss that
    doesn't let common dances dominate a likely-imbalanced dataset."""
    counts = dataset.df["label"].value_counts()
    weights = [
        1.0 / counts[dataset.label_encoder.decode(i)]
        for i in range(len(dataset.label_encoder))
    ]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights / weights.sum() * len(weights)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)

    return total_loss / n, correct / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default=str(config.SPLITS_DIR))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    args = parser.parse_args()

    splits = Path(args.splits)
    train_csv = splits / "train.csv"
    val_csv = splits / "val.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    label_encoder = TuneDataset.build_label_encoder(str(train_csv), str(val_csv))
    train_ds = TuneDataset(str(train_csv), label_encoder)
    val_ds = TuneDataset(str(val_csv), label_encoder)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = DanceCNN(num_classes=len(label_encoder)).to(device)
    weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "label_encoder.json", "w") as f:
        json.dump(label_encoder.to_dict(), f, indent=2)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch {epoch:3d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
            f"| val loss {val_loss:.3f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    torch.save(model.state_dict(), ckpt_dir / "last.pt")
    print(f"best val acc: {best_val_acc:.3f}")


if __name__ == "__main__":
    main()
