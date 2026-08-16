"""Build stratified train/val/test CSV splits from a master labels.csv.

Usage:
    python -m whichdance.split --labels data/labels.csv --out data/splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from whichdance import config


def build_splits(labels_csv: str, out_dir: str) -> None:
    df = pd.read_csv(labels_csv)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("labels.csv must have columns: filepath,label")

    counts = df["label"].value_counts()
    too_rare = counts[counts < 3]
    if not too_rare.empty:
        print(
            "Warning: these labels have < 3 examples and may break "
            f"stratified splitting: {too_rare.to_dict()}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=config.VAL_FRACTION + config.TEST_FRACTION,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
    )
    rel_test = config.TEST_FRACTION / (config.VAL_FRACTION + config.TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"classes={df['label'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(config.DATA_DIR / "labels.csv"))
    parser.add_argument("--out", default=str(config.SPLITS_DIR))
    args = parser.parse_args()
    build_splits(args.labels, args.out)


if __name__ == "__main__":
    main()
