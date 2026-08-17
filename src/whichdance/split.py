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

# A label needs at least this many examples to survive two rounds of
# stratified splitting (train/temp, then val/test) without a class dropping
# to zero members on one side. Below that, sklearn's train_test_split raises
# rather than guess how to divide a single example.
MIN_COUNT_FOR_SPLIT = 4


def build_splits(labels_csv: str, out_dir: str) -> None:
    df = pd.read_csv(labels_csv)
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("labels.csv must have columns: filepath,label")

    counts = df["label"].value_counts()
    rare_labels = counts[counts < MIN_COUNT_FOR_SPLIT].index
    rare_df = df[df["label"].isin(rare_labels)]
    common_df = df[~df["label"].isin(rare_labels)]

    if not rare_df.empty:
        print(
            f"Warning: these labels have < {MIN_COUNT_FOR_SPLIT} examples, too few to "
            "stratify into train/val/test - putting all of them into train only "
            f"(won't appear in val/test): {counts[rare_labels].to_dict()}"
        )

    train_df, temp_df = train_test_split(
        common_df,
        test_size=config.VAL_FRACTION + config.TEST_FRACTION,
        stratify=common_df["label"],
        random_state=config.RANDOM_SEED,
    )
    rel_test = config.TEST_FRACTION / (config.VAL_FRACTION + config.TEST_FRACTION)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test,
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED,
    )

    train_df = pd.concat([train_df, rare_df], ignore_index=True)

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
