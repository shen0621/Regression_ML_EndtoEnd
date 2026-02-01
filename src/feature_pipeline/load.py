"""
Load & time-split the raw dataset.

- Production default writes to data/raw/
- Tests can pass a temp `output_dir` so nothing in data/ is touched.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("../data/raw/")


def load_and_split_data(
    raw_path: str = "/Users/yaoshen/Documents/ML/Dataset/house_price/train.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load raw dataset, split into train/eval/holdout by date, and save to output_dir."""
    df = pd.read_csv(raw_path)
    print(df.shape)
    df.head(4)

    train_df, eval_df, holdout_df = np.split(df.sample(frac=1, random_state=42), 
                                   [int(.5 * len(df)), int(.75 * len(df))])

    print(f"Part 1 (50%): {len(train_df)} rows")
    print(f"Part 2 (25%): {len(eval_df)} rows")
    print(f"Part 3 (25%): {len(holdout_df)} rows")

    # Save
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train.csv", index=False)
    eval_df.to_csv(outdir / "eval.csv", index=False)
    holdout_df.to_csv(outdir / "holdout.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
