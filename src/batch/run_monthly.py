"""
Batch runner for monthly predictions.

- Loads holdout data
- Splits by year/month
- Runs inference
- Saves predictions per month to data/predictions/
"""

from pathlib import Path
import pandas as pd
from src.inference_pipeline.inference import predict

# -------------------
# Paths
# -------------------
DATA_DIR = Path("data/processed")
HOLDOUT_PATH = DATA_DIR / "cleaning_holdout.csv"
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_style_predictions():
    # Load holdout
    df = pd.read_csv(HOLDOUT_PATH)

    # Group by year + month
    grouped = df.groupby(["BldgType", "HouseStyle"])

    all_outputs = []
    for (BldgType, HouseStyle), group in grouped:
        print(f" Running predictions:")

        preds_df = predict(group)

        out_path = OUTPUT_DIR / f"preds_{BldgType}_{HouseStyle}.csv"
        preds_df.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        all_outputs.append(preds_df)

    return pd.concat(all_outputs, ignore_index=True)


if __name__ == "__main__":
    all_preds = run_style_predictions()
    print("🎉 Batch inference complete.")
    print(all_preds.head())
