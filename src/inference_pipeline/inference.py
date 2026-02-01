"""
Inference pipeline for Housing Regression MLE.

- Takes RAW input data (same schema as holdout.csv).
- Applies preprocessing + feature engineering using saved encoders.
- Aligns features with training.
- Returns predictions.
"""

# Raw → preprocess → feature engineering → align schema → model.predict → predictions.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load


# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)

# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "SalePrice"]  # excluding price column
else:
    TRAIN_FEATURE_COLUMNS = None


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
) -> pd.DataFrame:
    # Step 1: Preprocess raw input
    df = input_df

    y_true = None
    if "SalePrice" in df.columns:
        y_true = df["SalePrice"].tolist()
        df = df.drop(columns=["SalePrice"])

    # Step 5: Align columns with training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Step 6: Load model & predict
    model = load(model_path)
    preds = model.predict(df)

    # Step 7: Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
