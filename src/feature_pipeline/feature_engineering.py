"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""

import pandas as pd
import numpy as np
from scipy.stats import skew

PROCESSED_DIR = Path("../data/processed/")


# ---------- feature functions ----------

def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)


    train_df = train_df.drop(train_df[train_df['LotFrontage']>150].index)
    train_df['LotFrontage']=train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean())
    train_df = train_df.drop(train_df[train_df['LotArea']>50001].index)

    all_data = pd.concat([train_df, eval_df, holdout_df], ignore_index=True)

    print("\nall_data",all_data.shape)
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    categorical_features = pd.DataFrame(all_data.describe(include = ['O'])).columns

    prices = pd.DataFrame({"price":train_df["SalePrice"], "log(price + 1)":np.log1p(train_df["SalePrice"])})
    train_df["SalePrice"] = np.log1p(train_df["SalePrice"])


    #missing data
    REMOVING_THRESH = 0.8

    total = all_data.isnull().sum().sort_values(ascending=False)
    percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(10))

    all_data = all_data.drop((missing_data[missing_data['Percent'] > REMOVING_THRESH]).index,1)


    all_data['LotArea']=np.log1p(all_data['LotArea'])

    all_data["TotBsmtFin"] = all_data["BsmtFinSF1"] + all_data["BsmtFinSF2"]

    all_data = all_data.drop("BsmtFinSF1",1)
    all_data = all_data.drop("BsmtFinSF2",1)

    all_data["TotBath"] = all_data["FullBath"] + 0.5*all_data["HalfBath"] + all_data["BsmtFullBath"] + 0.5*all_data["BsmtHalfBath"]

    all_data = all_data.drop("FullBath",1)
    all_data = all_data.drop("HalfBath",1)
    all_data = all_data.drop("BsmtFullBath",1)
    all_data = all_data.drop("BsmtHalfBath",1)


    all_data["TotArea"] = all_data["GrLivArea"] + all_data["TotalBsmtSF"]

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

    skewed_feats = skewed_feats[skewed_feats > 0.1]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = all_data.drop("BsmtFinType1",1)
    all_data = all_data.drop("2ndFlrSF",1)
    all_data = all_data.drop("BedroomAbvGr",1)

    all_data = all_data.drop("LowQualFinSF",1)
    all_data = all_data.drop("3SsnPorch",1)
    all_data = all_data.drop('Condition2',1)

    dummies = pd.get_dummies(all_data)
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.median())
    print("all_data dim: ",all_data.shape)

    train_df = all_data.iloc[:len(train_df)]
    print(train_df.shape)
    eval_df = all_data.iloc[len(train_df):(len(train_df)+len(eval_df))]
    print(eval_df.shape)
    holdout_df = all_data.iloc[len(train_df)+len(eval_df):]
    print(holdout_df.shape)

    
    # Save engineered data
    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    run_feature_engineering()
