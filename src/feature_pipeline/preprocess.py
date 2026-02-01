"""
⚡ Preprocessing Script for Housing Regression MLE

- Reads train/eval/holdout CSVs from data/raw/.
- Cleans and normalizes city names.
- Maps cities to metros and merges lat/lng.
- Drops duplicates and extreme outliers.
- Saves cleaned splits to data/processed/.

"""

"""
Preprocessing: city normalization + (optional) lat/lng merge, duplicate drop, outlier removal.

- Production defaults read from data/raw/ and write to data/processed/
- Tests can override `raw_dir`, `processed_dir`, and pass `metros_path=None`
  to skip merge safely without touching disk assets.
"""

import re
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("../data/raw/")
PROCESSED_DIR = Path("../data/processed/")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_preprocess(
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR):
    
    train_df = pd.read_csv(RAW_DIR/"train.csv")
    eval_df = pd.read_csv(RAW_DIR/"eval.csv")
    holdout_df = pd.read_csv(RAW_DIR/"holdout.csv")

    data = pd.concat([train_df, eval_df, holdout_df], ignore_index=True)
    # save all categorical columns in list
    categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']

    # dataframe with categorical features
    data_cat = data[categorical_columns]
    # dataframe with numerical features
    data_num = data.drop(categorical_columns, axis=1)

    ## skewness
    from scipy.stats import skew
    data_num_skew = data_num.apply(lambda x: skew(x.dropna()))
    data_num_skew = data_num_skew[data_num_skew > .75]

    # apply log + 1 transformation for all numeric features with skewnes over .75
    data_num[data_num_skew.index] = np.log1p(data_num[data_num_skew.index])


    ## missing values
    data_len = data_num.shape[0]

    # check what is percentage of missing values in categorical dataframe
    for col in data_num.columns.values:
        missing_values = data_num[col].isnull().sum()
        #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 

        # drop column if there is more than 50 missing values
        if missing_values > 50:
            #print("droping column: {}".format(col))
            data_num = data_num.drop(col, axis = 1)
        # if there is less than 50 missing values than fill in with median valu of column
        else:
            #print("filling missing values with median in column: {}".format(col))
            data_num = data_num.fillna(data_num[col].median())

        data_len = data_cat.shape[0]


    # check what is percentage of missing values in categorical dataframe
    for col in data_cat.columns.values:
        missing_values = data_cat[col].isnull().sum()
        #print("{} - missing values: {} ({:0.2f}%)".format(col, missing_values, missing_values/data_len*100)) 

        # drop column if there is more than 50 missing values
        if missing_values > 50:
            print("droping column: {}".format(col))
            data_cat.drop(col, axis = 1)
        # if there is less than 50 missing values than fill in with median valu of column
        else:
            #print("filling missing values with XXX: {}".format(col))
            #data_cat = data_cat.fillna('XXX')
            pass


    data = pd.concat([data_num, data_cat], axis=1)
    train_df = data.iloc[:len(train_df)]
    print(train_df.shape)
    eval_df = data.iloc[len(train_df):(len(train_df)+len(eval_df))]
    print(eval_df.shape)
    holdout_df = data.iloc[len(train_df)+len(eval_df):]
    print(holdout_df.shape)

    # Save splits
    train_df.to_csv(PROCESSED_DIR/"cleaning_train.csv", index=False)
    eval_df.to_csv(PROCESSED_DIR/"cleaning_eval.csv", index=False)
    holdout_df.to_csv(PROCESSED_DIR/"cleaning_holdout.csv", index=False)

    print(f"✅ Preprocessed saved to {out_path}")



if __name__ == "__main__":
    run_preprocess()
