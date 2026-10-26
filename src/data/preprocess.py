"""
preprocess.py
-------------
Loads raw Lending Club data, cleans it, engineers features,
and splits it into yearly time windows for drift analysis.
"""

import pandas as pd
import numpy as np
import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path):
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Raw data shape: {df.shape}")
    return df


def clean_data(df, config):
    """Clean and prepare the dataset."""
    logger.info("Cleaning data...")

    # loans still in progress (e.g. "Current") have no resolved outcome, so only
    # keep rows that ended in one of these five statuses.
    target_map = {
        "Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1,
        "Late (31-120 days)": 1,
        "Late (16-30 days)": 1,
    }
    df = df[df[config["data"]["target_column"]].isin(target_map.keys())].copy()
    df["target"] = df[config["data"]["target_column"]].map(target_map)

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["year"] = df["issue_d"].dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    start = config["data"]["start_year"]
    end = config["data"]["end_year"]
    df = df[(df["year"] >= start) & (df["year"] <= end)]

    numerical = config["features"]["numerical"]
    for col in numerical:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical = config["features"]["categorical"]
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # int_rate and revol_util come in as "13.99%" strings in the raw CSV
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "").astype(float)
    if "emp_length" in df.columns:
        emp_map = {
            "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
            "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
            "8 years": 8, "9 years": 9, "10+ years": 10, "n/a": -1
        }
        df["emp_length"] = df["emp_length"].map(emp_map).fillna(-1)

    keep_cols = numerical + categorical + ["target", "year"]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # rows missing more than 30% of their columns aren't worth imputing
    df = df.dropna(thresh=int(len(df.columns) * 0.7))

    for col in numerical:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    logger.info(f"Cleaned data shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    return df


def build_category_schema(df, config):
    """Capture the full set of categories per categorical feature from the full
    cleaned dataset. This shared schema is the 'fitted encoder': applying it to
    each window guarantees identical, correctly-aligned one-hot columns even when
    a category is absent in a given year."""
    categorical = [c for c in config["features"]["categorical"] if c in df.columns]
    schema = {col: sorted(df[col].astype(str).unique()) for col in categorical}
    return schema


def encode_categoricals(df, config, category_schema=None):
    """One-hot encode categorical features.

    When `category_schema` is provided, each categorical column is cast to a
    pandas Categorical with the full, fixed category list before get_dummies.
    This makes drop_first deterministic across windows (the same baseline is
    always dropped) and emits an all-zero column for categories absent in the
    current slice - so every window shares an identical column schema.
    """
    logger.info("Encoding categorical features...")
    categorical = [c for c in config["features"]["categorical"] if c in df.columns]
    df = df.copy()
    if category_schema is not None:
        for col in categorical:
            df[col] = pd.Categorical(df[col].astype(str), categories=category_schema[col])
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    return df


def split_by_year(df_cleaned, config, category_schema, master_columns):
    """Split the (un-encoded) cleaned data into yearly windows, encoding each
    window with the shared schema and aligning it to `master_columns` so every
    window CSV has the exact same columns in the same order."""
    windows_path = Path(config["data"]["windows_path"])
    windows_path.mkdir(parents=True, exist_ok=True)

    years = sorted(df_cleaned["year"].unique())
    logger.info(f"Splitting into yearly windows: {years}")

    for year in years:
        window_df = df_cleaned[df_cleaned["year"] == year].copy()
        encoded_window = encode_categoricals(window_df, config, category_schema)
        # Align to the shared schema: add missing dummy cols as 0, keep order.
        encoded_window = encoded_window.reindex(columns=master_columns, fill_value=0)
        out_path = windows_path / f"window_{year}.csv"
        encoded_window.to_csv(out_path, index=False)
        logger.info(
            f"  Year {year}: {len(encoded_window)} records, "
            f"{encoded_window.shape[1]} cols -> {out_path}"
        )

    return years


def run_preprocessing(config_path="configs/config.yaml"):
    config = load_config(config_path)

    raw_df = load_raw_data(config["data"]["raw_path"])
    cleaned_df = clean_data(raw_df, config)

    processed_path = Path(config["data"]["processed_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed data to {processed_path}")

    # Fit the encoder on the full cleaned dataset (not per-window) so every
    # yearly window ends up with identical, correctly-aligned columns.
    category_schema = build_category_schema(cleaned_df, config)
    encoded_df = encode_categoricals(cleaned_df, config, category_schema)
    master_columns = list(encoded_df.columns)
    logger.info(f"Shared encoder schema: {len(master_columns)} columns")

    years = split_by_year(cleaned_df, config, category_schema, master_columns)

    logger.info("Preprocessing complete!")
    return encoded_df, years


if __name__ == "__main__":
    run_preprocessing()
