"""Load UCI Maternal Health Risk dataset + validate user input."""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.config import DATA_PATH, FEATURE_RANGES, FEATURE_ORDER

NUMERIC_COLS = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def load_dataset(save_if_missing: bool = True) -> pd.DataFrame:
    """Load cached CSV or fetch from UCI (ID=863). Always returns clean dtypes."""

    # 1) Try local CSV cache
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        dataset = fetch_ucirepo(id=863)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        # UCI returns target as 'RiskLevel' but column order can vary
        df.columns = NUMERIC_COLS + ["RiskLevel"]
        if save_if_missing:
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)

    # 2) Force numeric dtypes (defends against pyarrow extension arrays)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["RiskLevel"] = df["RiskLevel"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=NUMERIC_COLS + ["RiskLevel"]).reset_index(drop=True)

    return df


def validate_input(features: dict) -> tuple[bool, str]:
    """Return (True, '') if all features in-range, else (False, message)."""
    for name, (lo, hi) in FEATURE_RANGES.items():
        val = features.get(name)
        if val is None:
            return False, f"{name} is required."
        if not (lo <= val <= hi):
            return False, f"{name} must be between {lo} and {hi}."
    return True, ""