"""Data loading and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Load dataset and remove duplicate rows."""
    df = pd.read_csv(data_path)
    return df.drop_duplicates().reset_index(drop=True)


def make_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Create stratified train-test split from dataframe."""
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )


def fit_scaler(x_train: pd.DataFrame) -> StandardScaler:
    """Fit scaler on training features."""
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler
