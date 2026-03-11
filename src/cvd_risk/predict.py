"""Inference helpers for heart disease model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import joblib
import numpy as np
import pandas as pd
import torch

from .config import DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH, FEATURE_COLUMNS
from .model import HeartDiseaseNet


def load_model_and_scaler(
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
):
    model = HeartDiseaseNet(input_size=len(FEATURE_COLUMNS))
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_proba_from_dataframe(df: pd.DataFrame, model: HeartDiseaseNet, scaler) -> np.ndarray:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    features = df[FEATURE_COLUMNS]
    scaled = scaler.transform(features)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        return model(tensor).squeeze().numpy()


def predict_from_records(records: Iterable[dict]) -> List[float]:
    model, scaler = load_model_and_scaler()
    df = pd.DataFrame(records)
    probs = predict_proba_from_dataframe(df, model, scaler)
    if np.isscalar(probs):
        return [float(probs)]
    return [float(p) for p in probs]
