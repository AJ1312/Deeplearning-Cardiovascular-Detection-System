"""Training utilities for heart disease model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_COLUMNS,
    RANDOM_SEED,
)
from .data import fit_scaler, load_dataset, make_train_test_split
from .model import HeartDiseaseNet


def set_seed(seed: int = RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    scaler_path: Path = DEFAULT_SCALER_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> Dict[str, float]:
    """Train model, persist artifacts, and return evaluation metrics."""
    set_seed()

    df = load_dataset(data_path)
    x_train, x_test, y_train, y_test = make_train_test_split(df)

    scaler = fit_scaler(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = HeartDiseaseNet(input_size=len(FEATURE_COLUMNS))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_prob = model(x_test_tensor).squeeze().numpy()

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
