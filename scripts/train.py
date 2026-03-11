#!/usr/bin/env python3
"""Train and persist the heart disease model artifacts."""

import argparse
from pathlib import Path

from cvd_risk.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train heart disease risk model")
    parser.add_argument("--data", type=Path, default=None, help="Path to input CSV dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    kwargs = {"epochs": args.epochs}
    if args.data:
        kwargs["data_path"] = args.data

    metrics = train_model(**kwargs)
    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
