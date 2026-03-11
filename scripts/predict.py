#!/usr/bin/env python3
"""Run prediction on one or many records from a CSV file."""

import argparse
from pathlib import Path

import pandas as pd

from cvd_risk.predict import load_model_and_scaler, predict_proba_from_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict CVD risk probabilities from CSV")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing feature columns")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    model, scaler = load_model_and_scaler()
    df = pd.read_csv(args.input)
    probs = predict_proba_from_dataframe(df, model, scaler)
    df["predicted_risk"] = probs

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
    else:
        print(df[["predicted_risk"]].head(10).to_string(index=False))
