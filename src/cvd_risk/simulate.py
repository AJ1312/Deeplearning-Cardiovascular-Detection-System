"""Risk simulation helper for one-feature what-if analysis."""

from __future__ import annotations

from typing import Dict

from .config import FEATURE_COLUMNS
from .predict import predict_from_records


def simulate_risk_change(input_data: Dict[str, float], modified_feature: str, new_value: float):
    """Compare baseline and modified risk for a single patient profile."""
    if modified_feature not in FEATURE_COLUMNS:
        raise ValueError(f"Feature '{modified_feature}' is not valid.")

    base_input = dict(input_data)
    changed_input = dict(input_data)
    changed_input[modified_feature] = new_value

    old_risk = predict_from_records([base_input])[0]
    new_risk = predict_from_records([changed_input])[0]

    if old_risk == 0:
        change_pct = float("inf")
    else:
        change_pct = ((new_risk - old_risk) / old_risk) * 100.0

    return {
        "old_risk": old_risk,
        "new_risk": new_risk,
        "change_percent": change_pct,
        "modified_feature": modified_feature,
        "new_value": new_value,
    }
