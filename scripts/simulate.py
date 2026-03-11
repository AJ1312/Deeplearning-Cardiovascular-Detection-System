#!/usr/bin/env python3
"""Run single-feature risk simulation from JSON-like CLI input."""

import argparse
import json

from cvd_risk.simulate import simulate_risk_change


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate what-if risk changes")
    parser.add_argument("--patient", required=True, help="Patient record as JSON string")
    parser.add_argument("--feature", required=True, help="Feature to modify")
    parser.add_argument("--value", required=True, type=float, help="New value")
    args = parser.parse_args()

    patient = json.loads(args.patient)
    result = simulate_risk_change(patient, args.feature, args.value)
    print(json.dumps(result, indent=2))
