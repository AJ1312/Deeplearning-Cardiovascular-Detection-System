"""Project-wide constants and default paths."""

from pathlib import Path

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN = "target"
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_disease_net.pt"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
