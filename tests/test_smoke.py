from pathlib import Path

from cvd_risk.data import load_dataset


def test_dataset_exists_and_loads():
    path = Path("data/raw/heart.csv")
    assert path.exists()
    df = load_dataset(path)
    assert not df.empty
    assert "target" in df.columns
