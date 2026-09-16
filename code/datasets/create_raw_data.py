"""Save scikit-learn's bundled Wine dataset as a local CSV."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[2]


def main():
    wine = load_wine(as_frame=True)
    frame = wine.data.copy()
    frame.columns = frame.columns.str.lower().str.replace(r"[^a-z0-9_]+", "_", regex=True)
    frame["target"] = [wine.target_names[index] for index in wine.target]
    output = ROOT / "data/raw/wine.csv"
    if output.exists():
        existing = pd.read_csv(output)
        if set(existing.columns) != set(frame.columns) or existing.empty:
            raise ValueError(f"Existing CSV has an invalid schema or no rows: {output}")
        if not existing["target"].dropna().isin(wine.target_names).all():
            raise ValueError(f"Existing CSV contains unknown target labels: {output}")
        print(f"Kept {output}: {len(existing)} rows, 13 features, "
              f"{existing['target'].nunique()} classes")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {output}: {len(frame)} rows, 13 features, 3 classes")


if __name__ == "__main__":
    main()
