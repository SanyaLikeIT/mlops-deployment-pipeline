"""Clean all Wine features and write a deterministic stratified split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline",
]
CLASSES = {"class_0", "class_1", "class_2"}


def clean_data(frame):
    frame = frame.copy()
    frame.columns = (frame.columns.str.strip().str.lower()
                     .str.replace(r"[^a-z0-9_]+", "_", regex=True).str.strip("_"))
    missing = set(FEATURES + ["target"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization")
    original = len(frame)
    frame = frame[FEATURES + ["target"]].drop_duplicates()
    duplicates = original - len(frame)
    frame["target"] = frame["target"].astype("string").str.strip().str.lower().replace("", pd.NA)
    missing_targets = int(frame["target"].isna().sum())
    frame = frame.dropna(subset=["target"])
    if not frame["target"].isin(CLASSES).all():
        raise ValueError("Targets must be class_0, class_1, or class_2")
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    medians = numeric.median()
    if medians.isna().any():
        invalid = medians.index[medians.isna()].tolist()
        raise ValueError(f"Cannot impute columns with no valid numeric values: {invalid}")
    frame[FEATURES] = numeric.fillna(medians)
    before = len(frame)
    frame = frame.drop_duplicates()
    duplicates += before - len(frame)
    q1, q3 = frame[FEATURES].quantile(0.25), frame[FEATURES].quantile(0.75)
    iqr = q3 - q1
    # Skip features whose zero IQR cannot provide a useful spread estimate.
    varying = iqr.index[iqr > 0]
    inside = ((frame[varying] >= q1[varying] - 1.5 * iqr[varying])
              & (frame[varying] <= q3[varying] + 1.5 * iqr[varying])).all(axis=1)
    result = frame.loc[inside].reset_index(drop=True)
    print(f"Rows: raw={original}, duplicates_removed={duplicates}, "
          f"missing_targets_removed={missing_targets}, outliers_removed={int((~inside).sum())}, "
          f"cleaned={len(result)}")
    return result


def main():
    cleaned = clean_data(pd.read_csv(ROOT / "data/raw/wine.csv"))
    try:
        train, test = train_test_split(
            cleaned, test_size=0.2, random_state=42, stratify=cleaned["target"]
        )
    except ValueError as error:
        raise ValueError(f"Cannot create a stratified 80/20 split: {error}") from error
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "train.csv", index=False)
    test.to_csv(output / "test.csv", index=False)
    print(f"Saved {len(train)} training rows: {output / 'train.csv'}")
    print(f"Saved {len(test)} test rows: {output / 'test.csv'}")


if __name__ == "__main__":
    main()
