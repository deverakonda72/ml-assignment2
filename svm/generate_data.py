"""
generate_data.py  -  synthetic test-data generator
==================================================

Creates a synthetic multi-class classification dataset:
    * 12 numeric features   (feature_1 ... feature_12)
    * 1 target column       ("target", classes 0, 1, 2, 3)
    * 2000 rows by default  (the assignment needs at least 600)

The model file in this folder imports `generate_dataset()` from here during
its train step, so you do not have to run this file yourself. You can run it
on its own to look at the data:

    python generate_data.py                  # writes data/dataset.csv
    python generate_data.py --rows 1000 --classes 3

Every model folder has an identical copy of this file with the same seed (42),
so all five models learn from exactly the same data.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

HERE = os.path.dirname(os.path.abspath(__file__))       # this folder
DEFAULT_OUT = os.path.join(HERE, "data", "dataset.csv")

N_FEATURES = 12
MIN_ROWS = 600
TARGET_COLUMN = "target"


def generate_dataset(n_rows=2000, n_classes=4, random_state=42):
    """Return a pandas DataFrame with 12 feature columns and a 'target' column."""
    if n_rows < MIN_ROWS:
        raise ValueError(f"n_rows must be at least {MIN_ROWS}")

    X, y = make_classification(
        n_samples=n_rows,
        n_features=N_FEATURES,
        n_informative=10,        # 10 features really help to tell classes apart
        n_redundant=1,           # 1 feature is a mix of the informative ones
        n_repeated=0,            # (the last feature is pure noise)
        n_classes=n_classes,
        n_clusters_per_class=2,  # every class is made of 2 separate blobs
        class_sep=1.2,           # how far apart the classes are (bigger = easier)
        flip_y=0.03,             # 3% of labels are flipped on purpose (label noise)
        random_state=random_state,
    )

    columns = [f"feature_{i + 1}" for i in range(N_FEATURES)]
    df = pd.DataFrame(np.round(X, 6), columns=columns)
    df[TARGET_COLUMN] = y
    return df


def save_dataset(df, path=DEFAULT_OUT):
    """Save the DataFrame to CSV, creating the folder first if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset")
    parser.add_argument("--rows", type=int, default=2000, help="number of rows (>= 600)")
    parser.add_argument("--classes", type=int, default=4, help="number of classes")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to save the CSV")
    args = parser.parse_args()

    data = generate_dataset(args.rows, args.classes, args.seed)
    path = save_dataset(data, args.out)
    print(f"Saved {len(data)} rows x {N_FEATURES} features to {path}")
    print("Rows per class:")
    print(data[TARGET_COLUMN].value_counts().sort_index().to_string())
