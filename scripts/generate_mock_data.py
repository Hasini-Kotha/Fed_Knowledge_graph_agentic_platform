"""
Generate mock data for testing the FL pipeline without the real dataset.

Run from the project root:
    python scripts/generate_mock_data.py

This creates:
  - data/splits/client_{a,b,c}_train.csv
  - data/splits/client_{a,b,c}_val.csv
  - data/splits/global_test.csv

NOTE: The original setup.py was misnamed and also overwrote configs.
      This script ONLY generates data splits; configs are managed via configs/*.yaml.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# Directories
dirs = [
    "src/data", "src/models", "src/evaluation", "src/fl", "src/main",
    "configs", "data/splits",
    "artifacts/preprocessors", "artifacts/local_models",
    "artifacts/global_model", "artifacts/reports",
]
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# Mock Data — matches Kaggle MLG-ULB schema (Time, V1-V28, Amount, Class)
np.random.seed(42)
columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def generate_mock_data(n_samples: int = 1000, n_frauds: int = 10) -> pd.DataFrame:
    df = pd.DataFrame(np.random.randn(n_samples, len(columns)), columns=columns)
    df["Time"] = np.random.randint(0, 100_000, n_samples)
    df["Amount"] = np.abs(np.random.randn(n_samples)) * 100
    df["Class"] = 0
    fraud_idx = np.random.choice(n_samples, n_frauds, replace=False)
    df.loc[fraud_idx, "Class"] = 1
    # Add learnable signal
    df.loc[fraud_idx, "V1"] += 5
    df.loc[fraud_idx, "V2"] -= 5
    return df


clients = ["client_a", "client_b", "client_c"]
for client in clients:
    generate_mock_data(n_samples=2000, n_frauds=20).to_csv(
        f"data/splits/{client}_train.csv", index=False
    )
    generate_mock_data(n_samples=500, n_frauds=5).to_csv(
        f"data/splits/{client}_val.csv", index=False
    )

generate_mock_data(n_samples=1500, n_frauds=15).to_csv(
    "data/splits/global_test.csv", index=False
)

print("Mock data generated successfully in data/splits/")
print("Next: run 'python src/main/run_single_baseline.py --client client_a' for each client.")
