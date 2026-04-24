import os
import pandas as pd
import numpy as np
import yaml

# Directories
dirs = [
    "src/data", "src/models", "src/evaluation", "src/fl", "src/main",
    "configs", "data/splits", 
    "artifacts/preprocessors", "artifacts/local_models", "artifacts/global_model", "artifacts/reports"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Mock Data Generation
# Columns: Time, V1-V28, Amount, Class
np.random.seed(42)
columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

def generate_mock_data(n_samples=1000, n_frauds=10):
    df = pd.DataFrame(np.random.randn(n_samples, len(columns)), columns=columns)
    df["Time"] = np.random.randint(0, 100000, n_samples)
    df["Amount"] = np.abs(np.random.randn(n_samples)) * 100
    df["Class"] = 0
    fraud_indices = np.random.choice(n_samples, n_frauds, replace=False)
    df.loc[fraud_indices, "Class"] = 1
    # Add some correlation for the models to learn
    df.loc[fraud_indices, "V1"] += 5
    df.loc[fraud_indices, "V2"] -= 5
    return df

clients = ["client_a", "client_b", "client_c"]
for client in clients:
    train_df = generate_mock_data(n_samples=2000, n_frauds=20)
    val_df = generate_mock_data(n_samples=500, n_frauds=5)
    train_df.to_csv(f"data/splits/{client}_train.csv", index=False)
    val_df.to_csv(f"data/splits/{client}_val.csv", index=False)

global_test_df = generate_mock_data(n_samples=1500, n_frauds=15)
global_test_df.to_csv("data/splits/global_test.csv", index=False)

# Configs
data_config = {
    "numeric_cols": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
    "label_col": "Class"
}
with open("configs/data_config.yaml", "w") as f:
    yaml.dump(data_config, f)

model_config = {
    "numeric_cols": ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"],
    "hidden_dims": [64, 32],
    "dropout_rate": 0.2,
    "scaler_type": "robust",
    "train": {
        "epochs": 5,
        "batch_size": 256,
        "lr": 0.001,
        "seed": 42
    },
    "eval": {
        "batch_size": 512,
        "threshold": 0.5
    }
}
with open("configs/model_config.yaml", "w") as f:
    yaml.dump(model_config, f)

fl_config = {
    "num_rounds": 10,
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0,
    "min_fit_clients": 3,
    "min_evaluate_clients": 3,
    "min_available_clients": 3,
    "local_epochs": 3,
    "batch_size": 256,
    "lr": 0.001,
    "secure_update": {
        "max_norm": 1.0,
        "noise_multiplier": 0.0,
        "validate": True
    }
}
with open("configs/fl_config.yaml", "w") as f:
    yaml.dump(fl_config, f)

print("Setup completed.")
