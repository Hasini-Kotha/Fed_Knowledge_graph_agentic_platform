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
