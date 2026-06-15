import sys
import torch
import numpy as np
from pathlib import Path

# Add current working directory to sys.path
sys.path.insert(0, str(Path.cwd()))

from src.gateway.local_fl_runner import load_client_data
from src.models.Fed_model import create_model
from src.models.train_engine import train_one_round

def test_local_training():
    model_config = {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "dropout": 0.20,
    }
    
    print("Loading data for client_a...")
    X_train, y_train, X_val, y_val, input_dim = load_client_data("client_a", model_config)
    print(f"X_train shape: {X_train.shape}, y_train positive class: {y_train.sum().item()}/{len(y_train)}")
    
    model = create_model(input_dim=input_dim, config=model_config)
    
    p = list(model.parameters())[0]
    print(f"Before training - First layer mean: {p.mean().item():.6f}, std: {p.std().item():.6f}")
    
    train_config = {
        "epochs": 3,
        "batch_size": 256,
        "lr": 0.01,
        "mu": 0.0,
        "optimizer": "adamw",
    }
    
    print("Running train_one_round...")
    params, metrics = train_one_round(
        model=model,
        X_train=X_train,
        y_train=y_train,
        train_config=train_config,
        X_val=X_val,
        y_val=y_val,
    )
    
    p_after = params[0]
    print(f"After training - First layer mean: {p_after.mean().item():.6f}, std: {p_after.std().item():.6f}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    test_local_training()
