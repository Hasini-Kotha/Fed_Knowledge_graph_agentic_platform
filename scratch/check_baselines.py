import sys
import torch
import numpy as np
from pathlib import Path

# Add current working directory to sys.path
sys.path.insert(0, str(Path.cwd()))

def check_baselines():
    for client in ["a", "b", "c"]:
        path = f"artifacts/local_models/client_{client}_baseline.pt"
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            # Check model state dict keys
            model_state = ckpt.get("model_state", {})
            first_layer_name = next(iter(model_state))
            w = model_state[first_layer_name].numpy()
            print(f"Client {client} baseline: first layer name={first_layer_name}, shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}")
        except Exception as e:
            print(f"Client {client} baseline failed to load: {e}")

if __name__ == "__main__":
    check_baselines()
