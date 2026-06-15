import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

def inspect_checkpoints():
    dir_path = Path("artifacts/global_model")
    if not dir_path.exists():
        print("global_model dir does not exist")
        return
        
    for p in sorted(dir_path.glob("*.pt")):
        mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"File: {p.name} | Modified: {mtime} | Size: {p.stat().st_size} bytes")
        try:
            ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
            weights = ckpt.get("weights", ckpt.get("parameters", []))
            if weights:
                first = np.array(weights[0])
                print(f"  First weight shape: {first.shape}, mean: {first.mean():.6f}, std: {first.std():.6f}, max_abs: {np.max(np.abs(first)):.6f}")
            else:
                # check if state_dict keys exist
                model_state = ckpt.get("model_state", None)
                if model_state:
                    first_layer_name = next(iter(model_state))
                    w = model_state[first_layer_name].numpy()
                    print(f"  State dict first layer: {first_layer_name}, shape: {w.shape}, mean: {w.mean():.6f}, std: {w.std():.6f}")
                else:
                    print("  No weights/parameters/model_state keys found in checkpoint.")
        except Exception as e:
            print(f"  Failed to read: {e}")

if __name__ == "__main__":
    inspect_checkpoints()
