import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Add current working directory to sys.path
sys.path.insert(0, str(Path.cwd()))

import torch
import numpy as np

from src.gateway.routes import _get_initial_random_weights

def test():
    weights = _get_initial_random_weights()
    first = np.array(weights[0])
    print(f"Random weights generated: shape={first.shape}, mean={first.mean():.6f}, std={first.std():.6f}")

if __name__ == "__main__":
    test()
