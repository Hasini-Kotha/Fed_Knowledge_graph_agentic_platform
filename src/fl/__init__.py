"""FL module."""
from src.fl.client import FederatedClient, create_client_fn
from src.fl.strategy import WeightedFedAvg, FedProxStrategy, TrimmedMeanStrategy, build_strategy
from src.fl.manual_loop import run_manual_simulation
from src.fl.secure_update import protect_update