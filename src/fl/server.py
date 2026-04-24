"""
Server setup utilities for federated simulation.
"""

import flwr as fl
import torch
from typing import Dict, Any

from src.models.mlp import create_model

def create_server_config(fl_config: Dict[str, Any]) -> fl.server.ServerConfig:
    """
    Creates a Flower ServerConfig based on the provided fl_config dict.
    """
    num_rounds = fl_config.get("num_rounds", 10)
    return fl.server.ServerConfig(num_rounds=num_rounds)

def get_initial_parameters(model_config: Dict[str, Any], input_dim: int) -> fl.common.Parameters:
    """
    Creates initial model parameters for the server.
    Ensures all clients start from the exact same random initialization.
    """
    model = create_model(input_dim=input_dim, config=model_config)
    initial_weights = model.get_parameters()
    return fl.common.ndarrays_to_parameters(initial_weights)

def run_federated_server(strategy, server_config, initial_parameters) -> fl.server.History:
    """
    Starts the Flower server.
    Note: In simulation mode (flwr.simulation), this function is NOT used directly.
    See run_fl_simulation.py for the simulation runner.
    """
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )
    return history
