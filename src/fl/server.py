"""Federated Learning Server — Flower Server Configuration and Factory.

Encapsulates server-side logic:
  - Configuration of Flower server settings.
  - Instantiation of the FedProx Strategy.
  - Helper to start standalone server or simulation.
"""

import logging
from typing import Dict, Any, Optional
import flwr as fl
from src.fl.strategy import FedProxStrategy

logger = logging.getLogger(__name__)


def create_server_config(num_rounds: int) -> fl.server.ServerConfig:
    """Create a Flower ServerConfig instance.

    Args:
        num_rounds: Number of federated learning rounds.
    """
    return fl.server.ServerConfig(num_rounds=num_rounds)


def build_fedprox_strategy(
    fl_config: Dict[str, Any],
    initial_parameters: fl.common.Parameters,
    artifacts_dir: str = "artifacts/global_model"
) -> FedProxStrategy:
    """Build the FedProx strategy with hyperparameters from the configuration.

    Args:
        fl_config: FL configuration dictionary containing strategy parameters.
        initial_parameters: Initial model weights serialized as Flower Parameters.
        artifacts_dir: Directory to save global model checkpoints.
    """
    fedprox_mu = fl_config.get("fedprox_mu", 0.01)
    num_clients = fl_config.get("min_clients", 3)
    fraction_fit = fl_config.get("fraction_fit", 1.0)
    fraction_evaluate = fl_config.get("fraction_evaluate", 1.0)

    strategy = FedProxStrategy(
        mu=fedprox_mu,
        artifacts_dir=artifacts_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )
    return strategy


def run_standalone_server(
    server_address: str,
    config: fl.server.ServerConfig,
    strategy: fl.server.strategy.Strategy
) -> None:
    """Start a standalone gRPC Flower server. Useful for real-world deployments.

    Args:
        server_address: Host and port to listen on (e.g. "[::]:8080").
        config: Server configuration instance.
        strategy: Strategy instance to use for aggregation.
    """
    logger.info(f"Starting standalone Flower server on {server_address}")
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy
    )
