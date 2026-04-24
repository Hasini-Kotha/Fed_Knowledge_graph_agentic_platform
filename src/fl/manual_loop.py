"""
Ray-free implementation of the Federated Learning loop.
Provides a fallback when Ray is unavailable (common on Windows + Python 3.13).
"""

import logging
import flwr as fl
from typing import Callable, Any

logger = logging.getLogger(__name__)

def run_manual_simulation(
    client_fn: Callable[[str], fl.client.NumPyClient],
    num_clients: int,
    num_rounds: int,
    strategy: Any,
):
    """
    Manually executes the FedAvg loop without Ray.
    This is synchronous and runs on a single thread.
    """
    logger.info("Starting Ray-free manual FL simulation...")
    
    # 1. Initial Parameters
    parameters = strategy.initial_parameters
    
    for current_round in range(1, num_rounds + 1):
        logger.info(f"--- [Manual Loop] Round {current_round}/{num_rounds} ---")
        
        # --- Fit Phase ---
        client_results_fit = []
        for cid in range(num_clients):
            client = client_fn(str(cid))
            params_nd, num_examples, metrics = client.fit(
                fl.common.parameters_to_ndarrays(parameters), 
                {}
            )
            
            # Mock the ClientProxy needed by strategy.aggregate_fit
            client_results_fit.append((
                None, 
                fl.common.FitRes(
                    status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                    parameters=fl.common.ndarrays_to_parameters(params_nd),
                    num_examples=num_examples,
                    metrics=metrics
                )
            ))
            
        # Aggregate Fit
        aggregated_params, _ = strategy.aggregate_fit(current_round, client_results_fit, [])
        if aggregated_params:
            parameters = aggregated_params
            
        # --- Evaluate Phase ---
        client_results_eval = []
        for cid in range(num_clients):
            client = client_fn(str(cid))
            loss, num_examples, metrics = client.evaluate(
                fl.common.parameters_to_ndarrays(parameters), 
                {}
            )
            
            client_results_eval.append((
                None,
                fl.common.EvaluateRes(
                    status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                    loss=loss,
                    num_examples=num_examples,
                    metrics=metrics
                )
            ))
            
        # Aggregate Evaluate
        strategy.aggregate_evaluate(current_round, client_results_eval, [])

    logger.info("Manual FL simulation completed.")
