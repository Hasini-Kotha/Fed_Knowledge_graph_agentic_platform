"""
Ray-free manual implementation of the Federated Learning loop.
Fallback when Ray/flwr.simulation is unavailable (e.g. Windows + Python 3.13).

FIXES vs original:
  - client_fn receives string CIDs from a configurable client_ids list,
    not raw integers, matching the cid_to_client mapping in run_fl_simulation.
  - Final aggregated parameters are returned to the caller.
  - Client instances are created once per round phase (not re-loaded twice).
"""

import logging
import flwr as fl
from typing import Callable, Any, List, Optional

logger = logging.getLogger(__name__)


def run_manual_simulation(
    client_fn: Callable[[str], fl.client.NumPyClient],
    client_ids: List[str],
    num_rounds: int,
    strategy: Any,
) -> Optional[List]:
    """Execute FedAvg loop without Ray, synchronously on one thread.

    Args:
        client_fn: Callable that maps a string client-ID to a NumPyClient.
        client_ids: Ordered list of string client IDs, e.g. ["0", "1", "2"].
                    These must match the keys expected by client_fn.
        num_rounds: Total number of federated rounds.
        strategy: A Flower strategy (e.g. WeightedFedAvg).

    Returns:
        Final aggregated parameters as ndarrays, or None if aggregation failed.
    """
    logger.info(
        "Starting Ray-free manual FL simulation: %d clients, %d rounds.",
        len(client_ids), num_rounds,
    )

    parameters = strategy.initial_parameters
    final_parameters = None

    for current_round in range(1, num_rounds + 1):
        logger.info("--- [Manual Loop] Round %d / %d ---", current_round, num_rounds)

        # ---- Fit phase -------------------------------------------------------
        fit_results = []
        for cid in client_ids:
            client = client_fn(cid)
            params_nd, num_examples, metrics = client.fit(
                fl.common.parameters_to_ndarrays(parameters), {}
            )
            fit_results.append((
                None,  # ClientProxy placeholder
                fl.common.FitRes(
                    status=fl.common.Status(
                        code=fl.common.Code.OK, message="Success"
                    ),
                    parameters=fl.common.ndarrays_to_parameters(params_nd),
                    num_examples=num_examples,
                    metrics=metrics,
                ),
            ))

        aggregated_params, _ = strategy.aggregate_fit(current_round, fit_results, [])
        if aggregated_params is not None:
            parameters = aggregated_params
            final_parameters = fl.common.parameters_to_ndarrays(aggregated_params)

        # ---- Evaluate phase --------------------------------------------------
        eval_results = []
        for cid in client_ids:
            client = client_fn(cid)
            loss, num_examples, metrics = client.evaluate(
                fl.common.parameters_to_ndarrays(parameters), {}
            )
            eval_results.append((
                None,
                fl.common.EvaluateRes(
                    status=fl.common.Status(
                        code=fl.common.Code.OK, message="Success"
                    ),
                    loss=loss,
                    num_examples=num_examples,
                    metrics=metrics,
                ),
            ))

        strategy.aggregate_evaluate(current_round, eval_results, [])

    logger.info("Manual FL simulation completed (%d rounds).", num_rounds)
    return final_parameters
