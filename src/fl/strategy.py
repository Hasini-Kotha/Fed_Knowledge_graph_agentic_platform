"""
Custom FedAvg strategy for federated learning platform.
"""

import logging
import json
import pathlib
import torch
import numpy as np
import flwr as fl
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class WeightedFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy with weighted averaging by client sample count,
    per-round checkpointing, and metrics logging.
    """
    def __init__(self, artifacts_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.artifacts_dir = pathlib.Path(artifacts_dir)
        self.global_model_dir = self.artifacts_dir / "global_model"
        self.global_model_dir.mkdir(parents=True, exist_ok=True)
        self.round_metrics_history = []
        
    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], 
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        logger.info(f"[Round {server_round}] aggregate_fit called with {len(results)} results and {len(failures)} failures.")
        
        if len(failures) > 0:
            logger.warning(f"[Round {server_round}] Had {len(failures)} failures during fit.")
            
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id", "unknown")
            train_loss = fit_res.metrics.get("train_loss", "N/A")
            num_samples = fit_res.num_examples
            logger.info(f"[Round {server_round}] Result from {client_id}: {num_samples} samples, train_loss: {train_loss}")
            
        aggregated_parameters, metrics_dict = super().aggregate_fit(server_round, results, failures)
        
        # Save checkpoint if aggregation was successful
        if aggregated_parameters is not None:
            params_list = fl.common.parameters_to_ndarrays(aggregated_parameters)
            self.save_round_checkpoint(server_round, params_list, metrics_dict)
            
        return aggregated_parameters, metrics_dict

    def aggregate_evaluate(
        self, 
        server_round: int, 
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], 
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        
        logger.info(f"[Round {server_round}] aggregate_evaluate called with {len(results)} results.")
        
        if len(failures) > 0:
            logger.warning(f"[Round {server_round}] Had {len(failures)} failures during evaluate.")
            
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)
        
        if not results:
            return aggregated_loss, {}
            
        total_samples = sum(res.num_examples for _, res in results)
        
        aggregated_metrics = {
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
        for _, eval_res in results:
            weight = eval_res.num_examples / total_samples
            metrics = eval_res.metrics
            
            for k in aggregated_metrics.keys():
                # Get the metric or default to 0.0 if not present
                val = metrics.get(k)
                if val is not None:
                    # Flower metrics values can be int, float, str, bytes, bool
                    try:
                        aggregated_metrics[k] += float(val) * weight
                    except (ValueError, TypeError):
                        pass
        
        aggregated_metrics["participating_clients"] = len(results)
        
        logger.info(f"[Round {server_round}] Aggregated Metrics: {aggregated_metrics}")
        
        # Log to history
        self.round_metrics_history.append({
            "round": server_round,
            "loss": aggregated_loss,
            "metrics": aggregated_metrics
        })
        
        return aggregated_loss, aggregated_metrics

    def save_round_checkpoint(self, server_round: int, parameters: List[np.ndarray], metrics: Dict[str, Any]) -> None:
        """
        Saves the global model weights and metrics as a .pt checkpoint.
        """
        path = self.global_model_dir / f"round_{server_round:03d}_checkpoint.pt"
        
        checkpoint = {
            "round": server_round,
            "parameters": [p.tolist() for p in parameters],
            "metrics": metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"[Round {server_round}] Checkpoint saved to {path}")

    def save_metrics_history(self) -> None:
        """
        Saves self.round_metrics_history as a JSON file.
        """
        path = self.global_model_dir / "training_history.json"
        
        with open(path, "w") as f:
            json.dump(self.round_metrics_history, f, indent=2)
            
        logger.info(f"Training history saved to {path}")

def build_strategy(artifacts_dir: str, fl_config: Dict[str, Any]) -> WeightedFedAvg:
    """
    Creates and returns a WeightedFedAvg instance with settings from fl_config.
    """
    fraction_fit = fl_config.get("fraction_fit", 1.0)
    fraction_evaluate = fl_config.get("fraction_evaluate", 1.0)
    min_fit_clients = fl_config.get("min_fit_clients", 3)
    min_available_clients = fl_config.get("min_available_clients", 3)
    min_evaluate_clients = fl_config.get("min_evaluate_clients", 3)
    
    return WeightedFedAvg(
        artifacts_dir=artifacts_dir,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=None # We override aggregate_evaluate instead
    )
