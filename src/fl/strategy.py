"""FL Strategy — FedProx, WeightedFedAvg, and TrimmedMean aggregation.

Implements professional-grade federated aggregation strategies with:
- FedProx: Proximal term to penalize local model drift
- WeightedFedAvg: Sample-size weighted averaging with checkpointing
- TrimmedMean: Byzantine fault tolerance via outlier filtering
"""

import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import flwr as fl
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


class WeightedFedAvg(FedAvg):
    """Federated Averaging with weighted metrics and per-round checkpointing.
    
    Extends Flower's FedAvg to:
    - Compute weighted average of per-client metrics
    - Save model checkpoint after each round
    - Log training history for analysis
    
    Args:
        artifacts_dir: Directory to save checkpoints and metrics
        **kwargs: Passed to FedAvg
    """
    
    def __init__(self, artifacts_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self._cached_parameters = None
        self.metrics_history = []
        
        logger.info(f"WeightedFedAvg initialized, checkpoints: {self.artifacts_dir}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated[0] is not None:
            self._cached_parameters = aggregated[0]
        
        return aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}
        
        weighted_metrics = {}
        metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "loss"]
        
        for key in metric_keys:
            total_weight = 0
            weighted_sum = 0.0
            
            for _, evaluate_res in results:
                metrics = evaluate_res.metrics
                weight = evaluate_res.num_examples
                
                if key in metrics:
                    weighted_sum += metrics[key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_metrics[key] = weighted_sum / total_weight
        
        weighted_metrics["round"] = server_round
        
        if self._cached_parameters is not None:
            self.save_round_checkpoint(server_round, self._cached_parameters, weighted_metrics)
        
        self.metrics_history.append(weighted_metrics)
        self.save_metrics_history()
        
        logger.info(f"Round {server_round}: loss={weighted_metrics.get('loss', 'N/A'):.4f}, "
                   f"roc_auc={weighted_metrics.get('roc_auc', 'N/A'):.4f}, "
                   f"pr_auc={weighted_metrics.get('pr_auc', 'N/A'):.4f}")
        
        return weighted_metrics.get("loss"), weighted_metrics
    
    def save_round_checkpoint(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        metrics: Dict[str, Any]
    ):
        checkpoint_path = self.artifacts_dir / f"round_{server_round:03d}_checkpoint.pt"
        
        weights = [torch.tensor(p) for p in parameters.tensors]
        
        checkpoint = {
            "round": server_round,
            "weights": weights,
            "metrics": metrics,
        }
        
        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_metrics_history(self):
        history_path = self.artifacts_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class FedProxStrategy(fl.server.strategy.FedAvg):
    """FedProx strategy with proximal term coefficient.
    
    The proximal term (mu * ||w - w_global||^2) penalizes local model drift,
    improving stability across heterogeneous (Non-IID) datasets.
    """
    
    def __init__(
        self,
        artifacts_dir: str,
        mu: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mu = mu
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._cached_parameters = None
        self.metrics_history = []
        
        logger.info(f"FedProx initialized: mu={mu}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated[0] is not None:
            self._cached_parameters = aggregated[0]
        
        return aggregated
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}
        
        weighted_metrics = {}
        metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "loss"]
        
        for key in metric_keys:
            total_weight = 0
            weighted_sum = 0.0
            
            for _, evaluate_res in results:
                metrics = evaluate_res.metrics
                weight = evaluate_res.num_examples
                
                if key in metrics:
                    weighted_sum += metrics[key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_metrics[key] = weighted_sum / total_weight
        
        weighted_metrics["round"] = server_round
        weighted_metrics["mu"] = self.mu
        
        if self._cached_parameters is not None:
            self.save_round_checkpoint(server_round, self._cached_parameters, weighted_metrics)
        
        self.metrics_history.append(weighted_metrics)
        self.save_metrics_history()
        
        logger.info(f"Round {server_round} (FedProx mu={self.mu}): "
                   f"loss={weighted_metrics.get('loss', 'N/A'):.4f}, "
                   f"roc_auc={weighted_metrics.get('roc_auc', 'N/A'):.4f}")
        
        return weighted_metrics.get("loss"), weighted_metrics
    
    def save_round_checkpoint(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        metrics: Dict[str, Any]
    ):
        checkpoint_path = self.artifacts_dir / f"round_{server_round:03d}_checkpoint.pt"
        
        weights = [torch.tensor(p) for p in parameters.tensors]
        
        checkpoint = {
            "round": server_round,
            "weights": weights,
            "metrics": metrics,
            "strategy": "fedprox",
            "mu": self.mu,
        }
        
        torch.save(checkpoint, str(checkpoint_path))
    
    def save_metrics_history(self):
        history_path = self.artifacts_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class TrimmedMeanStrategy(fl.server.strategy.FedAvg):
    """Trimmed Mean aggregation for Byzantine fault tolerance.
    
    Trims the top and bottom beta fraction of weight updates before averaging,
    filtering out malicious or poor-quality updates.
    
    Args:
        artifacts_dir: Checkpoint directory
        beta: Fraction to trim from each end (default 0.1 = trim 10%)
        **kwargs: Passed to FedAvg
    """
    
    def __init__(
        self,
        artifacts_dir: str,
        beta: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._cached_parameters = None
        self.metrics_history = []
        
        logger.info(f"TrimmedMean initialized: beta={beta}")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        if len(results) < 3:
            return super().aggregate_fit(server_round, results, failures)
        
        all_weights = []
        for _, fit_res in results:
            all_weights.append([np.array(p) for p in fit_res.parameters.tensors])
        
        n_clients = len(all_weights)
        n_trim = max(1, int(n_clients * self.beta))
        
        trimmed_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = np.stack([w[layer_idx] for w in all_weights])
            sorted_weights = np.sort(layer_weights, axis=0)
            
            if n_trim > 0:
                trimmed = sorted_weights[n_trim:n_clients-n_trim]
            else:
                trimmed = sorted_weights
            
            trimmed_mean = np.mean(trimmed, axis=0)
            trimmed_weights.append(trimmed_mean)
        
        aggregated_params = fl.common.ndarrays_to_parameters(trimmed_weights)
        
        self._cached_parameters = aggregated_params
        
        return aggregated_params, {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if not results:
            return None, {}
        
        weighted_metrics = {}
        metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "loss"]
        
        for key in metric_keys:
            total_weight = 0
            weighted_sum = 0.0
            
            for _, evaluate_res in results:
                metrics = evaluate_res.metrics
                weight = evaluate_res.num_examples
                
                if key in metrics:
                    weighted_sum += metrics[key] * weight
                    total_weight += weight
            
            if total_weight > 0:
                weighted_metrics[key] = weighted_sum / total_weight
        
        weighted_metrics["round"] = server_round
        weighted_metrics["beta"] = self.beta
        
        if self._cached_parameters is not None:
            self.save_round_checkpoint(server_round, self._cached_parameters, weighted_metrics)
        
        self.metrics_history.append(weighted_metrics)
        self.save_metrics_history()
        
        logger.info(f"Round {server_round} (TrimmedMean beta={self.beta}): "
                   f"roc_auc={weighted_metrics.get('roc_auc', 'N/A'):.4f}")
        
        return weighted_metrics.get("loss"), weighted_metrics
    
    def save_round_checkpoint(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        metrics: Dict[str, Any]
    ):
        checkpoint_path = self.artifacts_dir / f"round_{server_round:03d}_checkpoint.pt"
        
        weights = [torch.tensor(p) for p in parameters.tensors]
        
        checkpoint = {
            "round": server_round,
            "weights": weights,
            "metrics": metrics,
            "strategy": "trimmed_mean",
            "beta": self.beta,
        }
        
        torch.save(checkpoint, str(checkpoint_path))
    
    def save_metrics_history(self):
        history_path = self.artifacts_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


def build_strategy(
    artifacts_dir: str,
    fl_config: Dict[str, Any],
    initial_parameters: fl.common.Parameters,
    strategy_type: str = "fedprox"
) -> fl.server.strategy.Strategy:
    """Factory to create the appropriate FL strategy.
    
    Args:
        artifacts_dir: Checkpoint directory
        fl_config: FL configuration dict
        initial_parameters: Initial model weights
        strategy_type: 'fedavg', 'fedprox', or 'trimmed_mean'
        
    Returns:
        Flower strategy instance
    """
    common_args = {
        "fraction_fit": fl_config.get("fraction_fit", 1.0),
        "fraction_evaluate": fl_config.get("fraction_evaluate", 1.0),
        "min_fit_clients": fl_config.get("min_clients", 3),
        "min_evaluate_clients": fl_config.get("min_clients", 3),
        "min_available_clients": fl_config.get("min_clients", 3),
        "initial_parameters": initial_parameters,
        "accept_failures": False,
    }
    
    if strategy_type == "fedprox":
        return FedProxStrategy(
            artifacts_dir=artifacts_dir,
            mu=fl_config.get("mu", 0.01),
            **common_args
        )
    elif strategy_type == "trimmed_mean":
        return TrimmedMeanStrategy(
            artifacts_dir=artifacts_dir,
            beta=fl_config.get("beta", 0.1),
            **common_args
        )
    else:
        return WeightedFedAvg(artifacts_dir=artifacts_dir, **common_args)
