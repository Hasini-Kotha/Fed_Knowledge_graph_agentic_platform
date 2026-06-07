"""FedProx Strategy — Flower server-side aggregation.

The proximal term mu is applied CLIENT-SIDE during local training.
The server communicates mu to every client via configure_fit config dict.
Aggregation is standard weighted average (identical to FedAvg server-side).
"""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateRes, FitIns, FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

logger = logging.getLogger(__name__)


class FedProxStrategy(FedAvg):
    """Server-side FedProx aggregation strategy.

    Args:
        mu:            Proximal term coefficient forwarded to every client.
        artifacts_dir: Directory where per-round .pt checkpoints are saved.
        **kwargs:      Passed to FedAvg (fraction_fit, min_clients, etc.).
    """

    def __init__(
        self,
        mu: float = 0.01,
        artifacts_dir: str = "artifacts/global_model",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._cached_params: Optional[List[np.ndarray]] = None
        self.metrics_history: List[Dict[str, Any]] = []
        logger.info("FedProxStrategy  mu=%.4f  checkpoints=%s", mu, self.artifacts_dir)

    # ------------------------------------------------------------------
    # Inject mu + round number into every client's fit config
    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {"round": server_round, "mu": self.mu}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(c, fit_ins) for c in clients]

    # ------------------------------------------------------------------
    # Weighted average of client parameters
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(r.num_examples for _, r in results)
        first = parameters_to_ndarrays(results[0][1].parameters)
        aggregated = [np.zeros_like(a) for a in first]

        for _, fit_res in results:
            w = fit_res.num_examples / total_examples
            for i, arr in enumerate(parameters_to_ndarrays(fit_res.parameters)):
                aggregated[i] += w * arr

        self._cached_params = aggregated
        logger.info(
            "Round %d | aggregated %d clients | %d examples | mu=%.4f",
            server_round, len(results), total_examples, self.mu,
        )
        return ndarrays_to_parameters(aggregated), {
            "round": server_round,
            "total_examples": float(total_examples),
        }

    # ------------------------------------------------------------------
    # Weighted evaluation metrics + checkpoint save
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total = sum(r.num_examples for _, r in results)
        wm: Dict[str, float] = {"round": float(server_round)}

        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"):
            s, n = 0.0, 0
            for _, er in results:
                if key in er.metrics:
                    s += er.metrics[key] * er.num_examples
                    n += er.num_examples
            if n:
                wm[key] = s / n

        if self._cached_params is not None:
            self._save_checkpoint(server_round, self._cached_params, wm)

        self.metrics_history.append(wm)
        self._save_metrics_history()

        logger.info(
            "Round %d eval | PR-AUC=%.4f | ROC-AUC=%.4f",
            server_round,
            wm.get("pr_auc", 0.0),
            wm.get("roc_auc", 0.0),
        )
        return 1.0 - wm.get("pr_auc", 0.0), wm

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        round_num: int,
        parameters: List[np.ndarray],
        metrics: Dict[str, Any],
    ) -> None:
        ckpt = {
            "round": round_num,
            "weights": parameters,       # List[np.ndarray] — loaded by run_global_eval
            "metrics": metrics,
            "strategy": "fedprox",
            "mu": self.mu,
        }
        path = self.artifacts_dir / f"round_{round_num:03d}_checkpoint.pt"
        torch.save(ckpt, path)
        logger.info("Checkpoint saved → %s", path)

    def _save_metrics_history(self) -> None:
        path = self.artifacts_dir / "metrics_history.json"
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
