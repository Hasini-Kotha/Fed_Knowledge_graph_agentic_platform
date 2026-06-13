"""Real FL Aggregation Script.

Takes locally trained client model checkpoints, performs FedProx/FedAvg aggregation, 
and exports the global model checkpoint.
"""

import sys
import yaml
import json
import logging
import pathlib
import torch
import numpy as np
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from src.models.Fed_model import create_model, LiteFraudNet
from src.models.train_engine import load_local_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def aggregate_parameters(client_params_list: List[List[torch.Tensor]], client_weights: List[float]) -> List[torch.Tensor]:
    """Perform weighted federated averaging of model parameters."""
    total_weight = sum(client_weights)
    aggregated_params = []
    
    # Initialize aggregated parameter tensors with zeros matching the shape of the first client's parameters
    for param_idx in range(len(client_params_list[0])):
        agg_param = torch.zeros_like(client_params_list[0][param_idx])
        for client_idx, params in enumerate(client_params_list):
            weight = client_weights[client_idx] / total_weight
            agg_param += params[param_idx] * weight
        aggregated_params.append(agg_param)
        
    return aggregated_params


def main():
    logger.info("=" * 60)
    logger.info("FEDERATED LEARNING: CENTRAL AGGREGATION (FedProx)")
    logger.info("=" * 60)

    # Configs
    model_config_path = pathlib.Path("configs/model_config.yaml")
    fl_config_path = pathlib.Path("configs/fl_config.yaml")

    if not model_config_path.exists() or not fl_config_path.exists():
        logger.error("Config files missing. Please check configs/")
        sys.exit(1)

    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    with open(fl_config_path, "r") as f:
        fl_config = yaml.safe_load(f)

    strategy = fl_config.get("strategy", "fedprox")
    mu = fl_config.get("fedprox_mu", 0.01) if strategy == "fedprox" else 0.0
    logger.info(f"Strategy: {strategy} | mu={mu}")

    artifacts_dir = pathlib.Path("artifacts")
    local_models_dir = artifacts_dir / "local_models"
    global_model_dir = artifacts_dir / "global_model"
    global_model_dir.mkdir(parents=True, exist_ok=True)

    clients = ["client_a", "client_b", "client_c"]
    client_params_list = []
    client_weights = []
    client_pr_aucs = []
    client_roc_aucs = []
    client_f1s = []

    # Number of training samples per client (since they are split equally in our data pipeline)
    # Alternatively we can extract it or default it.
    sample_sizes = {
        "client_a": 64556,
        "client_b": 64556,
        "client_c": 64556
    }

    logger.info("--- Loading client model checkpoints ---")
    for client in clients:
        ckpt_path = local_models_dir / f"{client}_baseline.pt"
        if not ckpt_path.exists():
            logger.error(
                f"Checkpoint for {client} not found at {ckpt_path}. "
                "Please run 'python src/main/run_single_baseline.py --client <name>' for all clients first."
            )
            sys.exit(1)

        checkpoint = load_local_checkpoint(str(ckpt_path), device="cpu")

        input_dim = checkpoint.get("input_dim", 64)
        model = create_model(input_dim=input_dim, config=model_config)
        model.load_state_dict(checkpoint["model_state"])

        client_params_list.append(model.get_parameters())

        # Use n_train stored in metrics (written by run_single_baseline) for proper weighting.
        # Falls back to equal weight (1.0) if not present.
        metrics = checkpoint.get("metrics", {})
        n_train = float(metrics.get("n_train", 1.0))
        client_weights.append(n_train if n_train > 0 else 1.0)

        client_pr_aucs.append(metrics.get("pr_auc", 0.0))
        client_roc_aucs.append(metrics.get("roc_auc", 0.0))
        client_f1s.append(metrics.get("f1", 0.0))

        logger.info(
            f"  {client}: PR-AUC={metrics.get('pr_auc', 0.0):.4f} | "
            f"ROC-AUC={metrics.get('roc_auc', 0.0):.4f} | "
            f"F1={metrics.get('f1', 0.0):.4f} | "
            f"n_train={n_train:.0f}"
        )

    logger.info("--- Aggregating parameters via FedProx (weighted average) ---")
    aggregated_params = aggregate_parameters(client_params_list, client_weights)

    # Determine next round number
    existing = sorted(global_model_dir.glob("round_*_checkpoint.pt"))
    next_round = len(existing) + 1

    avg_metrics = {
        "pr_auc": float(np.mean(client_pr_aucs)),
        "roc_auc": float(np.mean(client_roc_aucs)),
        "f1": float(np.mean(client_f1s)),
        "participating_clients": len(clients),
        "strategy": strategy,
        "fedprox_mu": mu,
    }

    global_checkpoint = {
        "round": next_round,
        "parameters": [p.tolist() for p in aggregated_params],
        "metrics": avg_metrics,
    }

    global_ckpt_path = global_model_dir / f"round_{next_round:03d}_checkpoint.pt"
    torch.save(global_checkpoint, global_ckpt_path)

    logger.info(f"Global checkpoint saved → {global_ckpt_path}")
    logger.info(
        f"Aggregated metrics: PR-AUC={avg_metrics['pr_auc']:.4f} | "
        f"ROC-AUC={avg_metrics['roc_auc']:.4f} | "
        f"F1={avg_metrics['f1']:.4f}"
    )
    logger.info("Run 'python src/main/run_global_eval.py' to evaluate the global model.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
