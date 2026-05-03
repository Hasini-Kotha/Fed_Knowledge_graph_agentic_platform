"""Manual FL Loop — Ray-free fallback for Windows."""

import logging
import numpy as np
import torch
from typing import List, Dict, Any, Callable
from pathlib import Path

import flwr as fl

logger = logging.getLogger(__name__)


def run_manual_simulation(
    client_fn: Callable,
    client_ids: List[str],
    num_rounds: int,
    strategy,
    data_dir: str = "data/splits",
    mapping_path: str = "configs/mapping.json",
    model_config: Dict[str, Any] = None,
    train_config: Dict[str, Any] = None,
    privacy_config: Dict[str, Any] = None
):
    """Run a synchronous FedAvg/FedProx loop without Ray.
    
    For each round:
    1. Train each client locally with current global weights
    2. Aggregate weights
    3. Evaluate all clients
    4. Aggregate metrics
    
    Args:
        client_fn: Factory function to create clients
        client_ids: List of client identifiers
        num_rounds: Number of FL rounds
        strategy: FL strategy (FedProx, FedAvg, etc.)
    """
    logger.info(f"Manual FL loop: {num_rounds} rounds, {len(client_ids)} clients")
    
    from src.core.metadata_engine import MetadataMapper
    from src.core.vectorizer import DynamicVectorizer
    from src.models import create_model
    from src.models.train_engine import train_one_round, evaluate_client
    
    mapper = MetadataMapper(mapping_path)
    
    vectorizer_path = Path("artifacts/global_vectorizer.pkl")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}. Run data pipeline first.")
    
    vectorizer = DynamicVectorizer.load(str(vectorizer_path))
    
    import pandas as pd
    import numpy as np
    
    client_data = {}
    for client_id in client_ids:
        train_df = pd.read_csv(f"{data_dir}/{client_id}_train.csv")
        val_df = pd.read_csv(f"{data_dir}/{client_id}_val.csv")
        
        X_train, y_train = vectorizer.fit_transform(train_df, mapper) if client_id == client_ids[0] else vectorizer.transform(train_df, mapper)
        X_val = vectorizer.transform(val_df, mapper)
        y_val = torch.tensor(val_df[mapper.get_target_column()].values.astype(np.float32))
        
        client_data[client_id] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val
        }
    
    model_type = model_config.get("model_type", "mlp") if model_config else "mlp"
    input_dim = vectorizer.get_feature_dim()
    
    global_model = create_model(input_dim, model_config or {}, model_type)
    global_params = [p.clone().detach() for p in global_model.parameters()]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = global_model.to(device)
    
    metrics_history = []
    
    for round_num in range(1, num_rounds + 1):
        logger.info(f"--- Round {round_num}/{num_rounds} ---")
        
        fit_results = []
        for client_id in client_ids:
            local_model = create_model(input_dim, model_config or {}, model_type).to(device)
            local_model.set_parameters(global_params)
            
            train_cfg = {
                "epochs": train_config.get("epochs", 3) if train_config else 3,
                "lr": train_config.get("lr", 0.001) if train_config else 0.001,
                "batch_size": train_config.get("batch_size", 256) if train_config else 256,
                "mu": train_config.get("mu", 0.01) if train_config else 0.01,
                "round": round_num,
            }
            
            updated_params, metrics = train_one_round(
                local_model,
                client_data[client_id]['X_train'],
                client_data[client_id]['y_train'],
                train_cfg,
                device
            )
            
            fit_results.append((client_id, updated_params, len(client_data[client_id]['X_train']), metrics))
        
        aggregated_params = _aggregate_weights(fit_results, strategy)
        global_params = aggregated_params
        global_model.set_parameters(global_params)
        
        eval_results = []
        for client_id in client_ids:
            eval_model = create_model(input_dim, model_config or {}, model_type).to(device)
            eval_model.set_parameters(global_params)
            
            eval_cfg = {"batch_size": 512}
            eval_metrics = evaluate_client(
                eval_model,
                client_data[client_id]['X_val'],
                client_data[client_id]['y_val'],
                eval_cfg,
                device
            )
            eval_results.append((client_id, eval_metrics, len(client_data[client_id]['X_val'])))
        
        round_metrics = _aggregate_metrics(eval_results, round_num)
        metrics_history.append(round_metrics)
        
        _save_checkpoint(round_num, global_params, round_metrics, "artifacts/global_model")
        
        logger.info(f"Round {round_metrics['round']}: "
                  f"loss={round_metrics.get('loss', 'N/A'):.4f}, "
                  f"roc_auc={round_metrics.get('roc_auc', 'N/A'):.4f}, "
                  f"pr_auc={round_metrics.get('pr_auc', 'N/A'):.4f}")
    
    _save_metrics_history(metrics_history, "artifacts/global_model")
    
    logger.info(f"FL simulation complete: {num_rounds} rounds")
    return metrics_history


def _aggregate_weights(fit_results, strategy):
    """Weighted average of client parameters."""
    total_samples = sum(r[2] for r in fit_results)
    
    aggregated = None
    for _, params, n_samples, _ in fit_results:
        weight = n_samples / total_samples
        if aggregated is None:
            aggregated = [p * weight for p in params]
        else:
            aggregated = [a + p * weight for a, p in zip(aggregated, params)]
    
    return aggregated


def _aggregate_metrics(eval_results, round_num):
    """Weighted average of client evaluation metrics."""
    total_samples = sum(r[2] for r in eval_results)
    
    metrics = {"round": round_num}
    metric_keys = ["loss", "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    
    for key in metric_keys:
        weighted_sum = 0.0
        for _, client_metrics, n_samples in eval_results:
            if key in client_metrics:
                weighted_sum += client_metrics[key] * n_samples
        metrics[key] = weighted_sum / total_samples
    
    return metrics


def _save_checkpoint(round_num, params, metrics, artifacts_dir):
    """Save round checkpoint."""
    import torch
    from pathlib import Path
    
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(artifacts_dir) / f"round_{round_num:03d}_checkpoint.pt"
    
    checkpoint = {
        "round": round_num,
        "weights": params,
        "metrics": metrics,
    }
    
    torch.save(checkpoint, str(checkpoint_path))


def _save_metrics_history(history, artifacts_dir):
    """Save metrics history."""
    import json
    from pathlib import Path
    
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    history_path = Path(artifacts_dir) / "training_history.json"
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)