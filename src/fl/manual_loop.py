"""Manual FL Loop — Ray-free fallback for Windows.

Synchronous FedAvg/FedProx loop that runs without Ray backend.
For each round:
  1. Train each client locally with current global weights
  2. Aggregate weights (weighted by client sample size)
  3. Evaluate all clients on the new global model
  4. Aggregate metrics and save checkpoint
"""

import logging
import hashlib
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def run_manual_simulation(
    client_ids: List[str],
    data_dir: str,
    mapping_path: str,
    vectorizer_path: str,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    fl_config: Dict[str, Any],
    privacy_config: Dict[str, Any],
    artifacts_dir: str = "artifacts/global_model",
    strategy_type: str = "fedprox",
):
    """Run a synchronous FL simulation loop without Ray.

    Args:
        client_ids: List of client identifiers
        data_dir: Directory containing client CSV files
        mapping_path: Path to mapping.json
        vectorizer_path: Path to fitted vectorizer pickle
        model_config: Model configuration
        train_config: Training configuration
        fl_config: FL configuration (num_rounds, strategy, mu, beta, etc.)
        privacy_config: Differential privacy configuration
        artifacts_dir: Directory to save checkpoints
        strategy_type: 'fedprox', 'fedavg', or 'trimmed_mean'
    """
    import pandas as pd
    from src.data.preprocess import ClientPreprocessor
    from src.models.tab_transformer import create_model
    from src.models.train_engine import train_one_round, evaluate_client
    from src.fl.secure_update import protect_update
    from src.fl.dp_accountant import RDPAccountant

    logger.info("Manual FL simulation: %d rounds, strategy=%s, %d clients",
               fl_config.get('num_rounds', 10), strategy_type, len(client_ids))

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    client_data = {}
    for cid in client_ids:
        train_df = pd.read_csv(f"{data_dir}/{cid}_train.csv")
        val_df = pd.read_csv(f"{data_dir}/{cid}_val.csv")

        # Load the preprocessor specific to this client
        prep_path = Path(vectorizer_path).parent / f"{cid}_preprocessor.pkl"
        preprocessor = ClientPreprocessor.load(str(prep_path))

        X_train, y_train = preprocessor.transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

        padding_mask = preprocessor.get_padding_mask() if hasattr(preprocessor, "get_padding_mask") else None
        if padding_mask is not None and not isinstance(padding_mask, torch.Tensor):
            padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
            
        if padding_mask is None:
            padding_mask = torch.ones(preprocessor.get_feature_dim(), dtype=torch.bool)

        client_data[cid] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'padding_mask': padding_mask,
            'n_train': len(X_train),
        }
        logger.info("%s: train=%s, val=%s, active_features=%d/%d",
                   cid, X_train.shape, X_val.shape,
                   padding_mask.sum().item(), preprocessor.get_feature_dim())

    model_type = model_config.get("model_type", "mlp")
    input_dim = preprocessor.get_feature_dim()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = create_model(input_dim, model_config, model_type)
    global_params = [p.clone().detach() for p in global_model.parameters()]

    mu = fl_config.get("fedprox_mu", 0.01) if strategy_type == "fedprox" else 0.0
    beta = fl_config.get("beta", 0.1) if strategy_type == "trimmed_mean" else 0.0

    metrics_history = []
    round_num = 0

    dp_enabled = privacy_config.get("enabled", False)
    noise_mult = privacy_config.get("noise_multiplier", 0.0)
    total_clients = len(client_ids)
    sample_rate = 1.0 if total_clients <= 3 else min(3, total_clients) / total_clients

    accountant = RDPAccountant(
        noise_multiplier=noise_mult if dp_enabled else 0.0,
        sample_rate=sample_rate,
        delta=privacy_config.get("delta", 1e-5),
    )

    for round_num in range(1, fl_config.get("num_rounds", 10) + 1):
        logger.info("--- Round %d/%d ---", round_num, fl_config.get('num_rounds', 10))

        fit_results = []
        for cid in client_ids:
            local_model = create_model(input_dim, model_config, model_type).to(device)
            local_model.set_parameters(global_params)

            tc = {
                "epochs": train_config.get("epochs", 3),
                "batch_size": train_config.get("batch_size", 256),
                "lr": train_config.get("lr", 0.001),
                "mu": mu,
                "round": round_num,
                "optimizer": "adamw",
            }

            updated_params, metrics = train_one_round(
                local_model,
                client_data[cid]['X_train'],
                client_data[cid]['y_train'],
                tc,
                device,
                padding_mask=client_data[cid]['padding_mask'],
                X_val=client_data[cid]['X_val'],
                y_val=client_data[cid]['y_val'],
            )

            if dp_enabled:
                ref_params = [p.cpu().numpy() for p in global_params]
                updated_np = [p.cpu().numpy() for p in updated_params]
                protected, audit = protect_update(updated_np, ref_params, privacy_config)
                updated_params = [torch.tensor(p) for p in protected]
                metrics["audit"] = audit

            fit_results.append((cid, updated_params, client_data[cid]['n_train'], metrics))
            logger.info("  %s: train_loss=%.4f", cid, metrics.get('train_loss', 0))

        if strategy_type == "trimmed_mean" and len(fit_results) >= 3:
            aggregated = _trimmed_mean_aggregate(fit_results, beta=beta)
        else:
            aggregated = _weighted_aggregate(fit_results)

        global_params = aggregated
        global_model.set_parameters(global_params)

        if dp_enabled:
            accountant.step()

        eval_results = []
        for cid in client_ids:
            eval_model = create_model(input_dim, model_config, model_type).to(device)
            eval_model.set_parameters(global_params)

            ec = {"batch_size": 512}
            em = evaluate_client(
                eval_model,
                client_data[cid]['X_val'],
                client_data[cid]['y_val'],
                ec,
                device,
                padding_mask=client_data[cid]['padding_mask']
            )
            eval_results.append((cid, em, client_data[cid]['n_train']))

        round_metrics = _aggregate_metrics(eval_results, round_num)
        round_metrics["strategy"] = strategy_type
        if mu > 0:
            round_metrics["mu"] = mu
        if beta > 0:
            round_metrics["beta"] = beta

        if dp_enabled:
            dp_budget = accountant.get_privacy_budget()
            round_metrics["dp_epsilon"] = dp_budget.epsilon
            round_metrics["dp_delta"] = dp_budget.delta
            round_metrics["dp_noise_multiplier"] = dp_budget.noise_multiplier

        _save_checkpoint(round_num, global_params, round_metrics, artifacts_dir)
        metrics_history.append(round_metrics)

        dp_info = ""
        if dp_enabled:
            dp_info = f", eps={accountant.get_epsilon():.4f}"
        logger.info("  Global: roc_auc=%.4f, pr_auc=%.4f, f1=%.4f%s",
                   round_metrics['roc_auc'], round_metrics['pr_auc'],
                   round_metrics['f1'], dp_info)

    if dp_enabled:
        dp_report = accountant.report()
        logger.info(dp_report)
        _save_dp_report(accountant, artifacts_dir)

    _save_metrics_history(metrics_history, artifacts_dir)
    logger.info("FL simulation complete: %d rounds", round_num)

    return metrics_history


def _weighted_aggregate(fit_results):
    """Weighted average of client parameters (FedAvg/FedProx)."""
    total_samples = sum(r[2] for r in fit_results)

    aggregated = None
    for _, params, n_samples, _ in fit_results:
        weight = n_samples / total_samples
        if aggregated is None:
            aggregated = [p * weight for p in params]
        else:
            aggregated = [a + p * weight for a, p in zip(aggregated, params)]

    return aggregated


def _trimmed_mean_aggregate(fit_results, beta=0.1):
    """Trimmed Mean aggregation for Byzantine fault tolerance."""
    n_clients = len(fit_results)
    n_trim = max(1, int(n_clients * beta))

    all_weights = []
    for _, params, _, _ in fit_results:
        all_weights.append([p.cpu().numpy() for p in params])

    trimmed_weights = []
    for layer_idx in range(len(all_weights[0])):
        layer_weights = np.stack([w[layer_idx] for w in all_weights])
        sorted_weights = np.sort(layer_weights, axis=0)

        if n_trim > 0:
            trimmed = sorted_weights[n_trim:n_clients-n_trim]
        else:
            trimmed = sorted_weights

        trimmed_mean = np.mean(trimmed, axis=0)
        trimmed_weights.append(torch.tensor(trimmed_mean))

    return trimmed_weights


def _aggregate_metrics(eval_results, round_num):
    """Weighted average of client evaluation metrics."""
    total_samples = sum(r[2] for r in eval_results)

    metrics = {"round": round_num}
    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]

    for key in metric_keys:
        weighted_sum = sum(r[1].get(key, 0) * r[2] for r in eval_results)
        metrics[key] = weighted_sum / total_samples

    return metrics


def _save_checkpoint(round_num, params, metrics, artifacts_dir):
    """Save round checkpoint with SHA-256 checksum."""
    checkpoint_path = Path(artifacts_dir) / f"round_{round_num:03d}_checkpoint.pt"

    weights = [p.clone().detach() for p in params]

    checksum = hashlib.sha256()
    for p in weights:
        checksum.update(p.cpu().numpy().tobytes())

    checkpoint = {
        "round": round_num,
        "weights": weights,
        "metrics": metrics,
        "checksum": checksum.hexdigest(),
    }

    torch.save(checkpoint, str(checkpoint_path))


def _save_metrics_history(history, artifacts_dir):
    """Save metrics history."""
    history_path = Path(artifacts_dir) / "training_history.json"

    serializable = []
    for h in history:
        serializable.append({k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in h.items()})

    with open(history_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def _save_dp_report(accountant, artifacts_dir):
    """Save differential privacy budget report."""
    from src.fl.dp_accountant import RDPAccountant
    report_path = Path(artifacts_dir) / "dp_budget_report.json"

    budget = accountant.get_privacy_budget()
    report = {
        "mechanism": "gaussian_subsampled",
        "noise_multiplier": budget.noise_multiplier,
        "sample_rate": budget.sample_rate,
        "total_rounds": budget.rounds,
        "delta": budget.delta,
        "epsilon": budget.epsilon,
        "history": accountant.get_history(),
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info("DP budget report saved to %s", report_path)
