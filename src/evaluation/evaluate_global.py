"""Global Model Evaluation."""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.models import create_model
from src.models.train_engine import predict_proba, evaluate_model
from src.evaluation.metrics import compute_optimal_threshold, print_metrics_report

logger = logging.getLogger(__name__)


def load_best_checkpoint(
    global_model_dir: str = "artifacts/global_model",
    metric: str = "pr_auc"
) -> Optional[Dict[str, Any]]:
    """Load the best checkpoint by a given metric."""
    checkpoint_dir = Path(global_model_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = list(checkpoint_dir.glob("round_*_checkpoint.pt"))
    if not checkpoints:
        logger.error("No checkpoints found")
        return None
    
    best_ckpt = None
    best_value = -float('inf')
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        value = ckpt.get("metrics", {}).get(metric, 0)
        if value > best_value:
            best_value = value
            best_ckpt = ckpt
    
    logger.info(f"Best checkpoint: round {best_ckpt['round']} ({metric}={best_value:.4f})")
    return best_ckpt


def evaluate_global_model(
    checkpoint: Dict[str, Any],
    global_test_csv: str,
    mapping_path: str,
    vectorizer_path: str,
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate global model on holdout test set."""
    import pandas as pd
    
    mapper = MetadataMapper(mapping_path)
    vectorizer = DynamicVectorizer.load(vectorizer_path)
    
    test_df = pd.read_csv(global_test_csv)
    test_result = vectorizer.transform(test_df, mapper)
    
    if isinstance(test_result, dict):
        X_test = test_result["data"]
        padding_mask = test_result.get("mask")
    else:
        X_test = test_result
        padding_mask = None
    
    y_test = test_df[mapper.get_target_column()].values.astype(np.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_test.shape[1]
    model_type = model_config.get("model_type", "mlp")
    
    model = create_model(input_dim, model_config, model_type).to(device)
    
    if "weights" in checkpoint:
        weights = checkpoint["weights"]
        if isinstance(weights, list) and len(weights) > 0:
            model.set_parameters(weights)
    
    eval_config = {"batch_size": model_config.get("eval_batch_size", 512)}
    metrics = evaluate_model(model, X_test, y_test, device, eval_config["batch_size"], padding_mask=padding_mask)
    
    probs = predict_proba(model, X_test, device, eval_config["batch_size"], padding_mask=padding_mask)
    optimal_threshold = compute_optimal_threshold(y_test, probs, metric="f1")
    metrics["optimal_threshold"] = optimal_threshold
    
    logger.info(f"Global model evaluation complete:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return metrics


def generate_evaluation_report(
    round_number: int,
    round_metrics: Dict[str, Any],
    global_metrics: Dict[str, Any],
    output_path: str = "artifacts/reports"
):
    """Generate comparison report."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    report = {
        "best_round": round_number,
        "fl_validation": round_metrics,
        "global_holdout": global_metrics,
    }
    
    json_path = Path(output_path) / "global_evaluation_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    txt_path = Path(output_path) / "global_evaluation_report.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Global Model Evaluation Report\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Best Round: {round_number}\n\n")
        f.write("FL Validation Metrics:\n")
        for k, v in round_metrics.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
        f.write("\nGlobal Holdout Metrics:\n")
        for k, v in global_metrics.items():
            if isinstance(v, (int, float)):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")
    
    logger.info(f"Reports saved to {output_path}")