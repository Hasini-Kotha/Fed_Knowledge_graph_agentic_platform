
"""Global Model Evaluation."""

import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

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
    best_value = -float("inf")
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        value = ckpt.get("metrics", {}).get(metric, 0)
        if value > best_value:
            best_value = value
            best_ckpt = ckpt
    
    logger.info(f"Best checkpoint: round {best_ckpt['round']} ({metric}={best_value:.4f})")
    return best_ckpt


def evaluate_global_model(
    parameters,
    global_test_csv,
    preprocessor_path,
    model_config,
    eval_config
):
    import pandas as pd
    from src.data.preprocess import ClientPreprocessor
    from src.models.tab_transformer import create_model
    from src.models.train_engine import evaluate_model
    import torch
    
    test_df = pd.read_csv(global_test_csv)
    preprocessor = ClientPreprocessor.load(preprocessor_path)
    X_test, y_test = preprocessor.transform(test_df)
    
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        
    padding_mask = preprocessor.get_padding_mask() if hasattr(preprocessor, "get_padding_mask") else None
    if padding_mask is not None and not isinstance(padding_mask, torch.Tensor):
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = preprocessor.get_feature_dim()
    model = create_model(input_dim, model_config).to(device)
    model.set_parameters([torch.tensor(p) for p in parameters] if isinstance(parameters[0], np.ndarray) else parameters)
    
    metrics = evaluate_model(model, X_test, y_test, device, eval_config.get("batch_size", 512), padding_mask)
    return metrics


def generate_evaluation_report(
    round_number: int,
    round_metrics: Dict[str, Any],
    global_metrics: Dict[str, Any],
    output_path: str = "artifacts/reports"
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "best_round": round_number,
        "fl_validation": round_metrics,
        "global_holdout": global_metrics,
    }
    
    json_path = Path(output_path).parent / "global_evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    txt_path = Path(output_path)
    with open(txt_path, "w") as f:
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

 
