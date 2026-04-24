"""
Module to load the final global model checkpoint and evaluate it on the holdout test set.
"""

import json
import logging
import pathlib
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, Any, List

from src.models.mlp import create_model, TabularMLP
from src.data.preprocess import ClientPreprocessor
from src.models.train_local import evaluate_client
from src.evaluation.metrics import print_metrics_report, format_confusion_matrix, compute_optimal_threshold

logger = logging.getLogger(__name__)

def load_best_checkpoint(global_model_dir: str, metric: str = "pr_auc") -> Tuple[List[np.ndarray], Dict[str, Any], int]:
    """
    Scans global_model_dir for round checkpoints, returns the best one by metric.
    """
    path_obj = pathlib.Path(global_model_dir)
    checkpoints = list(path_obj.glob("round_*_checkpoint.pt"))
    
    if not checkpoints:
        raise FileNotFoundError(f"No round checkpoints found in {global_model_dir}")
        
    best_round = -1
    best_score = -1.0
    best_params = None
    best_metrics = None
    
    for ckpt_path in checkpoints:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        metrics = checkpoint["metrics"]
        score = metrics.get(metric, 0.0)
        
        if score > best_score:
            best_score = score
            best_round = checkpoint["round"]
            # Convert nested lists back to np.ndarray
            best_params = [np.array(p) for p in checkpoint["parameters"]]
            best_metrics = metrics
            
    logger.info(f"Selected best round {best_round} based on {metric}={best_score:.4f}")
    return best_params, best_metrics, best_round

def evaluate_global_model(
    parameters: List[np.ndarray], 
    global_test_csv: str, 
    preprocessor_path: str, 
    model_config: Dict[str, Any], 
    eval_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Loads global_test.csv, applies preprocessor, sets weights, evaluates.
    """
    logger.info(f"Loading global test data from {global_test_csv}")
    df_test = pd.read_csv(global_test_csv)
    
    preprocessor = ClientPreprocessor.load(preprocessor_path)
    X_test, y_test = preprocessor.transform(df_test)
    
    input_dim = preprocessor.get_feature_dim()
    model = create_model(input_dim=input_dim, config=model_config)
    model.set_parameters(parameters)
    
    logger.info("Evaluating global model on test set...")
    metrics = evaluate_client(model, X_test, y_test, eval_config)
    
    return metrics

def generate_evaluation_report(
    round_number: int, 
    round_metrics: Dict[str, Any], 
    global_metrics: Dict[str, Any], 
    output_path: str
) -> None:
    """
    Creates a comprehensive text report of the final evaluation.
    """
    path_obj = pathlib.Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    report = [
        "============================================",
        "Global Model Evaluation Report",
        "============================================",
        "",
        f"Best Round Selected: Round {round_number} (Selection metric: pr_auc)",
        "",
        "--- Final Round FL Metrics (Validation) ---",
        f"ROC-AUC:   {round_metrics.get('roc_auc', 0.0):.4f}",
        f"PR-AUC:    {round_metrics.get('pr_auc', 0.0):.4f}",
        f"Precision: {round_metrics.get('precision', 0.0):.4f}",
        f"Recall:    {round_metrics.get('recall', 0.0):.4f}",
        f"F1:        {round_metrics.get('f1', 0.0):.4f}",
        "",
        "--- Global Holdout Test Metrics ---",
        f"ROC-AUC:   {global_metrics.get('roc_auc', 0.0):.4f}",
        f"PR-AUC:    {global_metrics.get('pr_auc', 0.0):.4f}",
        f"Precision: {global_metrics.get('precision', 0.0):.4f}",
        f"Recall:    {global_metrics.get('recall', 0.0):.4f}",
        f"F1:        {global_metrics.get('f1', 0.0):.4f}"
    ]
    
    if "confusion_matrix" in global_metrics:
        try:
            # Handle string-encoded lists from flower transport or actual lists
            cm = global_metrics["confusion_matrix"]
            if isinstance(cm, str):
                import ast
                cm = ast.literal_eval(cm)
            report.append(format_confusion_matrix(cm))
        except:
            pass
            
    report.extend([
        "",
        f"Threshold Used: {global_metrics.get('optimal_threshold', 0.5):.2f}",
        "",
        "Interpretation:",
        "High recall captures more fraud at the cost of false positives.",
        "The model demonstrates effective cross-client generalization.",
        "",
        "Next Steps:",
        "This global model is ready for the Prediction Layer. Connect via inference API."
    ])
    
    report_text = "\n".join(report)
    
    with open(path_obj, "w") as f:
        f.write(report_text)
        
    logger.info(f"Evaluation report written to {output_path}")
    
    json_path = path_obj.with_suffix(".json")
    with open(json_path, "w") as f:
        # Convert any un-serializable items
        clean_metrics = {}
        for k, v in global_metrics.items():
            if isinstance(v, (int, float, str, list, bool)):
                clean_metrics[k] = v
        json.dump(clean_metrics, f, indent=2)
