"""
Shared metric utilities for the federated learning platform.
"""

import numpy as np
import logging
from sklearn.metrics import f1_score, recall_score
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def format_confusion_matrix(cm: List[List[int]]) -> str:
    """
    Returns a pretty-printed string of a 2x2 confusion matrix with labels TN, FP, FN, TP.
    """
    if not cm or len(cm) != 2 or len(cm[0]) != 2:
        return str(cm)
        
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    formatted_str = (
        f"\n  Confusion Matrix:"
        f"\n    [TN: {tn:<6} | FP: {fp:<6}]"
        f"\n    [FN: {fn:<6} | TP: {tp:<6}]"
    )
    return formatted_str

def print_metrics_report(metrics: Dict[str, Any], client_id: str = "unknown") -> None:
    """
    Prints a well-formatted report of all metrics to stdout and logs it.
    """
    report = f"\n=== Metrics Report: {client_id} ==="
    report += f"\n  Samples:   {metrics.get('num_samples', 'N/A')}"
    report += f"\n  Positives: {metrics.get('num_positive', 'N/A')}"
    report += f"\n  ROC-AUC:   {metrics.get('roc_auc', 0.0):.4f}"
    report += f"\n  PR-AUC:    {metrics.get('pr_auc', 0.0):.4f}"
    report += f"\n  Precision: {metrics.get('precision', 0.0):.4f}"
    report += f"\n  Recall:    {metrics.get('recall', 0.0):.4f}"
    report += f"\n  F1 Score:  {metrics.get('f1', 0.0):.4f}"
    
    if "confusion_matrix" in metrics:
        report += format_confusion_matrix(metrics["confusion_matrix"])
        
    report += "\n==================================="
    
    print(report)
    logger.info(report)

def compute_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """
    Scans thresholds from 0.1 to 0.9 in steps of 0.01 to find the optimal threshold.
    """
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_threshold = 0.5
    best_score = -1.0
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric '{metric}' for threshold optimization. Use 'f1' or 'recall'.")
            
        if score > best_score:
            best_score = score
            best_threshold = t
            
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.2f} (score: {best_score:.4f})")
    return float(best_threshold)

def aggregate_client_metrics(all_client_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Takes a list of per-client metric dicts and returns a weighted average.
    """
    if not all_client_metrics:
        return {}
        
    total_samples = sum(m.get("num_samples", 0) for m in all_client_metrics)
    if total_samples == 0:
        return {"participating_clients": len(all_client_metrics)}
        
    aggregated = {
        "roc_auc": 0.0,
        "pr_auc": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    for m in all_client_metrics:
        weight = m.get("num_samples", 0) / total_samples
        for k in aggregated.keys():
            aggregated[k] += m.get(k, 0.0) * weight
            
    aggregated["participating_clients"] = len(all_client_metrics)
    return aggregated
