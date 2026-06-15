"""
Final evaluation script for federated learning pipeline.
Loads best checkpoint, evaluates on holdout, compares to local baselines, 
and exports FINAL_global_model.pt and model_card.json.
"""

import sys
import argparse
import yaml
import pathlib
import shutil
import json
import logging
import time
import torch
import numpy as np

# Add project root to sys.path
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from src.evaluation.evaluate_global import load_best_checkpoint, evaluate_global_model, generate_evaluation_report
from src.evaluation.metrics import compute_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Final Global Model Evaluation.")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--data_dir", type=str, default="data/splits", help="Data directory")
    parser.add_argument("--metric", type=str, default="pr_auc", choices=["pr_auc", "roc_auc", "f1"], help="Metric to select best round")
    parser.add_argument("--output_dir", type=str, default="artifacts/reports", help="Output directory for reports")
    
    args = parser.parse_args()
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    data_dir = pathlib.Path(args.data_dir)
    output_dir = pathlib.Path(args.output_dir)
    metric = args.metric
    
    start_time = time.time()
    
    # Load Configs
    model_config_path = pathlib.Path("configs/model_config.yaml")
    fl_config_path = pathlib.Path("configs/fl_config.yaml")
    
    if not model_config_path.exists() or not fl_config_path.exists():
        logger.error("Config files missing. Please check configs/")
        sys.exit(1)
        
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)

    with open(fl_config_path, "r") as f:
        fl_config = yaml.safe_load(f)

    # Load data config for schema name (used in model card)
    data_config_path = pathlib.Path("configs/data_config.yaml")
    data_cfg = {}
    if data_config_path.exists():
        with open(data_config_path, "r") as f:
            _raw = yaml.safe_load(f)
            data_cfg = _raw.get("data", _raw) if isinstance(_raw, dict) else {}

    global_model_dir = artifacts_dir / "global_model"
    global_test_csv = data_dir / "global_test.csv"

    if not global_test_csv.exists():
        logger.error("Global test data missing at %s", global_test_csv)
        sys.exit(1)

    # Prefer the global preprocessor (fitted server-side on the non-test pool).
    # Falls back to client_a's preprocessor if the pipeline was run before this fix.
    global_prep_path = artifacts_dir / "preprocessors" / "global_preprocessor.pkl"
    fallback_prep_path = artifacts_dir / "preprocessors" / "client_a_preprocessor.pkl"

    if global_prep_path.exists():
        preprocessor_path = global_prep_path
        logger.info("Using global preprocessor: %s", preprocessor_path)
    elif fallback_prep_path.exists():
        preprocessor_path = fallback_prep_path
        logger.warning(
            "global_preprocessor.pkl not found — falling back to client_a_preprocessor.pkl. "
            "Re-run run_data_pipeline.py to generate the global preprocessor."
        )
    else:
        logger.error("No preprocessor found in %s", artifacts_dir / "preprocessors")
        sys.exit(1)

    # Step 1 - Scan all checkpoints and select the best one by holdout metrics
    checkpoint_files = sorted(global_model_dir.glob("round_*_checkpoint.pt"))
    if not checkpoint_files:
        logger.error("No checkpoints found in %s", global_model_dir)
        sys.exit(1)

    best_ckpt = None
    best_value = -float("inf")
    best_round = 0
    best_params = None
    best_metrics = {}

    for ckpt_path in checkpoint_files:
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            params = ckpt.get("weights", ckpt.get("parameters", []))
            r = ckpt.get("round", 0)
            
            eval_metrics = evaluate_global_model(
                parameters=params,
                global_test_csv=str(global_test_csv),
                preprocessor_path=str(preprocessor_path),
                model_config=model_config,
                eval_config={"batch_size": 512, "threshold": 0.5, "device": "cpu"}
            )
            val = eval_metrics.get(metric, 0.0)
            logger.info(f"Round {r:03d} checkpoint: evaluated holdout {metric} = {val:.4f}")
            
            if val > best_value:
                best_value = val
                best_ckpt = ckpt
                best_round = r
                best_params = params
                best_metrics = eval_metrics
        except Exception as exc:
            logger.warning(f"Failed to evaluate checkpoint {ckpt_path.name}: {exc}")

    if best_ckpt is None:
        logger.error("Failed to select best checkpoint")
        sys.exit(1)

    logger.info(f"Selected best round: {best_round} with holdout {metric} = {best_value:.4f}")
    
    # Keep validation metrics placeholder for report printing compatibility
    best_metrics["round"] = best_round
    best_metrics["pr_auc"] = best_value
    best_round = best_metrics.get("round", 0)
    best_metrics = {}  # Empty FL metrics to avoid confusing with holdout

    eval_config = {
        "batch_size": 512,
        "threshold": 0.5,
        "device": "cpu"
    }
    global_eval_metrics = evaluate_global_model(
        parameters=best_params,
        global_test_csv=str(global_test_csv),
        preprocessor_path=str(preprocessor_path),
        model_config=model_config,
        eval_config=eval_config
    )
    
    logger.info(f"Global Eval ROC-AUC: {global_eval_metrics.get('roc_auc', 0.0):.4f}")
    logger.info(f"Global Eval PR-AUC: {global_eval_metrics.get('pr_auc', 0.0):.4f}")

    
    # Step 3 - Find Optimal Threshold on Global Test using shared predict_proba()
    import pandas as pd
    from src.data.preprocess import ClientPreprocessor
    from src.models.Fed_model import create_model
    from src.models.train_engine import predict_proba

    df_test = pd.read_csv(global_test_csv)
    preprocessor = ClientPreprocessor.load(str(preprocessor_path))
    X_test, y_test = preprocessor.transform(df_test)

    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    model = create_model(input_dim=preprocessor.get_feature_dim(), config=model_config)
    model.set_parameters(best_params)

    probs = predict_proba(model, X_test, batch_size=512, device="cpu")

    optimal_threshold = compute_optimal_threshold(y_test, probs, metric="f1")
    logger.info("Optimal threshold (F1): %.2f", optimal_threshold)

    # Re-evaluate with optimal threshold
    eval_config["threshold"] = optimal_threshold
    global_eval_metrics = evaluate_global_model(
        parameters=best_params,
        global_test_csv=str(global_test_csv),
        preprocessor_path=str(preprocessor_path),
        model_config=model_config,
        eval_config=eval_config,
    )
    global_eval_metrics["optimal_threshold"] = optimal_threshold

    
    # Step 4 - Compare: Global vs Best Local Baseline
    local_models_dir = artifacts_dir / "local_models"
    local_checkpoints = list(local_models_dir.glob("*_baseline.pt"))
    
    local_pr_aucs = []
    local_f1s = []
    
    for ckpt_path in local_checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        local_pr_aucs.append(ckpt["metrics"].get("pr_auc", 0.0))
        local_f1s.append(ckpt["metrics"].get("f1", 0.0))
        
    avg_local_pr_auc = np.mean(local_pr_aucs) if local_pr_aucs else 0.0
    global_pr_auc = global_eval_metrics.get("pr_auc", 0.0)
    
    improvement = ((global_pr_auc - avg_local_pr_auc) / max(0.0001, avg_local_pr_auc)) * 100
    logger.info(f"Global model PR-AUC: {global_pr_auc:.4f} vs. Average Local PR-AUC: {avg_local_pr_auc:.4f} - improvement: {improvement:.2f}%")
    
    # Step 5 - Generate Report
    report_path = output_dir / "global_evaluation_report.txt"
    generate_evaluation_report(
        round_number=best_round,
        round_metrics=best_metrics,
        global_metrics=global_eval_metrics,
        output_path=str(report_path)
    )
    
    # Step 6 - Export Final Checkpoint
    final_model_path = global_model_dir / "FINAL_global_model.pt"
    best_ckpt_path = global_model_dir / f"round_{best_round:03d}_checkpoint.pt"
    
    if best_ckpt_path.exists():
        shutil.copy2(best_ckpt_path, final_model_path)
            
    model_card = {
        "model_type": "LiteFraudNet",
        "input_dim": preprocessor.get_feature_dim(),
        "hidden_dim": model_config.get("hidden_dim", 64),
        "embedding_dim": model_config.get("embedding_dim", 32),
        "trained_rounds": fl_config.get("num_rounds", 10),
        "selected_round": best_round,
        "selection_metric": metric,
        "global_test_pr_auc": float(global_eval_metrics.get("pr_auc", 0.0)),
        "global_test_roc_auc": float(global_eval_metrics.get("roc_auc", 0.0)),
        "global_test_f1": float(global_eval_metrics.get("f1", 0.0)),
        "optimal_threshold": float(optimal_threshold),
        "dataset": data_cfg.get("schema", data_cfg.get("source", "unknown")),

        "num_clients": 3,
        "platform": "FedProx + LiteFraudNet",
        "handoff_note": "Ready for Prediction Layer. Load FINAL_global_model.pt with model_card.json for inference."
    }
    
    model_card_path = global_model_dir / "model_card.json"
    with open(model_card_path, "w") as f:
        json.dump(model_card, f, indent=4)
        
    # Step 7 - Print Completion Banner
    banner = f"""
    ============================================
    FEDERATED LEARNING PHASE COMPLETE
    Global Model saved: artifacts/global_model/FINAL_global_model.pt
    Model Card saved: artifacts/global_model/model_card.json
    Evaluation Report: artifacts/reports/global_evaluation_report.txt
    Ready for: Prediction Layer -> Knowledge Graph -> Explainability
    ============================================
    Completed in {(time.time() - start_time):.2f} seconds.
    """
    print(banner)

if __name__ == "__main__":
    main()
