"""Retrain LiteFraudNet + preprocessor on the user's Kaggle credit card dataset.

Usage:
    python scripts/retrain_on_kaggle.py

Output:
    - artifacts/preprocessors/global_preprocessor.pkl  (refitted on real data)
    - artifacts/global_model/FINAL_global_model.pt     (trained weights)
    - artifacts/global_model/model_card.json           (updated metadata)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import ClientPreprocessor
from src.models.Fed_model import LiteFraudNet
from src.models.train_engine import train_one_round, evaluate_model, predict_proba

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = r"C:\Users\likit\Downloads\creditcard.csv"
MAPPING_PATH = "configs/mapping.json"
ARTIFACTS_DIR = Path("artifacts")
PREPROCESSOR_DIR = ARTIFACTS_DIR / "preprocessors"
MODEL_DIR = ARTIFACTS_DIR / "global_model"
VECTOR_SIZE = 64
EPOCHS = 50
BATCH_SIZE = 512
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    logger.info("Device: %s", DEVICE)

    # 1. Load CSV
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    logger.info("Class distribution:\n%s", df["Class"].value_counts())

    # 2. Fit preprocessor on ALL data (we don't need label for fitting, just transform)
    logger.info("Fitting preprocessor (vector_size=%d)...", VECTOR_SIZE)
    preprocessor = ClientPreprocessor(
        mapping_path=MAPPING_PATH,
        vector_size=VECTOR_SIZE,
        scaler_type="robust",
    )
    X_all, y_all = preprocessor.fit_transform(df)
    logger.info("Preprocessor fitted. Output shape: %s", X_all.shape)

    # 3. Split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    logger.info("Train: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))

    # 4. Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # 5. Create model
    model = LiteFraudNet(
        input_dim=VECTOR_SIZE,
        hidden_dim=64,
        embedding_dim=32,
        dropout=0.20,
    )
    logger.info("Model: %s", model)

    # 6. Train
    train_config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "optimizer": "adamw",
        "mu": 0.0,
        "early_stopping_patience": 5,
    }
    logger.info("Training for up to %d epochs...", EPOCHS)
    _, metrics = train_one_round(
        model,
        X_train_t,
        y_train_t,
        train_config,
        device=DEVICE,
        X_val=X_val_t,
        y_val=y_val_t,
    )
    logger.info("Training metrics: %s", metrics)

    # 7. Evaluate on test set
    test_metrics = evaluate_model(model, X_test_t, y_test_t, DEVICE, batch_size=512)
    logger.info("Test metrics: %s", test_metrics)

    # 8. Find optimal threshold from validation set using F1 on precision-recall curve.
    # With real 0.17% fraud rate this gives a meaningful operational threshold.
    val_probs = predict_proba(model, X_val_t, DEVICE, batch_size=512)
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > best_idx else 0.5
    logger.info("Optimal threshold from val F1: %.4f", optimal_threshold)

    # 9. Save artifacts
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Preprocessor
    prep_path = PREPROCESSOR_DIR / "global_preprocessor.pkl"
    preprocessor.save(str(prep_path))
    logger.info("Preprocessor saved → %s", prep_path)

    # Model weights as state_dict (matches predictor loading path)
    model_path = MODEL_DIR / "FINAL_global_model.pt"
    torch.save({"model_state": model.state_dict()}, str(model_path))
    logger.info("Model weights saved → %s", model_path)

    # Model card
    model_card = {
        "model_type": "LiteFraudNet",
        "input_dim": VECTOR_SIZE,
        "hidden_dim": 64,
        "embedding_dim": 32,
        "trained_rounds": 1,
        "selected_round": 1,
        "selection_metric": "pr_auc",
        "global_test_pr_auc": round(test_metrics.get("pr_auc", 0.0), 4),
        "global_test_roc_auc": round(test_metrics.get("roc_auc", 0.0), 4),
        "global_test_f1": round(test_metrics.get("f1", 0.0), 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "dataset": "kaggle_credit_card",
        "num_clients": 1,
        "platform": "retrain_on_kaggle",
    }
    card_path = MODEL_DIR / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    logger.info("Model card saved → %s", card_path)

    # Quick smoke test: run predictor on same CSV
    logger.info("Running smoke test via GlobalModelPredictor...")
    from src.prediction.predictor import GlobalModelPredictor
    predictor = GlobalModelPredictor.from_artifacts(str(ARTIFACTS_DIR), device=str(DEVICE))
    result_df = predictor.predict(df)
    mean_risk = result_df["fraud_risk_score"].mean()
    logger.info("Smoke test: mean risk score = %.6f", mean_risk)
    logger.info("Smoke test: risk range = [%.6f, %.6f]", 
                result_df["fraud_risk_score"].min(), result_df["fraud_risk_score"].max())

    # Check that real fraud transactions get higher scores
    fraud_avg = result_df[result_df["Class"] == 1]["fraud_risk_score"].mean()
    legit_avg = result_df[result_df["Class"] == 0]["fraud_risk_score"].mean()
    logger.info("Avg risk — Fraud: %.6f, Legit: %.6f (ratio: %.2fx)", fraud_avg, legit_avg, 
                fraud_avg / max(legit_avg, 1e-12))

    fraud_max = result_df[result_df["Class"] == 1]["fraud_risk_score"].max()
    legit_min = result_df[result_df["Class"] == 0]["fraud_risk_score"].min()
    logger.info("Max fraud risk: %.6f, Min legit risk: %.6f", fraud_max, legit_min)
    
    if test_metrics.get("roc_auc", 0) > 0.7:
        logger.info("SUCCESS: Model has predictive power (ROC-AUC > 0.7)")
    else:
        logger.warning("MODEL NEEDS IMPROVEMENT: ROC-AUC = %.4f", test_metrics.get("roc_auc", 0))

    logger.info("Retraining complete!")


if __name__ == "__main__":
    main()
