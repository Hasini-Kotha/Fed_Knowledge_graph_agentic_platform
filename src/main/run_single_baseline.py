"""
End-to-end baseline runner for a single client.
Demonstrates the full Phase 1-3 dynamic vectorization pipeline.

YAML config structure (configs/model_config.yaml):
  vector_size: 64          # Phase 1 global contract dimension
  scaler_type: robust      # 'robust' or 'standard'
  hidden_dims: [64, 32]
  dropout_rate: 0.2
  train:
    epochs: 5
    batch_size: 256
    lr: 0.001
    seed: 42
  eval:
    batch_size: 512
    threshold: 0.5

Mapping file (configs/mapping.json):
  The ONLY place where industry-specific column names appear at runtime.
  Switch datasets by swapping this file — no Python changes required.
"""

import sys
import logging
import argparse
import yaml
import pathlib
import pandas as pd
import numpy as np
import torch

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from src.data.preprocess import ClientPreprocessor
from src.models.tab_transformer import create_model
from src.models.train_engine import train_one_round, evaluate_client, save_local_checkpoint
from src.evaluation.metrics import print_metrics_report, compute_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run single baseline for a client.")
    parser.add_argument("--client", type=str, default="client_a", choices=["client_a", "client_b", "client_c"], help="Client ID")
    parser.add_argument("--data_dir", type=str, default="data/splits", help="Directory containing data splits")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save artifacts")
    # Phase 2: each client can override mapping path for their own domain map
    parser.add_argument("--mapping", type=str, default="configs/mapping.json", help="Path to client's domain mapping JSON")
    
    args = parser.parse_args()
    client = args.client
    data_dir = pathlib.Path(args.data_dir)
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    
    # Load Config
    config_path = pathlib.Path("configs/model_config.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    scaler_type  = config.get("scaler_type", "robust")
    vector_size  = config.get("vector_size", 64)
    train_config = config.get("train", {})
    eval_config  = config.get("eval", {})
    mapping_path = args.mapping  # Phase 2: read from CLI, not hardcoded
    
    logger.info(f"Starting baseline run for {client}")
    
    # Step 1 - Load Data
    train_path = data_dir / f"{client}_train.csv"
    val_path = data_dir / f"{client}_val.csv"
    
    if not train_path.exists() or not val_path.exists():
        logger.error(f"Data files missing for {client}. Please check {data_dir}")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Step 2 - Dynamic Vectorization (Phase 1, 2, 3)
    logger.info("Initialising Dynamic Vectorizer from mapping: %s", mapping_path)
    preprocessor = ClientPreprocessor(
        mapping_path=mapping_path,
        vector_size=vector_size,
        scaler_type=scaler_type,
    )
    logger.info("Mapping loaded: %s", preprocessor.get_mapping_summary())
    X_train_np, y_train_np = preprocessor.fit_transform(train_df)
    X_val_np, y_val_np = preprocessor.transform(val_df)
    
    # Convert to torch tensors — train_engine expects tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    
    import torch
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
    
    preprocessor_path = artifacts_dir / "preprocessors" / f"{client}_preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    
    # Step 3 - Create Model
    input_dim = preprocessor.get_feature_dim()
    model = create_model(input_dim=input_dim, config=config)
    
    # Step 4 - Train
    logger.info("Starting local training...")
    _, train_metrics = train_one_round(model, X_train, y_train, train_config)
    logger.info(f"Training metrics: {train_metrics}")
    
    # Step 5 - Evaluate
    logger.info("Starting evaluation...")
    eval_metrics = evaluate_client(model, X_val, y_val, eval_config)
    print_metrics_report(eval_metrics, client_id=client)
    
    # Step 6 - Find Optimal Threshold
    # Re-run inference to get probabilities
    from torch.utils.data import DataLoader
    from src.models.train_engine import ClientDataset
    
    dataset = ClientDataset(X_val, y_val)
    dataloader = DataLoader(dataset, batch_size=eval_config.get("batch_size", 512), shuffle=False)
    
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_X, _ in dataloader:
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            
    probs = np.array(all_probs).flatten()
    optimal_threshold = compute_optimal_threshold(y_val, probs, metric="f1")
    logger.info(f"Optimal threshold for {client}: {optimal_threshold}")
    
    # Add optimal threshold to metrics before saving
    eval_metrics["optimal_threshold"] = optimal_threshold
    
    # Step 7 - Save Checkpoint
    checkpoint_path = artifacts_dir / "local_models" / f"{client}_baseline.pt"
    save_local_checkpoint(model, preprocessor, eval_metrics, checkpoint_path)
    
    # Step 8 - Print Summary
    positive_ratio = (y_train.float().mean().item()) * 100
    summary = f"""
    Summary for {client}:
    - Feature dim: {input_dim}
    - Train samples: {len(X_train)}
    - Val samples: {len(X_val)}
    - Positive-class ratio (train): {positive_ratio:.2f}%
    - ROC-AUC: {eval_metrics.get('roc_auc', 0.0):.4f}
    - PR-AUC: {eval_metrics.get('pr_auc', 0.0):.4f}
    - F1: {eval_metrics.get('f1', 0.0):.4f}
    - Optimal Threshold: {optimal_threshold:.2f}
    """
    print(summary)
    logger.info("Baseline run completed successfully.")

if __name__ == "__main__":
    main()
