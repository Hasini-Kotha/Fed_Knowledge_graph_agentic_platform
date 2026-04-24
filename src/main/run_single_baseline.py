"""
End-to-end baseline runner for a single client.
This script is Person 2's deliverable. If this runs successfully for all 3 clients, 
Person 3 can begin Flower integration.

YAML config structure:
# configs/model_config.yaml
# numeric_cols: [Time, V1, V2, ..., V28, Amount]
# hidden_dims: [64, 32]
# dropout_rate: 0.2
# scaler_type: robust
# train:
#   epochs: 5
#   batch_size: 256
#   lr: 0.001
#   seed: 42
# eval:
#   batch_size: 512
#   threshold: 0.5
"""

import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import numpy as np

# Add project root to sys.path
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from src.data.preprocess import ClientPreprocessor
from src.models.mlp import create_model
from src.models.train_local import train_one_round, evaluate_client, save_local_checkpoint
from src.evaluation.metrics import print_metrics_report, compute_optimal_threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run single baseline for a client.")
    parser.add_argument("--client", type=str, default="client_a", choices=["client_a", "client_b", "client_c"], help="Client ID")
    parser.add_argument("--data_dir", type=str, default="data/splits", help="Directory containing data splits")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save artifacts")
    
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
        
    numeric_cols = config.get("numeric_cols")
    scaler_type = config.get("scaler_type", "robust")
    train_config = config.get("train", {})
    eval_config = config.get("eval", {})
    
    logger.info(f"Starting baseline run for {client}")
    
    # Step 1 - Load Data
    train_path = data_dir / f"{client}_train.csv"
    val_path = data_dir / f"{client}_val.csv"
    
    if not train_path.exists() or not val_path.exists():
        logger.error(f"Data files missing for {client}. Please check {data_dir}")
        sys.exit(1)
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Step 2 - Preprocess
    preprocessor = ClientPreprocessor(numeric_cols=numeric_cols, scaler_type=scaler_type)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    
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
    import torch
    from torch.utils.data import DataLoader
    from src.models.train_local import ClientDataset
    
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
    fraud_ratio = np.mean(y_train) * 100
    summary = f"""
    Summary for {client}:
    - Feature dim: {input_dim}
    - Train samples: {len(X_train)}
    - Val samples: {len(X_val)}
    - Fraud ratio (train): {fraud_ratio:.2f}%
    - ROC-AUC: {eval_metrics.get('roc_auc', 0.0):.4f}
    - PR-AUC: {eval_metrics.get('pr_auc', 0.0):.4f}
    - F1: {eval_metrics.get('f1', 0.0):.4f}
    - Optimal Threshold: {optimal_threshold:.2f}
    """
    print(summary)
    logger.info("Baseline run completed successfully.")

if __name__ == "__main__":
    main()
