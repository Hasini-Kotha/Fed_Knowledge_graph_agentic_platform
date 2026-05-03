"""
Main federated learning simulation runner.
Uses Flower's simulation mode (flwr.simulation.start_simulation) which runs
multiple clients on one machine with no actual networking needed.

FL config YAML structure:
# configs/fl_config.yaml
# num_rounds: 10
# fraction_fit: 1.0
# fraction_evaluate: 1.0
# min_fit_clients: 3
# min_evaluate_clients: 3
# min_available_clients: 3
# local_epochs: 3
# batch_size: 256
# lr: 0.001
# secure_update:
#   max_norm: 1.0
#   noise_multiplier: 0.0
#   validate: true
"""

import sys
import argparse
import yaml
import pathlib
import logging
import json
import time
import flwr as fl

# Add project root to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from src.fl.client import create_client_fn
from src.fl.strategy import build_strategy
from src.fl.server import create_server_config, get_initial_parameters
from src.data.preprocess import ClientPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run FL simulation.")
    parser.add_argument("--data_dir", type=str, default="data/splits", help="Directory containing data splits")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--num_rounds", type=int, default=None, help="Override number of rounds from config")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    artifacts_dir = pathlib.Path(args.artifacts_dir)
    
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
        
    if args.num_rounds is not None:
        fl_config["num_rounds"] = args.num_rounds
        
    # Check if data splits exist
    split_dir = pathlib.Path(data_dir)
    required_files = ["client_a_train.csv", "client_b_train.csv", "client_c_train.csv"]
    missing_files = [f for f in required_files if not (split_dir / f).exists()]
    
    if missing_files:
        logger.warning(f"Data splits missing: {missing_files}")
        logger.info("Automatically running data preparation pipeline...")
        try:
            from src.main.run_data_pipeline import run_pipeline
            run_pipeline()
        except ImportError:
            logger.error("Could not import run_pipeline. Please run 'python src/main/run_data_pipeline.py' manually.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            sys.exit(1)
            
    train_config = {
        "epochs": fl_config.get("local_epochs", 3),
        "batch_size": fl_config.get("batch_size", 256),
        "lr": fl_config.get("lr", 0.001),
        "eval_batch_size": model_config.get("eval", {}).get("batch_size", 512),
        "eval_threshold": model_config.get("eval", {}).get("threshold", 0.5),
        "secure_update": fl_config.get("secure_update", {})
    }
    
    # Step 1 - Validate Preprocessor Artifacts
    clients = ["client_a", "client_b", "client_c"]
    preprocessors_dir = artifacts_dir / "preprocessors"
    
    for client in clients:
        if not (preprocessors_dir / f"{client}_preprocessor.pkl").exists():
            logger.error(f"Run run_single_baseline.py for all 3 clients first (missing {client})")
            sys.exit(1)
            
    # Step 2 - Determine Input Dim
    client_a_prep = ClientPreprocessor.load(str(preprocessors_dir / "client_a_preprocessor.pkl"))
    input_dim = client_a_prep.get_feature_dim()
    logger.info(f"Input dimension: {input_dim}")
    
    # Step 3 - Get Initial Parameters
    initial_parameters = get_initial_parameters(model_config, input_dim)
    
    # Step 4 - Build Strategy (initial_parameters passed via __init__, not post-init)
    strategy = build_strategy(str(artifacts_dir), fl_config, initial_parameters=initial_parameters)
    
    # Step 5 - Define client_fn for simulation
    # Flower simulation uses integer-string CIDs ("0", "1", "2").
    # Manual fallback uses the same string keys via cid_to_client.
    cid_to_client = {
        "0": "client_a",
        "1": "client_b",
        "2": "client_c",
    }
    simulation_client_ids = list(cid_to_client.keys())  # ["0", "1", "2"]
    
    def client_fn(cid: str) -> fl.client.NumPyClient:
        client_id = cid_to_client[cid]
        return create_client_fn(
            client_id=client_id,
            data_dir=data_dir,
            artifacts_dir=str(artifacts_dir),
            model_config=model_config,
            train_config=train_config
        )
        
    # Step 6 - Run Simulation
    server_config = create_server_config(fl_config)
    num_rounds = fl_config.get("num_rounds", 10)
    
    logger.info("Starting FL simulation...")
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=3,
            config=server_config,
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0}
        )
    except ImportError as e:
        if "ray" in str(e).lower():
            logger.warning("Ray unavailable — falling back to manual Ray-free loop.")
            from src.fl.manual_loop import run_manual_simulation
            run_manual_simulation(
                client_fn=client_fn,
                client_ids=simulation_client_ids,
                num_rounds=num_rounds,
                strategy=strategy,
            )
        else:
            raise e
    except Exception as e:
        logger.error("Simulation failed (%s) — attempting manual fallback.", e)
        from src.fl.manual_loop import run_manual_simulation
        run_manual_simulation(
            client_fn=client_fn,
            client_ids=simulation_client_ids,
            num_rounds=num_rounds,
            strategy=strategy,
        )
    
    # Step 7 - Save History
    strategy.save_metrics_history()
    
    # Step 8 - Print Final Summary
    metrics_history = strategy.round_metrics_history
    if not metrics_history:
        logger.warning("No metrics history recorded.")
        sys.exit(1)
        
    total_rounds = len(metrics_history)
    final_round = metrics_history[-1]
    
    best_round = max(metrics_history, key=lambda x: x["metrics"].get("pr_auc", 0.0))
    
    summary = f"""
    ============================================
    FL Simulation Completed in {(time.time() - start_time) / 60:.2f} minutes
    - Total rounds completed: {total_rounds}
    - Final round ({final_round['round']}) Metrics:
      ROC-AUC: {final_round['metrics'].get('roc_auc', 0.0):.4f}
      PR-AUC:  {final_round['metrics'].get('pr_auc', 0.0):.4f}
      F1:      {final_round['metrics'].get('f1', 0.0):.4f}
      
    - Best round by PR-AUC: Round {best_round['round']} with PR-AUC = {best_round['metrics'].get('pr_auc', 0.0):.4f}
    - Global model checkpoints saved to: {artifacts_dir}/global_model/
    ============================================
    """
    print(summary)
    logger.info("FL simulation completed successfully.")

if __name__ == "__main__":
    main()
