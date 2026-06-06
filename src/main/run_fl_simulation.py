"""Flower FL Simulation Runner — FedProx + LiteFraudNet.

Execution order:
  1. python src/main/run_data_pipeline.py   ← split data & fit preprocessors
  2. python src/main/run_fl_simulation.py   ← this script
  3. python src/main/run_global_eval.py     ← evaluate global model

What this script does:
  - Pre-loads all client data + preprocessors from disk once.
  - Wraps each client in a FederatedClient (Flower NumPyClient).
  - Runs fl.simulation.start_simulation() for N rounds.
  - FedProxStrategy sends mu to every client each round via configure_fit.
  - Clients train locally with the proximal term, then send updated weights back.
  - Server aggregates (weighted average) and saves round checkpoint.
"""

import sys
import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.fl.client import FederatedClient
from src.fl.server import create_server_config, build_fedprox_strategy
from src.models.Fed_model import create_model
from src.data.preprocess import ClientPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
# Suppress redundant default dict dumps at the end of the Flower simulation
logging.getLogger("flwr").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_client_fn(
    client_ids: list,
    data_dir: str,
    artifacts_dir: str,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    privacy_config: Dict[str, Any],
):
    """Pre-load all client data once; return a Flower-compatible client_fn.

    Flower calls client_fn(cid) lazily, where cid is a string "0", "1", …
    We map cid → client_ids index.
    """
    import pandas as pd

    client_data: Dict[str, Dict] = {}

    for idx, cid_name in enumerate(client_ids):
        train_path = Path(data_dir) / f"{cid_name}_train.csv"
        val_path   = Path(data_dir) / f"{cid_name}_val.csv"
        prep_path  = Path(artifacts_dir) / "preprocessors" / f"{cid_name}_preprocessor.pkl"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Missing: {train_path}\n"
                "Run 'python src/main/run_data_pipeline.py' first."
            )
        if not prep_path.exists():
            raise FileNotFoundError(
                f"Missing: {prep_path}\n"
                "Run 'python src/main/run_single_baseline.py --client {cid_name}' "
                "or 'python src/main/run_data_pipeline.py' to create preprocessors."
            )

        preprocessor = ClientPreprocessor.load(str(prep_path))

        X_train_np, y_train_np = preprocessor.transform(pd.read_csv(train_path))
        X_val_np,   y_val_np   = preprocessor.transform(pd.read_csv(val_path))

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
        y_val   = torch.tensor(y_val_np,   dtype=torch.float32)

        padding_mask = (
            preprocessor.get_padding_mask()
            if hasattr(preprocessor, "get_padding_mask")
            else None
        )
        if padding_mask is not None and not isinstance(padding_mask, torch.Tensor):
            padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

        client_data[str(idx)] = dict(
            client_id=cid_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            padding_mask=padding_mask,
        )
        logger.info("  %s  train=%s  val=%s", cid_name, X_train.shape, X_val.shape)

    input_dim = client_data["0"]["X_train"].shape[1]

    def client_fn(cid: str) -> fl.client.NumPyClient:
        d = client_data[cid]
        return FederatedClient(
            client_id=d["client_id"],
            X_train=d["X_train"],   y_train=d["y_train"],
            X_val=d["X_val"],       y_val=d["y_val"],
            model_config=model_config,
            train_config=train_config,
            privacy_config=privacy_config,
            padding_mask=d["padding_mask"],
        )

    return client_fn, input_dim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FLOWER FEDERATED LEARNING — FedProx + LiteFraudNet")
    logger.info("=" * 60)

    # Load configs
    model_cfg_path = Path("configs/model_config.yaml")
    fl_cfg_path    = Path("configs/fl_config.yaml")

    if not model_cfg_path.exists() or not fl_cfg_path.exists():
        logger.error("Config files missing. Run 'python setup.py' first.")
        sys.exit(1)

    with open(model_cfg_path) as f:
        model_config = yaml.safe_load(f)
    with open(fl_cfg_path) as f:
        fl_config = yaml.safe_load(f)

    # FL hyper-parameters
    num_rounds   = fl_config.get("num_rounds", 10)
    num_clients  = fl_config.get("min_clients", 3)
    fedprox_mu   = fl_config.get("fedprox_mu", 0.01)
    fraction_fit = fl_config.get("fraction_fit", 1.0)

    artifacts_dir    = "artifacts"
    data_dir         = "data/splits"
    global_model_dir = Path(artifacts_dir) / "global_model"

    client_ids     = [f"client_{chr(97 + i)}" for i in range(num_clients)]
    privacy_config = fl_config.get("secure_update", {"enabled": False})

    train_config = {
        "epochs":     fl_config.get("local_epochs", 3),
        "batch_size": fl_config.get("batch_size", 256),
        "lr":         fl_config.get("lr", 0.001),
        "mu":         fedprox_mu,   # default; overridden by configure_fit each round
        "optimizer":  "adamw",
    }

    logger.info("Rounds=%d | Clients=%d | mu=%.4f", num_rounds, num_clients, fedprox_mu)

    # Pre-load data and get client factory
    logger.info("Loading client data and preprocessors …")
    client_fn, input_dim = build_client_fn(
        client_ids=client_ids,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        model_config=model_config,
        train_config=train_config,
        privacy_config=privacy_config,
    )

    # Initial global model parameters (check if we can resume from a previously saved final model)
    global_model = create_model(input_dim=input_dim, config=model_config)
    logger.info("Global model: %s", global_model)
    
    final_model_path = Path(global_model_dir) / "FINAL_global_model.pt"
    if final_model_path.exists():
        try:
            logger.info("Loading existing final model checkpoint to resume training: %s", final_model_path)
            ckpt = torch.load(str(final_model_path), map_location="cpu", weights_only=False)
            weights = ckpt.get("weights", ckpt.get("parameters", []))
            if weights:
                global_model.set_parameters(weights)
                logger.info("Successfully resumed global model weights from previous run (Round %d).", ckpt.get("round", 0))
        except Exception as e:
            logger.warning("Failed to load FINAL_global_model.pt (%s). Initializing with random weights.", e)

    initial_params = ndarrays_to_parameters(
        [p.cpu().numpy() for p in global_model.get_parameters()]
    )

    # FedProx strategy
    strategy = build_fedprox_strategy(
        fl_config=fl_config,
        initial_parameters=initial_params,
        artifacts_dir=str(global_model_dir)
    )

    # Run Flower simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=create_server_config(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Format and print history summary round-by-round
    logger.info("FEDERATED LEARNING HISTORY SUMMARY (Round-by-Round)\n")
    
    losses = dict(history.losses_distributed)
    metrics = history.metrics_distributed
    
    all_rounds = sorted(losses.keys())
    for r in all_rounds:
        loss_val = losses.get(r, 0.0)
        
        # Extract metrics for round r
        roc_auc = 0.0
        pr_auc = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        accuracy = 0.0
        
        for m_name, list_vals in metrics.items():
            val_dict = dict(list_vals)
            if r in val_dict:
                val = val_dict[r]
                if m_name == "roc_auc":
                    roc_auc = val
                elif m_name == "pr_auc":
                    pr_auc = val
                elif m_name == "f1":
                    f1 = val
                elif m_name == "precision":
                    precision = val
                elif m_name == "recall":
                    recall = val
                elif m_name == "accuracy":
                    accuracy = val
                    
        logger.info(
            f"Round {r:02d} -> Loss: {loss_val:.2f} | PR-AUC: {pr_auc:.2f} | ROC-AUC: {roc_auc:.2f} | "
            f"F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | Accuracy: {accuracy:.4f}"
        )

    logger.info("SIMULATION COMPLETE")
    logger.info("Checkpoints → %s", global_model_dir)
    logger.info("Next step  → python src/main/run_global_eval.py --metric pr_auc")
    return history


if __name__ == "__main__":
    main()
