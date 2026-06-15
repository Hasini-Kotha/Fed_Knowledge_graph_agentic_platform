"""local_fl_runner.py — Simulates all 3 bank clients running on one laptop.

This script is the HACKATHON DEMO entry point.
It runs the full federated learning loop through the gateway:

    For each FL round:
        For each client (bank_a, bank_b, bank_c):
            1. Login to get JWT
            2. Train locally using their data split
            3. Submit trained weights to POST /submit-update

After all rounds complete:
    Run run_global_eval.py to select the best model checkpoint.

Usage:
    # Terminal 1: Start the gateway
    uvicorn src.gateway.gateway:app --port 8000

    # Terminal 2: Run this script
    python src/gateway/local_fl_runner.py
"""

# Load .env FIRST — before any other import reads os.environ
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()   # reads .env from CWD (project root)

import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import ClientPreprocessor
from src.models.Fed_model import create_model
from src.models.train_engine import train_one_round, save_local_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("local_fl_runner")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = "http://localhost:8000"
DATA_DIR = Path("data/splits")
ARTIFACTS_DIR = Path("artifacts")

# Registered client credentials (must match what was registered in the DB)
CLIENTS = [
    {"client_id": "bank_a", "bank_name": "Alpha Bank",  "password": "BankAlpha123!"},
    {"client_id": "bank_b", "bank_name": "Beta Bank",   "password": "BankBeta123!"},
    {"client_id": "bank_c", "bank_name": "Gamma Bank",  "password": "BankGamma123!"},
]

# Maps gateway client_id → local data split name
CLIENT_DATA_MAP = {
    "bank_a": "client_a",
    "bank_b": "client_b",
    "bank_c": "client_c",
}

MODEL_VERSION = "LiteFraudNet-v1"


# ---------------------------------------------------------------------------
# Helper: register + login
# ---------------------------------------------------------------------------

def register_client(client: dict) -> None:
    """Register if not already registered (silently skips 409 Conflict)."""
    resp = requests.post(f"{GATEWAY_URL}/fl/register", json=client, timeout=10)
    if resp.status_code == 201:
        logger.info("Registered: %s", client["client_id"])
    elif resp.status_code == 409:
        logger.info("Already registered: %s", client["client_id"])
    else:
        logger.error("Registration failed for %s: %s", client["client_id"], resp.text)


def login(client: dict) -> str:
    """Login and return the JWT access token."""
    resp = requests.post(
        f"{GATEWAY_URL}/fl/login",
        json={"client_id": client["client_id"], "password": client["password"]},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    logger.info("JWT acquired: %s", client["client_id"])
    return token


# ---------------------------------------------------------------------------
# Helper: load client data
# ---------------------------------------------------------------------------

def load_client_data(split_name: str, model_config: dict):
    """Load preprocessed tensors for a client split."""
    preprocessor_path = ARTIFACTS_DIR / "preprocessors" / f"{split_name}_preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}\n"
            f"Run: python src/main/run_single_baseline.py --client {split_name}"
        )

    preprocessor = ClientPreprocessor.load(str(preprocessor_path))

    train_df_path = DATA_DIR / f"{split_name}_train.csv"
    val_df_path   = DATA_DIR / f"{split_name}_val.csv"

    import pandas as pd
    train_df = pd.read_csv(train_df_path)
    val_df   = pd.read_csv(val_df_path)

    X_train_np, y_train_np = preprocessor.transform(train_df)
    X_val_np,   y_val_np   = preprocessor.transform(val_df)

    return (
        torch.tensor(X_train_np, dtype=torch.float32),
        torch.tensor(y_train_np, dtype=torch.float32),
        torch.tensor(X_val_np,   dtype=torch.float32),
        torch.tensor(y_val_np,   dtype=torch.float32),
        preprocessor.get_feature_dim(),
    )


def fetch_global_weights(token: str, client_id: str) -> list:
    """Download and decrypt current global weights from the gateway."""
    from src.gateway.encryption import decrypt_weights
    
    resp = requests.get(
        f"{GATEWAY_URL}/fl/global-weights",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    
    encrypted_weights = data["encrypted_weights"]
    weights = decrypt_weights(encrypted_weights)
    logger.info("[%s] Downloaded and decrypted global weights from gateway.", client_id)
    return weights


# ---------------------------------------------------------------------------
# Helper: local training for one round
# ---------------------------------------------------------------------------

def train_local(client_id: str, split_name: str, model_config: dict,
                fl_config: dict, current_round: int, global_weights: list = None) -> tuple:
    """Train locally and return (weights_as_lists, n_train)."""
    logger.info("[%s] Loading data …", client_id)
    X_train, y_train, X_val, y_val, input_dim = load_client_data(split_name, model_config)

    model = create_model(input_dim=input_dim, config=model_config)

    # Load global model weights if available (downloaded and decrypted from gateway)
    if global_weights:
        model.set_parameters(global_weights)
        logger.info("[%s] Loaded decrypted global weights into local model parameters.", client_id)

    train_config = {
        "epochs":   fl_config.get("local_epochs", 3),
        "batch_size": fl_config.get("batch_size", 256),
        "lr":         fl_config.get("lr", 0.001),
        "mu":         fl_config.get("fedprox_mu", 0.01),
        "optimizer":  model_config.get("optimizer", "adamw"),
        "early_stopping_patience": 5,
    }

    logger.info("[%s] Training locally (round=%d, epochs=%d) …",
                client_id, current_round, train_config["epochs"])

    params, metrics = train_one_round(
        model=model,
        X_train=X_train,
        y_train=y_train,
        train_config=train_config,
        X_val=X_val,
        y_val=y_val,
    )

    logger.info("[%s] Local train done | PR-AUC=%.4f", client_id,
                metrics.get("pr_auc", 0.0))

    # Prove local model storage: Save the checkpoint locally to the bank's directory
    checkpoint_dir = ARTIFACTS_DIR / "checkpoints" / client_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"local_model_round_{current_round}.pt"
    
    save_local_checkpoint(model, None, metrics, str(checkpoint_path))
    logger.info("[%s] Saved local model checkpoint to: %s", client_id, checkpoint_path)

    weights_as_lists = [p.cpu().numpy().tolist() for p in params]
    return weights_as_lists, int(len(X_train))


# ---------------------------------------------------------------------------
# Helper: submit weights via gateway
# ---------------------------------------------------------------------------

def submit_weights(token: str, client_id: str, round_num: int,
                   weights: list, n_samples: int) -> dict:
    from src.gateway.encryption import encrypt_weights, sign_payload
    
    # Convert weights to list of numpy arrays
    np_weights = [np.array(w, dtype=np.float32) for w in weights]
    
    # Encrypt
    encrypted_weights = encrypt_weights(np_weights)
    
    # Sign
    signature = sign_payload(client_id, round_num, encrypted_weights)
    
    try:
        resp = requests.post(
            f"{GATEWAY_URL}/fl/submit-update",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "round_num":     round_num,
                "model_version": MODEL_VERSION,
                "n_samples":     n_samples,
                "encrypted_weights": encrypted_weights,
                "signature":     signature,
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error("HTTP Error from Gateway: %s | Response: %s", err, err.response.text)
        raise err
    data = resp.json()
    logger.info("[%s] Gateway: %s", client_id, data["message"])
    return data


def reset_server_state():
    """Authenticate as admin and reset in-memory round state on the gateway."""
    admin_credentials = {
        "client_id": "admin",
        "password": "AdminSecure123!"
    }
    logger.info("Logging in as admin to reset gateway FL state...")
    try:
        resp = requests.post(f"{GATEWAY_URL}/fl/login", json=admin_credentials, timeout=10)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        
        resp = requests.post(
            f"{GATEWAY_URL}/fl/admin/reset",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        resp.raise_for_status()
        logger.info("Gateway: %s", resp.json()["message"])
    except Exception as exc:
        logger.warning("Failed to reset gateway state (this is normal if starting first time): %s", exc)


# ---------------------------------------------------------------------------
# Main FL loop
# ---------------------------------------------------------------------------

import argparse

def run():
    parser = argparse.ArgumentParser(description="Local FL Runner for a specific bank")
    parser.add_argument("--bank", type=str, help="Client ID to run (e.g., bank_a, bank_b, bank_c)")
    args = parser.parse_args()

    # Load configs
    with open("configs/fl_config.yaml")    as f: fl_config    = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f: model_config = yaml.safe_load(f)

    num_rounds = fl_config.get("num_rounds", 10)

    active_clients = CLIENTS
    if args.bank:
        active_clients = [c for c in CLIENTS if c["client_id"] == args.bank]
        if not active_clients:
            logger.error("Bank %s not found in CLIENTS", args.bank)
            return

    logger.info("=" * 60)
    logger.info("LOCAL FL RUNNER — %d rounds, %d clients", num_rounds, len(active_clients))
    logger.info("=" * 60)

    # Only reset server state if running all or running Admin
    if not args.bank:
        reset_server_state()

    # Step 1: Register clients (idempotent)
    logger.info("--- Step 1: Registering clients ---")
    for client in active_clients:
        register_client(client)

    # Step 2: FL rounds
    for round_num in range(1, num_rounds + 1):
        logger.info("\n--- Round %d / %d ---", round_num, num_rounds)

        for client in active_clients:
            cid      = client["client_id"]
            split    = CLIENT_DATA_MAP[cid]
            token    = login(client)

            # Fetch and decrypt current global model weights from gateway
            global_weights = fetch_global_weights(token, cid)

            weights, n_samples = train_local(
                client_id=cid,
                split_name=split,
                model_config=model_config,
                fl_config=fl_config,
                current_round=round_num,
                global_weights=global_weights,
            )

            submit_weights(token, cid, round_num, weights, n_samples)

        logger.info("Round %d complete. Waiting 2s …", round_num)
        time.sleep(2)

    logger.info("\n" + "=" * 60)
    logger.info("LOCAL FL RUNNER COMPLETE")
    logger.info("Next step → python src/main/run_global_eval.py --metric pr_auc")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
