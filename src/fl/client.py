"""Federated Learning Client — Flower NumPyClient wrapper.

Each client manages its own local data, preprocessing, and training.
It communicates with the server only via model weights (never raw data).

Security: Before sending weights back to server, the client applies:
  1. Weight delta computation (local - global)
  2. L2 norm clipping (max_norm)
  3. Optional Gaussian noise injection (DP-SGD)
  4. SHA-256 checksum for audit trail
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch

import flwr as fl

from src.models.tab_transformer import create_model
from src.models.train_engine import train_one_round, evaluate_client
from src.fl.secure_update import protect_update

logger = logging.getLogger(__name__)


class FederatedClient(fl.client.NumPyClient):
    """Flower client wrapper for federated learning.

    Manages local training with secure update protection (norm clipping + DP noise).

    Args:
        client_id: Unique client identifier
        X_train: Training features (torch.Tensor)
        y_train: Training labels (torch.Tensor)
        X_val: Validation features
        y_val: Validation labels
        model_config: Model architecture configuration
        train_config: Training hyperparameters
        privacy_config: Differential privacy configuration
        padding_mask: Boolean mask for attention (for heterogeneous features)
    """

    def __init__(
        self,
        client_id: str,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        model_config: Dict[str, Any],
        train_config: Dict[str, Any],
        privacy_config: Dict[str, Any] = None,
        padding_mask: Optional[torch.Tensor] = None
    ):
        from typing import Optional

        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.model_config = model_config
        self.train_config = train_config
        self.privacy_config = privacy_config or {"enabled": False}
        self.padding_mask = padding_mask

        model_type = model_config.get("model_type", "mlp")
        input_dim = X_train.shape[1]

        self.model = create_model(input_dim, model_config, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        logger.info(f"Client {client_id}: model={model_type}, input_dim={input_dim}, "
                  f"active_features={self.padding_mask.sum().item() if self.padding_mask is not None else input_dim}")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return [p.cpu().numpy() for p in self.model.get_parameters()]

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        original_params = [torch.tensor(p) for p in parameters]
        self.model.set_parameters(original_params)

        train_cfg = self.train_config.copy()
        train_cfg.update({
            "epochs": config.get("local_epochs", self.train_config.get("epochs", 3)),
            "lr": config.get("lr", self.train_config.get("lr", 0.001)),
            "mu": config.get("mu", self.train_config.get("mu", 0.0)),
            "round": config.get("round", 0),
            "optimizer": config.get("optimizer", "adamw"),
        })

        updated_params, metrics = train_one_round(
            self.model,
            self.X_train,
            self.y_train,
            train_cfg,
            self.device,
            padding_mask=self.padding_mask,
            X_val=self.X_val,
            y_val=self.y_val,
        )

        updated_params_np = [p.cpu().numpy() for p in updated_params]

        if self.privacy_config.get("enabled", False):
            ref_params_np = [p.cpu().numpy() for p in original_params]
            protected_params, audit_log = protect_update(
                updated_params_np,
                ref_params_np,
                self.privacy_config
            )
            metrics["audit"] = audit_log
            updated_params_np = protected_params

        num_examples = len(self.X_train)
        return updated_params_np, num_examples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        self.model.set_parameters([torch.tensor(p) for p in parameters])

        eval_cfg = {"batch_size": self.model_config.get("eval_batch_size", 512)}
        metrics = evaluate_client(
            self.model,
            self.X_val,
            self.y_val,
            eval_cfg,
            self.device,
            padding_mask=self.padding_mask
        )

        num_examples = len(self.X_val)
        return float(metrics.get("roc_auc", 0.5) * -1 + 1), num_examples, metrics
def create_client_fn(
    client_id: str,
    data_dir: str,
    artifacts_dir: str,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    privacy_config: Dict[str, Any] = None
):
    import pandas as pd
    from pathlib import Path
    from src.data.preprocess import ClientPreprocessor

    train_df = pd.read_csv(f"{data_dir}/{client_id}_train.csv")
    val_df = pd.read_csv(f"{data_dir}/{client_id}_val.csv")

    prep_path = Path(artifacts_dir) / "preprocessors" / f"{client_id}_preprocessor.pkl"
    preprocessor = ClientPreprocessor.load(str(prep_path))

    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)

    padding_mask = preprocessor.get_padding_mask() if hasattr(preprocessor, "get_padding_mask") else None
    if padding_mask is not None and not isinstance(padding_mask, torch.Tensor):
        padding_mask = torch.tensor(padding_mask, dtype=torch.bool)

    return FederatedClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config=model_config,
        train_config=train_config,
        privacy_config=privacy_config,
        padding_mask=padding_mask
    )

