"""Federated Learning Client — Flower NumPyClient wrapper.

Each client manages its own local data, preprocessing, and training.
It communicates with the server only via model weights (never raw data).
"""

import logging
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

import flwr as fl

from src.models import create_model
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
        privacy_config: Dict[str, Any] = None
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.model_config = model_config
        self.train_config = train_config
        self.privacy_config = privacy_config or {"enabled": False}
        
        model_type = model_config.get("model_type", "mlp")
        input_dim = X_train.shape[1]
        
        self.model = create_model(input_dim, model_config, model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        logger.info(f"Client {client_id}: model={model_type}, input_dim={input_dim}")
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return [p.cpu().numpy() for p in self.model.get_parameters()]
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.model.set_parameters([torch.tensor(p) for p in parameters])
        
        train_cfg = self.train_config.copy()
        train_cfg.update({
            "epochs": config.get("local_epochs", self.train_config.get("epochs", 3)),
            "lr": config.get("lr", self.train_config.get("lr", 0.001)),
            "mu": config.get("mu", self.train_config.get("mu", 0.0)),
            "round": config.get("round", 0),
        })
        
        updated_params, metrics = train_one_round(
            self.model,
            self.X_train,
            self.y_train,
            train_cfg,
            self.device
        )
        
        updated_params_np = [p.cpu().numpy() for p in updated_params]
        
        if self.privacy_config.get("enabled", False):
            ref_params = [p.cpu().numpy() for p in parameters]
            updated_params_np = protect_update(
                updated_params_np,
                ref_params,
                self.privacy_config
            )
        
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
            self.device
        )
        
        num_examples = len(self.X_val)
        return float(metrics["loss"]), num_examples, metrics


def create_client_fn(
    client_id: str,
    data_dir: str,
    mapping_path: str,
    vectorizer_path: str,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    privacy_config: Dict[str, Any] = None
):
    """Factory function to create a FederatedClient.
    
    Loads CSV data, applies the fitted vectorizer, and creates the client.
    
    Args:
        client_id: Client identifier
        data_dir: Directory containing client CSV files
        mapping_path: Path to mapping.json
        vectorizer_path: Path to fitted vectorizer pickle
        model_config: Model configuration
        train_config: Training configuration
        privacy_config: Privacy configuration
        
    Returns:
        FederatedClient instance
    """
    import pandas as pd
    from src.core.metadata_engine import MetadataMapper
    from src.core.vectorizer import DynamicVectorizer
    
    mapper = MetadataMapper(mapping_path)
    vectorizer = DynamicVectorizer.load(vectorizer_path)
    
    train_df = pd.read_csv(f"{data_dir}/{client_id}_train.csv")
    val_df = pd.read_csv(f"{data_dir}/{client_id}_val.csv")
    
    X_train, y_train = vectorizer.fit_transform(train_df, mapper)
    X_val = vectorizer.transform(val_df, mapper)
    y_val = torch.tensor(val_df[mapper.get_target_column()].values.astype(np.float32))
    
    return FederatedClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config=model_config,
        train_config=train_config,
        privacy_config=privacy_config
    )