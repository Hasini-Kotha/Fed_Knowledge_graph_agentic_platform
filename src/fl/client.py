"""
Flower federated learning client wrapper.
"""

import logging
import pathlib
import numpy as np
import pandas as pd
import flwr as fl
from typing import Dict, Any, Tuple, List

from src.models.mlp import TabularMLP, create_model
from src.models.train_local import train_one_round, evaluate_client
from src.data.preprocess import ClientPreprocessor
from src.fl.secure_update import protect_update

logger = logging.getLogger(__name__)

class FederatedClient(fl.client.NumPyClient):
    """
    Flower client wrapping the local ML pipeline.
    """
    def __init__(
        self, 
        client_id: str, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray, 
        model_config: Dict[str, Any], 
        train_config: Dict[str, Any]
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_config = model_config
        self.train_config = train_config
        
        self.model = create_model(input_dim=X_train.shape[1], config=model_config)
        
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns the current model parameters."""
        return self.model.get_parameters()
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Trains the model locally using the provided global parameters."""
        logger.info(f"[{self.client_id}] fit() called on {len(self.X_train)} samples.")
        
        # Store reference parameters before setting the new ones
        reference_parameters = self.model.get_parameters()
        
        # Set parameters from server
        self.model.set_parameters(parameters)
        
        # Train local model
        updated_parameters, metrics_dict = train_one_round(
            self.model, 
            self.X_train, 
            self.y_train, 
            self.train_config
        )
        metrics_dict["client_id"] = self.client_id
        
        # Protect update before sending to server
        secure_config = self.train_config.get("secure_update", {})
        protected_params, audit_log = protect_update(
            updated_parameters,
            reference_parameters,
            secure_config
        )
        metrics_dict["secure_update_audit"] = str(audit_log) # flower metrics values must be scalars or strings
        
        return protected_params, len(self.X_train), metrics_dict
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluates the model locally using the provided global parameters."""
        logger.info(f"[{self.client_id}] evaluate() called on {len(self.X_val)} samples.")
        
        self.model.set_parameters(parameters)
        
        eval_config = {
            "batch_size": self.train_config.get("eval_batch_size", 512),
            "threshold": self.train_config.get("eval_threshold", 0.5),
            "device": self.train_config.get("device", "cpu")
        }
        
        metrics_dict = evaluate_client(self.model, self.X_val, self.y_val, eval_config)
        metrics_dict["client_id"] = self.client_id
        
        val_loss = metrics_dict.get("val_loss", 0.0)
        
        # Confusion matrix list cannot be natively transported over flower metrics
        # converting to string for transmission
        if "confusion_matrix" in metrics_dict:
            metrics_dict["confusion_matrix"] = str(metrics_dict["confusion_matrix"])
            
        return float(val_loss), len(self.X_val), metrics_dict

def create_client_fn(
    client_id: str, 
    data_dir: str, 
    artifacts_dir: str, 
    model_config: Dict[str, Any], 
    train_config: Dict[str, Any]
) -> FederatedClient:
    """
    Factory function to create a FederatedClient.
    Loads pre-split CSVs, loads the saved preprocessor, and transforms data.
    """
    data_path = pathlib.Path(data_dir)
    train_csv = data_path / f"{client_id}_train.csv"
    val_csv = data_path / f"{client_id}_val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Data files missing for {client_id}.")
        
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    preprocessor_path = pathlib.Path(artifacts_dir) / "preprocessors" / f"{client_id}_preprocessor.pkl"
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            f"Run run_single_baseline.py first to generate preprocessors."
        )
        
    preprocessor = ClientPreprocessor.load(str(preprocessor_path))
    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    
    return FederatedClient(
        client_id=client_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config=model_config,
        train_config=train_config
    )
