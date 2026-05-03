"""Unified Training Engine — Training loop, evaluation, and checkpoint management.

Provides a consistent training interface for both local baseline training
and federated client training. Handles batching, loss computation, backprop,
and metric calculation.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from src.models import TabularTransformer, TabularMLP, create_model

logger = logging.getLogger(__name__)


class ClientDataset:
    """PyTorch Dataset for client data."""
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute binary cross-entropy loss with logits.
    
    Args:
        y_pred: Raw logits from model
        y_true: Ground truth labels
        
    Returns:
        Scalar loss value
    """
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


import torch.nn.functional as F


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mu: float = 0.0
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: CPU or CUDA
        mu: FedProx proximal term coefficient (0.0 = no proximal term)
        
    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    global_params = None
    if mu > 0:
        global_params = [p.clone().detach() for p in model.parameters()]
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        loss = F.binary_cross_entropy_with_logits(y_pred, y_batch)
        
        if mu > 0:
            proximal_term = 0.0
            for local_param, global_param in zip(model.parameters(), global_params):
                proximal_term += (local_param - global_param).norm(2).pow(2)
            loss += (mu / 2) * proximal_term
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    
    return {"train_loss": avg_loss}


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 512
) -> Dict[str, float]:
    """Evaluate model on validation/test data.
    
    Args:
        model: PyTorch model
        X: Input features
        y: Ground truth labels
        device: CPU or CUDA
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            y_batch = y[i:i+batch_size]
            
            y_pred = model(X_batch)
            probs = torch.sigmoid(y_pred).squeeze().cpu().numpy()
            
            all_probs.extend(probs.flatten())
            all_labels.extend(y_batch.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    y_pred_binary = (all_probs >= 0.5).astype(int)
    
    metrics = {
        "loss": float(F.binary_cross_entropy_with_logits(
            torch.tensor(all_probs),
            torch.tensor(all_labels, dtype=torch.float32)
        )),
        "roc_auc": float(roc_auc_score(all_labels, all_probs)),
        "pr_auc": float(average_precision_score(all_labels, all_probs)),
        "f1": float(f1_score(all_labels, y_pred_binary, zero_division=0)),
        "precision": float(precision_score(all_labels, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(all_labels, y_pred_binary, zero_division=0)),
        "accuracy": float((y_pred_binary == all_labels).mean()),
    }
    
    return metrics


def predict_proba(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:
    """Generate probability predictions.
    
    Args:
        model: PyTorch model
        X: Input features
        device: CPU or CUDA
        batch_size: Batch size
        
    Returns:
        Array of probabilities
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            y_pred = model(X_batch)
            probs = torch.sigmoid(y_pred).squeeze().cpu().numpy()
            all_probs.extend(probs.flatten())
    
    return np.array(all_probs)


def train_one_round(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    train_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Train model for one round (multiple epochs).
    
    Used by FL clients for local training.
    
    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training labels
        train_config: Training configuration (epochs, batch_size, lr, mu)
        device: CPU or CUDA (auto-detected if None)
        
    Returns:
        (model_parameters, metrics_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    batch_size = train_config.get("batch_size", 256)
    epochs = train_config.get("epochs", 3)
    lr = train_config.get("lr", 0.001)
    mu = train_config.get("mu", 0.0)
    
    dataset = ClientDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        metrics = train_one_epoch(model, train_loader, optimizer, device, mu)
    
    parameters = model.get_parameters()
    metrics["round"] = train_config.get("round", 0)
    
    return parameters, metrics


def evaluate_client(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    eval_config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate model on client validation data.
    
    Args:
        model: PyTorch model
        X_val: Validation features
        y_val: Validation labels
        eval_config: Evaluation config (batch_size)
        device: CPU or CUDA
        
    Returns:
        Metrics dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    batch_size = eval_config.get("batch_size", 512)
    
    return evaluate_model(model, X_val, y_val, device, batch_size)


def save_local_checkpoint(
    model: nn.Module,
    vectorizer,
    metrics: Dict[str, Any],
    path: str
):
    """Save a local model checkpoint.
    
    Args:
        model: PyTorch model
        vectorizer: Fitted DynamicVectorizer
        metrics: Evaluation metrics
        path: Output path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": model.__class__.__name__,
        "input_dim": model.input_dim,
        "metrics": metrics,
        "vectorizer_state": vectorizer,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Local checkpoint saved: {path}")


def load_local_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load a local model checkpoint.
    
    Args:
        path: Checkpoint path
        device: CPU or CUDA
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device)
    logger.info(f"Local checkpoint loaded: {path}")
    
    return checkpoint