"""Unified Training Engine — Training loop, evaluation, and checkpoint management.

Provides a consistent training interface for both local baseline training
and federated client training. Handles batching, loss computation, backprop,
metric calculation, FedProx proximal term, and early stopping.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from src.models.Fed_model import create_model

logger = logging.getLogger(__name__)


class ClientDataset:

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    # Validation-based early stopping

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, metric: str = "pr_auc"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value = -float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, metric_value: float, model: nn.Module) -> bool:
        # Return True if training should stop
        if metric_value > self.best_value + self.min_delta:
            self.best_value = metric_value
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    padding_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Train model for one epoch with optional FedProx proximal term.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        optimizer: Optimizer (AdamW recommended for Transformer)
        device: CPU or CUDA
        mu: FedProx proximal term coefficient (0.0 = no proximal term)
        global_params: Global model weights for FedProx (required if mu > 0)
        padding_mask: Boolean mask for attention (optional)

    Returns:
        Dictionary with epoch metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        y_pred = model(X_batch, padding_mask=padding_mask)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_batch)

        if mu > 0 and global_params is not None:
            proximal_term = torch.tensor(0.0, device=device)
            for local_param, global_param in zip(model.parameters(), global_params):
                proximal_term += (local_param - global_param).norm(2).pow(2)
            loss = loss + (mu / 2) * proximal_term

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
    padding_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate model on validation/test data.

    Args:
        model: PyTorch model
        X: Input features
        y: Ground truth labels
        device: CPU or CUDA
        batch_size: Batch size for evaluation
        padding_mask: Boolean mask for attention

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

            y_pred = model(X_batch, padding_mask=padding_mask)
            probs = torch.sigmoid(y_pred).squeeze().cpu().numpy()

            all_probs.extend(probs.flatten())
            if isinstance(y_batch, torch.Tensor):
                all_labels.extend(y_batch.numpy().flatten())
            else:
                all_labels.extend(np.array(y_batch).flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    y_pred_binary = (all_probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(all_labels, all_probs)) if len(np.unique(all_labels)) > 1 else 0.5,
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
    batch_size: int = 512,
    padding_mask: Optional[torch.Tensor] = None
) -> np.ndarray:
    """Generate probability predictions."""
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            y_pred = model(X_batch, padding_mask=padding_mask)
            probs = torch.sigmoid(y_pred).squeeze().cpu().numpy()
            all_probs.extend(probs.flatten())

    return np.array(all_probs)


def train_one_round(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    train_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    padding_mask: Optional[torch.Tensor] = None,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Train model for one round (multiple epochs) with FedProx and early stopping.

    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training labels
        train_config: Training configuration (epochs, batch_size, lr, mu, optimizer)
        device: CPU or CUDA (auto-detected if None)
        padding_mask: Boolean mask for attention
        X_val: Validation features (for early stopping)
        y_val: Validation labels (for early stopping)

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
    optimizer_type = train_config.get("optimizer", "adamw")

    dataset = ClientDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    global_params = None
    if mu > 0:
        global_params = [p.clone().detach() for p in model.parameters()]

    early_stopping = EarlyStopping(
        patience=train_config.get("early_stopping_patience", 5),
        min_delta=1e-4,
        metric="pr_auc"
    )

    best_val_metrics = None

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, mu, global_params, padding_mask
        )

        if X_val is not None and y_val is not None:
            val_metrics = evaluate_model(model, X_val, y_val, device, batch_size=512, padding_mask=padding_mask)

            if early_stopping.step(val_metrics["pr_auc"], model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                early_stopping.load_best(model)
                break

            best_val_metrics = val_metrics

    parameters = model.get_parameters()

    if best_val_metrics:
        parameters, best_val_metrics = parameters, best_val_metrics
    else:
        best_val_metrics = {}

    result_metrics = {**train_metrics, **best_val_metrics}
    result_metrics["round"] = train_config.get("round", 0)

    return parameters, result_metrics


def evaluate_client(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    eval_config: Dict[str, Any],
    device: Optional[torch.device] = None,
    padding_mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate model on client validation data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    batch_size = eval_config.get("batch_size", 512)

    return evaluate_model(model, X_val, y_val, device, batch_size, padding_mask)


def save_local_checkpoint(
    model: nn.Module,
    vectorizer,
    metrics: Dict[str, Any],
    path: str
):
    """Save a local model checkpoint."""
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
    """Load a local model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    logger.info(f"Local checkpoint loaded: {path}")

    return checkpoint