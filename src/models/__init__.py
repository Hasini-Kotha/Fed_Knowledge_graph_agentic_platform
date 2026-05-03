"""Tabular Transformer (TabFT) — Professional-grade local model for fintech.

Implements a Transformer-based architecture for tabular data, designed for
high-stakes fintech decision-making (fraud detection, credit risk scoring).

Architecture:
    Input (vector_size) → Embedding → Transformer Encoder → Pooling → Output
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any


class TabularTransformer(nn.Module):
    """Transformer model for tabular binary classification.
    
    Args:
        input_dim: Dimension of input vector (from vectorizer)
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Feedforward dimension in transformer
        dropout: Dropout rate
        num_classes: Number of output classes (2 for binary)
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = x.unsqueeze(1)
        
        x = self.transformer(x)
        x = x.squeeze(1)
        
        logits = self.output_head(x)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_parameters(self) -> List[torch.Tensor]:
        return [p.clone().detach() for p in self.parameters()]
    
    def set_parameters(self, parameters: List[torch.Tensor]):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.data.clone()


class TabularMLP(nn.Module):
    """Fallback MLP model for tabular binary classification.
    
    Simpler architecture used when Transformer is too heavy.
    Architecture: Input → 64 → 32 → Output
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def get_parameters(self) -> List[torch.Tensor]:
        return [p.clone().detach() for p in self.parameters()]
    
    def set_parameters(self, parameters: List[torch.Tensor]):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = new_param.data.clone()


def create_model(
    input_dim: int,
    config: Dict[str, Any],
    model_type: str = "mlp"
) -> nn.Module:
    """Factory function to create the appropriate model.
    
    Args:
        input_dim: Input feature dimension
        config: Model configuration dict
        model_type: 'mlp' or 'transformer'
        
    Returns:
        PyTorch model
    """
    if model_type == "transformer":
        return TabularTransformer(
            input_dim=input_dim,
            d_model=config.get("d_model", 64),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 128),
            dropout=config.get("dropout", 0.1)
        )
    else:
        return TabularMLP(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=config.get("dropout", 0.2)
        )