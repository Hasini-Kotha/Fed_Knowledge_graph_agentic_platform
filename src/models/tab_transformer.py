"""Tabular Transformer (TabFT) — Professional-grade model for tabular fintech.

Implements a Transformer-based architecture with feature-level attention masking
to handle zero-padded features from heterogeneous clients.

Architecture:
    Input (128-dim) → Feature Embedding → Transformer Encoder (with key_padding_mask) → Pooling → MLP Head → Output

Key design decisions:
    1. Features are treated as "tokens" — no positional encoding since tabular data
       has no natural sequence order.
    2. Key-padding mask ensures zero-padded features (from clients with fewer columns)
       receive zero attention weight, preventing them from diluting the model.
    3. Numerical features are linearly projected to d_model; categorical features
       use nn.Embedding with a learned "missing" vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple


class FeatureEmbedding(nn.Module):
    """Projects each feature dimension into the transformer embedding space.

    For numerical features: Linear projection (feature_dim -> d_model)
    For categorical features: nn.Embedding with padding_idx=0 for missing values
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        categorical_indices: Optional[List[int]] = None,
        cat_cardinalities: Optional[Dict[int, int]] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.categorical_indices = categorical_indices or []

        self.numerical_proj = nn.Linear(1, d_model)

        self.categorical_embeddings = nn.ModuleDict()
        if cat_cardinalities:
            for idx, cardinality in cat_cardinalities.items():
                self.categorical_embeddings[str(idx)] = nn.Embedding(
                    num_embeddings=cardinality + 1,
                    embedding_dim=d_model,
                    padding_idx=0
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        numerical_embed = self.numerical_proj(x.unsqueeze(-1))

        for idx_str, emb in self.categorical_embeddings.items():
            idx = int(idx_str)
            cat_vals = x[:, idx].long()
            cat_embed = emb(cat_vals)
            numerical_embed[:, idx] = cat_embed

        return numerical_embed


class TabularTransformer(nn.Module):
    """Tabular Transformer with attention masking for heterogeneous features.

    Handles zero-padded features by applying a key_padding_mask during
    multi-head self-attention, ensuring padded positions receive -inf attention
    scores and contribute zero after softmax.

    Args:
        input_dim: Dimension of input vector (global contract size, default 128)
        d_model: Transformer embedding dimension (default 64)
        nhead: Number of attention heads (default 4)
        num_layers: Number of transformer encoder layers (default 2)
        dim_feedforward: Feedforward dimension (default 128)
        dropout: Dropout rate (default 0.2)
        num_classes: Number of output classes (1 for binary)
        categorical_indices: List of indices that are categorical
        cat_cardinalities: Dict mapping categorical index -> cardinality
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        num_classes: int = 1,
        categorical_indices: Optional[List[int]] = None,
        cat_cardinalities: Optional[Dict[int, int]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        self.feature_embedding = FeatureEmbedding(
            input_dim=input_dim,
            d_model=d_model,
            categorical_indices=categorical_indices,
            cat_cardinalities=cat_cardinalities
        )

        self.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional attention masking.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            padding_mask: Boolean tensor of shape (input_dim,) where True = active,
                         False = padded. Converted to key_padding_mask for attention.

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        embedded = self.feature_embedding(x)
        embedded = self.input_norm(embedded)

        if padding_mask is not None:
            if padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0).expand(batch_size, -1)

            key_padding_mask = ~padding_mask
            output = self.transformer(embedded, src_key_padding_mask=key_padding_mask)
        else:
            output = self.transformer(embedded)

        pooled = output.mean(dim=1)
        logits = self.output_head(pooled)

        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(x, padding_mask)
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

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.network(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract learned hidden-layer representation (before output head).

        Returns the output of the second-to-last layer — a semantically rich
        embedding that the model learned to discriminate fraud.  Used by the KG
        builder to compute behaviorally meaningful similarity edges instead of
        raw PCA features.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Embedding tensor of shape (N, hidden_dims[-1]).
        """
        # Run all layers except the very last Linear (output head)
        # network = [Linear, BN, ReLU, Dropout, ..., Linear(output)]
        layers = list(self.network.children())
        for layer in layers[:-1]:  # skip final Linear
            x = layer(x)
        return x  # shape: (N, hidden_dims[-1])  e.g. (N, 32)

    def predict_proba(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            dropout=config.get("dropout", 0.2),
            categorical_indices=config.get("categorical_indices"),
            cat_cardinalities=config.get("cat_cardinalities"),
        )
    else:
        return TabularMLP(
            input_dim=input_dim,
            hidden_dims=config.get("hidden_dims", [64, 32]),
            dropout=config.get("dropout", 0.2)
        ) 
