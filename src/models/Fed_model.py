"""LiteFraudNet — Lightweight Dual-Head ResNet for Federated Fraud Detection.

Architecture:
    Input (64) → Projection → ResBlock 1 → ResBlock 2
                                  ├── Embedding Head → L2 Norm → (N, 32)
                                  └── Classification Head → Sigmoid → (N, 1)

Key design :
    1. LayerNorm — per-sample normalisation; fully compatible
       with DP-SGD and stable across non-IID federated clients.
    2. GELU activations — smooth gradients; no dying-neuron risk on small data.
    3. Two residual blocks — richer representation without
       the vanishing-gradient problem.
    4. Dual-head design — dedicated embedding head (no Dropout) produces clean,
       deterministic vectors for the Knowledge Graph; classification head
       produces fraud probability.
    5. L2-normalised embeddings — all embeddings lie on the unit sphere so
       cosine similarity equals the dot product (fast, numerically precise).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any

class _ResidualBlock(nn.Module):
    """Single residual block for LiteFraudNet.
    Structure:
        x_in → Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm
             ↘━━━━━━━━━━━━━━━━━━━━━━━━ skip ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━↗
        x_out = GELU(x_in + block_output)
    Args:
        dim:     Feature dimension — same for input and output (required for skip).
        dropout: Dropout probability applied between the two linear layers.
    """

    def __init__(self, dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class LiteFraudNet(nn.Module):
    """Lightweight dual-head ResNet for federated fraud detection.

    Designed for hackathon / prototype deployments where training data is
    limited and communication cost must be low, while still generating
    high-quality L2-normalised embeddings for the downstream Knowledge Graph.

    Architecture::

        Input (64)
            └─► Projection: Linear(64→64) + LayerNorm + GELU + Dropout(0.10)
                └─► ResBlock 1: Linear(64→64→64) + skip + GELU
                    └─► ResBlock 2: Linear(64→64→64) + skip + GELU
                        │
                        ├─► Embedding Head:       Linear(64→32) + LN + L2Norm
                        │   OUTPUT: (N, 32)  ← KG cosine similarity
                        │
                        └─► Classification Head:  Linear(64→16) + GELU + Dropout
                                                  Linear(16→1)  + Sigmoid
                            OUTPUT: (N, 1)  ← fraud probability

    Key design decisions:
        1. **LayerNorm** — per-sample normalisation; DP-SGD compatible and
           stable across non-IID client distributions (no running stats).
        2. **GELU** — smooth activation; avoids dying-ReLU on small datasets.
        3. **Two residual blocks** — richer embeddings vs a 1-block version,
           with skip connections preventing vanishing gradients.
        4. **Dedicated embedding head, no Dropout** — deterministic vectors
           at inference time so cosine similarity is stable and reproducible.
        5. **L2 normalisation** — all embeddings on the unit sphere; cosine
           similarity equals the dot product (fast, numerically clean).

    Parameters: ~24 K  (hackathon-friendly)

    Args:
        input_dim:     Dimension of pre-processed input vector (default 64,
                       matching ClientPreprocessor vector_size).
        hidden_dim:    Internal working dimension for projection and ResBlocks.
        embedding_dim: Output dimension of the KG embedding head.
        dropout:       Dropout rate applied inside ResBlocks and clf head.
        num_classes:   1 for binary fraud classification.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.20,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # ── Input Projection ──────────────────────────────────────────────
        # Lifts raw features into the model's working space.
        # Lighter dropout (0.10) to preserve the initial feature signal.
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )

        # ── Two Residual Blocks ───────────────────────────────────────────
        # Each block refines the representation via a learned residual
        # correction; skip connections maintain gradient health throughout.
        self.res_block1 = _ResidualBlock(hidden_dim, dropout=dropout)
        self.res_block2 = _ResidualBlock(hidden_dim, dropout=dropout)

        # ── Embedding Head (KG) ───────────────────────────────────────────
        # No Dropout — we need clean, deterministic vectors for cosine
        # similarity at inference time. L2 normalisation is applied in
        # get_embeddings() to project onto the unit sphere.
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # ── Classification Head ───────────────────────────────────────────
        # Bottleneck (64 → 16 → 1) keeps parameter count low.
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # 64 → 16
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Shared backbone
    # ------------------------------------------------------------------

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Projection + 2 residual blocks → (Batch, hidden_dim)."""
        x = self.projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return raw logits (use with BCEWithLogitsLoss).

        Args:
            x:            (Batch, input_dim) preprocessed feature tensor.
            padding_mask: Unused — kept for API compatibility.

        Returns:
            logits: (Batch, 1)
        """
        return self.classification_head(self._backbone(x))

    # ------------------------------------------------------------------
    # Embedding extraction — Knowledge Graph interface
    # ------------------------------------------------------------------

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised embeddings for Knowledge Graph construction.

        Embeddings lie on the unit hypersphere, so:
            cosine_similarity(a, b) == dot(a, b)

        This makes SIMILAR_PATTERN edge computation fast and geometrically
        accurate. Fraud transactions will cluster tightly, enabling high-quality
        community detection and risk propagation in the KG.

        Args:
            x: (N, input_dim)

        Returns:
            embeddings: (N, embedding_dim) float32, L2-normalised.
        """
        backbone_out = self._backbone(x)
        emb = self.embedding_head(backbone_out)
        return F.normalize(emb, p=2, dim=-1)  # project onto unit sphere

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return fraud probability in [0.0, 1.0]."""
        return torch.sigmoid(self.forward(x))

    # ------------------------------------------------------------------
    # Federated Learning interface — Flower / manual loop compatible
    # ------------------------------------------------------------------

    def get_parameters(self) -> List[torch.Tensor]:
        """Return all trainable parameter tensors (for FL aggregation)."""
        return [p.clone().detach() for p in self.parameters()]

    def set_parameters(self, parameters: List[Any]) -> None:
        """Overwrite model parameters from an aggregated server list."""
        for param, new_param in zip(self.parameters(), parameters):
            if isinstance(new_param, torch.Tensor):
                param.data = new_param.data.clone()
            elif isinstance(new_param, np.ndarray):
                param.data = torch.from_numpy(new_param).clone()
            else:
                param.data = torch.tensor(new_param).clone()

    def __repr__(self) -> str:  # type: ignore[override]
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"LiteFraudNet("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_dim}, "
            f"emb={self.embedding_dim}, "
            f"params={total:,} total / {trainable:,} trainable)"
        )

def create_model(
    input_dim: int,
    config: Dict[str, Any],
    model_type: str = "lite_fraud_net",
) -> nn.Module:
    """Instantiate and return a model for the FL + KG pipeline.

    Currently the only supported model is ``'lite_fraud_net'``
    (:class:`LiteFraudNet`).  Any unrecognised ``model_type`` value also
    returns a ``LiteFraudNet`` so that legacy configs using ``'mlp'`` or
    ``'transformer'`` continue to work without crashing.

    Args:
        input_dim:  Input feature dimension (must match
                    ``ClientPreprocessor.vector_size`` — typically 64).
        config:     Model configuration dict.  Supported keys:

                    - ``hidden_dim``    (int,   default 64)
                    - ``embedding_dim`` (int,   default 32)
                    - ``dropout``       (float, default 0.20)

        model_type: Model identifier string.  Use ``'lite_fraud_net'``.

    Returns:
        :class:`LiteFraudNet` instance ready for training.
    """
    return LiteFraudNet(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 64),
        embedding_dim=config.get("embedding_dim", 32),
        dropout=config.get("dropout", 0.20),
    )
