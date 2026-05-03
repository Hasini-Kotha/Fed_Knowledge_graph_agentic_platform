"""Global contract definitions for the Federated Learning Platform.

This module defines the immutable constants and interfaces that all
components must adhere to, ensuring cross-domain compatibility.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class FeatureType(str, Enum):
    """Supported feature types for the metadata engine."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"


class ImputationStrategy(str, Enum):
    """Supported imputation strategies for missing values."""
    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    CONSTANT = "constant"
    DROP = "drop"


class AggregationStrategy(str, Enum):
    """Supported FL aggregation strategies."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class VectorContract:
    """Defines the fixed-size input tensor contract for the global model.
    
    All clients must produce tensors of shape (batch_size, vector_size).
    If a client has fewer features than vector_size, the remainder is zero-padded.
    If a client has more features, PCA reduction is applied.
    """
    vector_size: int = 128
    dtype: str = "float32"
    allow_padding: bool = True
    allow_reduction: bool = True
    
    def validate(self, actual_size: int) -> bool:
        if actual_size > self.vector_size and not self.allow_reduction:
            return False
        return True


@dataclass
class PrivacyConfig:
    """Differential privacy configuration for secure weight updates."""
    enabled: bool = False
    max_norm: float = 1.0
    noise_multiplier: float = 0.0
    delta: float = 1e-5


@dataclass
class FedProxConfig:
    """FedProx aggregation configuration."""
    mu: float = 0.01  # Proximal term coefficient
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 3
    min_evaluate_clients: int = 3
    min_available_clients: int = 3


@dataclass
class TrimmedMeanConfig:
    """Trimmed Mean aggregation configuration for Byzantine fault tolerance."""
    beta: float = 0.1  # Trim 10% from each end
    max_byzantine_clients: int = 0


DEFAULT_VECTOR_CONTRACT = VectorContract(vector_size=128)
DEFAULT_PRIVACY_CONFIG = PrivacyConfig()
DEFAULT_FEDPROX_CONFIG = FedProxConfig()
DEFAULT_TRIMMED_MEAN_CONFIG = TrimmedMeanConfig()