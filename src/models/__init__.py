"""Models module — LiteFraudNet: lightweight dual-head ResNet for FL + KG."""

from src.models.Fed_model import LiteFraudNet, create_model

__all__ = ["LiteFraudNet", "create_model"]