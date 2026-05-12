"""Global Model Predictor — Clean inference wrapper for the federated global model.

This module provides a single entry-point for Layer 2 (Prediction Layer).
It loads the saved global model and preprocessor, accepts a raw DataFrame,
and returns it with risk scores and predicted labels attached.

Usage:
    predictor = GlobalModelPredictor.from_artifacts("artifacts")
    scored_df = predictor.predict(raw_df)        # adds 'fraud_risk_score'
    scored_df = predictor.classify(raw_df)        # adds 'fraud_risk_score' + 'predicted_label'
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch

from src.models import create_model
from src.models.train_engine import predict_proba

logger = logging.getLogger(__name__)


class GlobalModelPredictor:
    """Inference wrapper for the federated global model.

    Attributes:
        model: PyTorch model loaded with global weights.
        preprocessor: Fitted ClientPreprocessor for feature transformation.
        model_card: Metadata dict from model_card.json.
        threshold: Optimal classification threshold from model card.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor,
        model_card: Dict[str, Any],
        device: str = "cpu",
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.model_card = model_card
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.threshold = model_card.get("optimal_threshold", 0.5)
        self.input_dim = model_card.get("input_dim", preprocessor.get_feature_dim())

        logger.info(
            "GlobalModelPredictor ready: input_dim=%d, threshold=%.2f, device=%s",
            self.input_dim, self.threshold, self.device,
        )

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str = "artifacts",
        model_config_path: str = "configs/model_config.yaml",
        device: str = "cpu",
    ) -> "GlobalModelPredictor":
        """Load the predictor from saved artifacts.

        Expects:
            artifacts/global_model/model_card.json
            artifacts/global_model/training_history.json  (for weights)
            artifacts/preprocessors/global_preprocessor.pkl  (or client_a fallback)

        Args:
            artifacts_dir: Root artifacts directory.
            model_config_path: Path to model_config.yaml.
            device: 'cpu' or 'cuda'.

        Returns:
            Initialized GlobalModelPredictor.
        """
        import yaml
        from src.data.preprocess import ClientPreprocessor

        artifacts = Path(artifacts_dir)

        # --- Load model config ---
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # --- Load model card ---
        model_card_path = artifacts / "global_model" / "model_card.json"
        if not model_card_path.exists():
            raise FileNotFoundError(f"model_card.json not found at {model_card_path}")

        with open(model_card_path, "r") as f:
            model_card = json.load(f)

        # --- Load preprocessor ---
        global_prep = artifacts / "preprocessors" / "global_preprocessor.pkl"
        client_a_prep = artifacts / "preprocessors" / "client_a_preprocessor.pkl"

        if global_prep.exists():
            preprocessor = ClientPreprocessor.load(str(global_prep))
            logger.info("Loaded global preprocessor: %s", global_prep)
        elif client_a_prep.exists():
            preprocessor = ClientPreprocessor.load(str(client_a_prep))
            logger.warning("Falling back to client_a preprocessor: %s", client_a_prep)
        else:
            raise FileNotFoundError(
                f"No preprocessor found in {artifacts / 'preprocessors'}. "
                "Run the data pipeline and baseline first."
            )

        input_dim = preprocessor.get_feature_dim()

        # --- Create model ---
        model_type = model_config.get("model_type", "mlp")
        model = create_model(input_dim, model_config, model_type)

        # --- Load weights from checkpoint or training history ---
        final_model_path = artifacts / "global_model" / "FINAL_global_model.pt"
        checkpoint_files = sorted(
            (artifacts / "global_model").glob("round_*_checkpoint.pt")
        )

        if final_model_path.exists():
            ckpt = torch.load(str(final_model_path), map_location=device, weights_only=False)
            params = ckpt.get("parameters", ckpt.get("weights", None))
            if params is not None:
                model.set_parameters([torch.tensor(np.array(p)) for p in params])
                logger.info("Loaded weights from FINAL_global_model.pt")
            else:
                model.load_state_dict(ckpt.get("model_state", ckpt))
                logger.info("Loaded state_dict from FINAL_global_model.pt")
        elif checkpoint_files:
            best_ckpt_path = checkpoint_files[-1]
            ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
            params = ckpt.get("parameters", ckpt.get("weights", None))
            if params is not None:
                model.set_parameters([torch.tensor(np.array(p)) for p in params])
            logger.info("Loaded weights from %s", best_ckpt_path.name)
        else:
            logger.warning(
                "No checkpoint found — using randomly initialized model weights. "
                "Run the FL simulation first to generate real weights."
            )

        return cls(model, preprocessor, model_card, device)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a raw DataFrame and add risk scores.

        Args:
            df: Raw transaction DataFrame with feature columns matching mapping.json.

        Returns:
            A copy of df with 'fraud_risk_score' column appended.
        """
        X, y = self.preprocessor.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        risk_scores = predict_proba(self.model, X_tensor, self.device, batch_size=512)

        result = df.copy()
        result["fraud_risk_score"] = risk_scores
        return result

    def classify(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """Run inference and add both risk scores and predicted labels.

        Args:
            df: Raw transaction DataFrame.
            threshold: Classification threshold. Uses optimal_threshold from
                       model_card if not provided.

        Returns:
            A copy of df with 'fraud_risk_score' and 'predicted_label' columns.
        """
        t = threshold if threshold is not None else self.threshold
        result = self.predict(df)
        result["predicted_label"] = (result["fraud_risk_score"] >= t).astype(int)
        return result

    def predict_scores_only(self, df: pd.DataFrame) -> np.ndarray:
        """Return just the risk score array (no DataFrame copy).

        Useful for downstream layers (KG, explainability) that need raw scores.
        """
        X, _ = self.preprocessor.transform(df)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return predict_proba(self.model, X_tensor, self.device, batch_size=512)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the predictor configuration."""
        return {
            "model_type": self.model_card.get("model_type", "unknown"),
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "device": str(self.device),
            "dataset": self.model_card.get("dataset", "unknown"),
            "global_test_pr_auc": self.model_card.get("global_test_pr_auc", 0.0),
            "global_test_roc_auc": self.model_card.get("global_test_roc_auc", 0.0),
            "trained_rounds": self.model_card.get("trained_rounds", 0),
        }

    def __repr__(self) -> str:
        return (
            f"GlobalModelPredictor("
            f"input_dim={self.input_dim}, "
            f"threshold={self.threshold:.2f}, "
            f"device={self.device})"
        )
