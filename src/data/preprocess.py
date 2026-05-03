"""
Dynamic Vectorizer — Domain-Agnostic Federated Learning Platform
=================================================================

Phase 1 — Global Contract:
    Every client outputs a NumPy array of shape (N, vector_size) regardless
    of the raw feature names, industry, or number of source columns.

Phase 2 — Local Metadata Mapping:
    Column names exist ONLY inside configs/mapping.json.  No Python code
    refers to "Amount", "Time", "blood_pressure", or any domain column.

Phase 3 — Universal Vectorization Pipeline:
    Numeric  → Impute (median) → RobustScaler / StandardScaler
    Categoric→ Impute (mode)   → OneHotEncoder (sparse=False)
    Combined → pad with zeros  (< vector_size)
             → PCA reduction   (> vector_size)
    Output   → float32 ndarray of shape (n_samples, vector_size)

Production notes:
    - Mapping schema is validated on load (Phase 2 contract enforcement).
    - PCA is fitted only on training data; transform path always uses the
      stored PCA to prevent data leakage.
    - Output dtype is always float32 to match PyTorch default precision.
    - Mapping metadata (version, domain, description) is preserved so the
      agentic layer (Phase 5) can explain decisions in human language.
"""

import logging
import pickle
import pathlib
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping Schema Validator (Phase 2 — Contract Enforcement)
# ---------------------------------------------------------------------------

_REQUIRED_MAPPING_KEYS = {"features", "target"}
_REQUIRED_FEATURE_KEYS = {"numeric", "categorical"}


def validate_mapping(mapping: Dict[str, Any]) -> None:
    """
    Validate the structure of a mapping.json dictionary.

    Raises:
        ValueError: if any required key is missing or has the wrong type.
    """
    missing_top = _REQUIRED_MAPPING_KEYS - set(mapping.keys())
    if missing_top:
        raise ValueError(
            f"mapping.json is missing required top-level keys: {missing_top}. "
            f"Expected structure: {{\"features\": {{\"numeric\": [...], \"categorical\": [...]}}, \"target\": \"...\"}}"
        )

    features_block = mapping.get("features", {})
    missing_feat = _REQUIRED_FEATURE_KEYS - set(features_block.keys())
    if missing_feat:
        raise ValueError(
            f"mapping.json['features'] is missing keys: {missing_feat}. "
            f"Both 'numeric' and 'categorical' lists must be present (either can be empty)."
        )

    if not isinstance(features_block.get("numeric"), list):
        raise ValueError("mapping.json['features']['numeric'] must be a list.")
    if not isinstance(features_block.get("categorical"), list):
        raise ValueError("mapping.json['features']['categorical'] must be a list.")
    if not isinstance(mapping.get("target"), str):
        raise ValueError("mapping.json['target'] must be a string (column name of the label).")

    numeric = features_block["numeric"]
    categorical = features_block["categorical"]
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(
            f"mapping.json: columns appear in both 'numeric' and 'categorical': {overlap}. "
            f"Each column must belong to exactly one type."
        )


# ---------------------------------------------------------------------------
# ClientPreprocessor
# ---------------------------------------------------------------------------

class ClientPreprocessor:
    """
    Domain-agnostic Dynamic Vectorizer for federated learning clients.

    Reads a mapping.json file at runtime to determine which columns to use,
    processes them through a universal numerical pipeline, and enforces an
    exact output size (vector_size) via zero-padding or PCA reduction.

    This class is the sole location in the codebase where column names appear
    at runtime — everywhere else in the FL pipeline only sees tensors of shape
    (BatchSize, vector_size).
    """

    # Mapping format version — bump this when the JSON schema changes.
    MAPPING_VERSION: str = "1.0"

    def __init__(
        self,
        mapping_path: str,
        vector_size: int = 64,
        scaler_type: str = "robust",
    ) -> None:
        """
        Args:
            mapping_path: Absolute or relative path to the client's mapping.json.
            vector_size:  Global contract dimension N.  Every output array will
                          have exactly vector_size columns.
            scaler_type:  'robust' (RobustScaler) or 'standard' (StandardScaler).
        """
        self.mapping_path = str(mapping_path)
        self.vector_size = vector_size
        self.scaler_type = scaler_type

        # ------------------------------------------------------------------
        # Phase 2: Load and validate the domain mapping at initialisation.
        # Any malformed mapping.json is caught immediately (fail-fast).
        # ------------------------------------------------------------------
        mapping_file = pathlib.Path(mapping_path)
        if not mapping_file.exists():
            raise FileNotFoundError(
                f"mapping.json not found at: {mapping_path}. "
                f"Every client must supply a local mapping file."
            )

        with open(mapping_file, "r") as f:
            self.mapping: Dict[str, Any] = json.load(f)

        validate_mapping(self.mapping)  # ← Bug #1 fix: fail-fast validation

        self.numeric_cols: List[str]     = self.mapping["features"]["numeric"]
        self.categorical_cols: List[str] = self.mapping["features"]["categorical"]
        self.label_col: str              = self.mapping["target"]
        # Preserve optional metadata for Phase 5 explainability
        self.domain_metadata: Dict[str, Any] = self.mapping.get("metadata", {})

        # ------------------------------------------------------------------
        # Phase 3: Build the universal ColumnTransformer
        # Bug #2 fix: ColumnTransformer crashes when passed NO transformers
        # (e.g. both lists are empty).  We guard with a runtime check here.
        # Bug #3 fix: RobustScaler/StandardScaler now each get their own
        # instance — sharing a single scaler object between the two branches
        # would silently fit it twice and corrupt its internal state.
        # ------------------------------------------------------------------
        if not self.numeric_cols and not self.categorical_cols:
            raise ValueError(
                "mapping.json has no numeric or categorical features defined. "
                "At least one feature column must be listed."
            )

        if scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(
                f"Unknown scaler_type: '{scaler_type}'. Choose 'robust' or 'standard'."
            )

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  scaler),
        ])

        # Bug #4 fix: sparse_output is the correct kwarg for scikit-learn>=1.2.
        # The old 'sparse=True' keyword raises a TypeError in recent versions.
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", numeric_transformer, self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", categorical_transformer, self.categorical_cols))

        # remainder='drop' prevents any non-listed columns (e.g. IDs, timestamps)
        # from leaking into the feature matrix.
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",   # ← Bug #5 fix: prevents silent column leakage
        )

        self.pca: Optional[PCA] = None
        self.feature_names_: Optional[List[str]] = None  # for Phase 5 explainability
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_vector_size(self, X: np.ndarray, is_fit: bool) -> np.ndarray:
        """
        Phase 3 — Padding and Truncating.
        Guarantees the output always has exactly self.vector_size columns.

        Args:
            X:      Processed feature matrix (n_samples, raw_dim).
            is_fit: True during fit_transform (PCA is fitted here).
                    False during transform (stored PCA is used — no data leakage).
        """
        current_dim = X.shape[1]

        if current_dim == self.vector_size:
            return X

        if current_dim < self.vector_size:
            pad_width = self.vector_size - current_dim
            logger.info(
                "Contract padding: %d → %d features (adding %d zero slots).",
                current_dim, self.vector_size, pad_width,
            )
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

        else:  # current_dim > self.vector_size
            logger.info(
                "Contract truncation: %d → %d features via PCA.",
                current_dim, self.vector_size,
            )
            if is_fit:
                # Bug #6 fix: PCA requires n_components ≤ min(n_samples, n_features).
                # Guard prevents a crash when the training split is very small.
                safe_components = min(self.vector_size, X.shape[0], X.shape[1])
                if safe_components < self.vector_size:
                    logger.warning(
                        "PCA n_components reduced from %d to %d due to small batch size.",
                        self.vector_size, safe_components,
                    )
                self.pca = PCA(n_components=safe_components, random_state=42)
                X = self.pca.fit_transform(X)
                # If we had to reduce components, pad up to vector_size
                if safe_components < self.vector_size:
                    X = np.pad(
                        X,
                        ((0, 0), (0, self.vector_size - safe_components)),
                        mode="constant",
                        constant_values=0.0,
                    )
            else:
                if self.pca is None:
                    raise RuntimeError(
                        "PCA has not been fitted. Call fit_transform on training data first."
                    )
                X = self.pca.transform(X)
                # Re-pad if safe_components < vector_size (same guard as fit path)
                if X.shape[1] < self.vector_size:
                    X = np.pad(
                        X,
                        ((0, 0), (0, self.vector_size - X.shape[1])),
                        mode="constant",
                        constant_values=0.0,
                    )

        return X

    def _get_feature_names_after_transform(self) -> List[str]:
        """
        Return generic feature slot names (feature_0 … feature_N-1).
        Phase 1 compliance: the global model only sees slot indices, never domain names.
        """
        return [f"feature_{i}" for i in range(self.vector_size)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the pipeline on training data and transform it.

        Returns:
            X: float32 array of shape (n_samples, vector_size).
            y: label array of shape (n_samples,).
        """
        if self.label_col not in df.columns:
            raise ValueError(
                f"Target column '{self.label_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        all_feature_cols = self.numeric_cols + self.categorical_cols
        missing = [c for c in all_feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns defined in mapping.json are missing from the DataFrame: {missing}. "
                f"Check that the CSV matches the client's mapping.json."
            )

        y = df[self.label_col].values.astype(np.float32)
        logger.info(
            "fit_transform: %d samples | numeric=%d | categorical=%d | target='%s'",
            len(df), len(self.numeric_cols), len(self.categorical_cols), self.label_col,
        )

        X_processed = self.column_transformer.fit_transform(df)
        X_final = self._enforce_vector_size(X_processed, is_fit=True)

        # Always output float32 for PyTorch compatibility
        X_final = X_final.astype(np.float32)

        self.feature_names_ = self._get_feature_names_after_transform()
        self.is_fitted = True

        logger.info(
            "fit_transform done: output shape=%s, dtype=%s",
            X_final.shape, X_final.dtype,
        )
        return X_final, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform unseen data using the already-fitted pipeline.

        Returns:
            X: float32 array of shape (n_samples, vector_size).
            y: label array of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor is not fitted. Call fit_transform on training data first."
            )

        if self.label_col not in df.columns:
            raise ValueError(
                f"Target column '{self.label_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        y = df[self.label_col].values.astype(np.float32)

        X_processed = self.column_transformer.transform(df)
        X_final = self._enforce_vector_size(X_processed, is_fit=False)
        X_final = X_final.astype(np.float32)

        return X_final, y

    def save(self, path: str) -> None:
        """Persist the fitted preprocessor to disk (pickle)."""
        if not self.is_fitted:
            logger.warning("Saving an unfitted preprocessor — this is unusual.")

        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Dynamic Preprocessor saved → %s", path_obj)

    @classmethod
    def load(cls, path: str) -> "ClientPreprocessor":
        """Load a saved preprocessor from disk."""
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Preprocessor checkpoint not found: {path}")

        with open(path_obj, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, expected ClientPreprocessor."
            )

        logger.info("Dynamic Preprocessor loaded ← %s", path_obj)
        return obj

    def get_feature_dim(self) -> int:
        """
        Return the guaranteed output dimension (= vector_size).
        Phase 1 contract: always returns N regardless of raw feature count.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform before get_feature_dim.")
        return self.vector_size

    def get_mapping_summary(self) -> Dict[str, Any]:
        """
        Return a human-readable summary of the domain mapping.
        Used by Phase 5 (Agentic Engine) to translate predictions back to
        domain language without exposing raw column names to the FL layer.
        """
        return {
            "mapping_version": self.MAPPING_VERSION,
            "mapping_path":    self.mapping_path,
            "vector_size":     self.vector_size,
            "numeric_cols":    self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "label_col":       self.label_col,
            "domain_metadata": self.domain_metadata,
            "pca_active":      self.pca is not None,
            "is_fitted":       self.is_fitted,
        }

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"ClientPreprocessor("
            f"scaler='{self.scaler_type}', "
            f"vector_size={self.vector_size}, "
            f"numeric={len(self.numeric_cols)}, "
            f"categorical={len(self.categorical_cols)}, "
            f"pca={'yes' if self.pca else 'no'}, "
            f"status='{status}')"
        )
