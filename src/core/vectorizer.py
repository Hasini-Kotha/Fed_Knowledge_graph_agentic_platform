"""Dynamic Vectorizer — Transforms raw DataFrames into fixed-size PyTorch tensors.

This is the Universal Transformer of the platform. It takes any DataFrame and a
MetadataMapper, then produces a torch.Tensor of shape (batch_size, vector_size).

Key features:
- Automatic feature alignment to global indices
- Numeric: StandardScaler with configurable imputation
- Categorical: OneHotEncoder with configurable imputation
- High-cardinality handling: Top-K + "other" bucket for oversized categoricals
- Zero-padding for missing global indices (gap logic)
- Boolean masking for attention (padded positions marked False)
- PCA reduction for oversized feature sets
- Save/load fitted parameters for inference consistency
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from src.core.metadata_engine import MetadataMapper, FeatureType, ImputationStrategy
from src.core.contract import VectorContract, HighCardinalityStrategy

logger = logging.getLogger(__name__)


class TopKEncoder:
    """One-hot encoder that keeps only the top-K most frequent categories.

    Categories beyond the top-K are mapped to a single "other" column.
    This prevents high-cardinality features from exceeding allocated space.

    Args:
        n_categories: Number of top categories to keep (1 slot is reserved for "other").
    """

    def __init__(self, n_categories: int):
        self.n_categories = n_categories
        self.top_categories_: List[Any] = []
        self.feature_names_out_: List[str] = []

    def fit(self, X, y=None):
        if hasattr(X, "flatten"):
            X = X.flatten()
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            X = X.values
        counts = pd.Series(X).value_counts()
        self.top_categories_ = counts.head(self.n_categories).index.tolist()
        self.feature_names_out_ = [f"top_{c}" for c in self.top_categories_]
        if len(counts) > self.n_categories:
            self.feature_names_out_.append("other")
        return self

    def transform(self, X):
        if hasattr(X, "flatten"):
            X = X.flatten()
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            X = X.values
        n_samples = len(X)
        n_cols = len(self.feature_names_out_)
        result = np.zeros((n_samples, n_cols), dtype=np.float32)

        for j, cat in enumerate(self.top_categories_):
            result[:, j] = (X == cat).astype(np.float32)

        if n_cols > len(self.top_categories_):
            known = np.zeros(n_samples, dtype=bool)
            for cat in self.top_categories_:
                known |= (X == cat)
            result[:, -1] = (~known).astype(np.float32)

        return result

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class DynamicVectorizer:
    """Transforms raw tabular data into fixed-size PyTorch tensors.

    The vectorizer is fit on training data and then reused for validation, test,
    and inference. It ensures that all clients produce tensors compatible with
    the global model architecture.

    Args:
        vector_size: Fixed output dimension (default: 128)
        scaler_type: 'standard' or 'robust' for numeric scaling

    Example:
        mapper = MetadataMapper("configs/neobank_a_mapping.json")
        vectorizer = DynamicVectorizer(vector_size=128)
        result = vectorizer.fit_transform(df, mapper)
        # result["data"].shape = (n_samples, 128)
        # result["mask"].shape = (128,) — True where features exist
    """

    def __init__(
        self,
        vector_size: int = 128,
        scaler_type: str = "standard"
    ):
        self.vector_size = vector_size
        self.scaler_type = scaler_type
        self._pipeline: Optional[ColumnTransformer] = None
        self._is_fitted = False
        self._feature_dim_before_vector: int = 0
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._column_to_global_index: Dict[str, int] = {}
        self._pca: Optional[PCA] = None
        self._mapping_summary: Dict[str, Any] = {}
        self._active_mask: Optional[np.ndarray] = None
        self._mapped_indices: List[int] = []
        self._categorical_encoders: Dict[str, Any] = {}

    def _build_pipeline(self, mapper: MetadataMapper) -> ColumnTransformer:
        """Build the sklearn preprocessing pipeline based on the mapper."""
        numeric_cols = mapper.get_numeric_columns()
        categorical_cols = mapper.get_categorical_columns()

        self._numeric_columns = numeric_cols
        self._categorical_columns = categorical_cols
        self._column_to_global_index = mapper.get_local_to_index()
        self._categorical_encoders = {}

        allocations = mapper.get_categorical_allocation()
        transformers = []

        if numeric_cols:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numeric', numeric_transformer, numeric_cols))

        for col_name in categorical_cols:
            allocated = allocations.get(col_name, 1)
            feature_mapping = next(
                (f for f in mapper.feature_mappings if f.local_name == col_name),
                None
            )

            if feature_mapping is not None and feature_mapping.max_cardinality is not None:
                max_cats = min(feature_mapping.max_cardinality, allocated) - 1
                if max_cats < 1:
                    max_cats = 1
                encoder = TopKEncoder(n_categories=max_cats)
                self._categorical_encoders[col_name] = encoder
                encoder_step = ('encoder', encoder)
            else:
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self._categorical_encoders[col_name] = encoder
                encoder_step = ('onehot', encoder)

            col_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                encoder_step
            ])
            transformers.append((f'cat_{col_name}', col_transformer, [col_name]))

        if not transformers:
            raise ValueError("No features found in mapping. Check mapping.json")

        pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        return pipeline

    def fit_transform(
        self,
        df: pd.DataFrame,
        mapper: MetadataMapper
    ) -> Dict[str, Any]:
        """Fit the vectorizer on training data and transform it.

        Args:
            df: Training DataFrame with features and target
            mapper: MetadataMapper for this client

        Returns:
            Dict with keys:
                "data": torch.Tensor of shape (n_samples, vector_size)
                "mask": torch.Tensor of shape (vector_size,) — True where features exist
                "y": numpy array of labels
        """
        if not self._is_fitted:
            self._pipeline = self._build_pipeline(mapper)
            X_processed = self._pipeline.fit_transform(df)
            self._is_fitted = True
        else:
            X_processed = self._pipeline.transform(df)

        self._feature_dim_before_vector = X_processed.shape[1]

        X_aligned, mask = self._align_to_global_indices(X_processed, mapper)
        X_vector = self._enforce_vector_size(X_aligned)

        y = df[mapper.get_target_column()].values.astype(np.float32)
        X_tensor = torch.tensor(X_vector, dtype=torch.float32)

        self._mapping_summary = mapper.summary()
        self._mapping_summary['processed_dim'] = self._feature_dim_before_vector
        self._mapping_summary['output_dim'] = self.vector_size

        logger.info(
            f"Vectorizer: {X_processed.shape} -> {X_tensor.shape} "
            f"(mask: {mask.sum()}/{len(mask)} active, y: {y.shape})"
        )

        return {
            "data": X_tensor,
            "mask": torch.tensor(mask, dtype=torch.bool),
            "y": y,
        }

    def transform(
        self,
        df: pd.DataFrame,
        mapper: Optional[MetadataMapper] = None
    ) -> Dict[str, Any]:
        """Transform unseen data using the fitted vectorizer.

        Args:
            df: DataFrame to transform
            mapper: Optional mapper (uses the one from fit_transform if not provided)

        Returns:
            Dict with keys "data" and "mask"
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Vectorizer must be fitted before transform. Call fit_transform() first.")

        X_processed = self._pipeline.transform(df)
        X_aligned, _ = self._align_to_global_indices(X_processed, mapper)
        X_vector = self._enforce_vector_size(X_aligned)

        return {
            "data": torch.tensor(X_vector, dtype=torch.float32),
            "mask": torch.tensor(self._active_mask, dtype=torch.bool) if self._active_mask is not None else torch.ones(self.vector_size, dtype=torch.bool),
        }

    def _align_to_global_indices(
        self,
        X_processed: np.ndarray,
        mapper: Optional[MetadataMapper]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align processed features to their global index positions.

        Returns:
            (aligned_matrix, boolean_mask)
        """
        if mapper is None:
            mask = np.ones(self.vector_size, dtype=bool)
            return X_processed, mask

        feature_mapping = mapper.feature_mappings
        has_categorical = len(mapper.get_categorical_columns()) > 0

        if not has_categorical:
            return self._align_numeric_only(X_processed, feature_mapping)

        numeric_features = [f for f in feature_mapping if f.feature_type == FeatureType.NUMERIC]
        categorical_features = [f for f in feature_mapping if f.feature_type == FeatureType.CATEGORICAL]

        n_samples = X_processed.shape[0]
        n_numerics = len(numeric_features)

        col_starts = {}
        current_pos = 0
        for name, _, cols in self._pipeline.transformers:
            if name == 'numeric':
                current_pos += 1
                continue
            if name.startswith('cat_'):
                transformer = self._pipeline.named_transformers_.get(name)
                if transformer is not None:
                    encoder_step = transformer.named_steps.get('encoder') or transformer.named_steps.get('onehot')
                    if encoder_step is not None:
                        n_out = len(encoder_step.get_feature_names_out())
                    else:
                        n_out = 1
                    col_starts[name] = (current_pos, n_out)
                    current_pos += n_out

        result = np.zeros((n_samples, self.vector_size), dtype=np.float32)
        mask = np.zeros(self.vector_size, dtype=bool)
        mapped_indices = []

        for i, f in enumerate(numeric_features):
            if f.global_index < self.vector_size:
                result[:, f.global_index] = X_processed[:, i]
                mask[f.global_index] = True
                mapped_indices.append(f.global_index)

        for f in categorical_features:
            encoder_name = f'cat_{f.local_name}'
            if encoder_name not in col_starts:
                continue
            start_pos, n_out = col_starts[encoder_name]

            for j in range(n_out):
                global_pos = f.global_index + j
                if global_pos < self.vector_size:
                    result[:, global_pos] = X_processed[:, start_pos + j]
                    mask[global_pos] = True
                    if global_pos not in mapped_indices:
                        mapped_indices.append(global_pos)

        self._active_mask = mask
        self._mapped_indices = sorted(mapped_indices)

        return result, mask

    def _align_numeric_only(
        self,
        X_processed: np.ndarray,
        feature_mapping
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align numeric-only features to global indices."""
        n_samples = X_processed.shape[0]
        result = np.zeros((n_samples, self.vector_size), dtype=np.float32)
        mask = np.zeros(self.vector_size, dtype=bool)
        mapped_indices = []

        for i, f in enumerate(feature_mapping):
            if i < X_processed.shape[1] and f.global_index < self.vector_size:
                result[:, f.global_index] = X_processed[:, i]
                mask[f.global_index] = True
                mapped_indices.append(f.global_index)

        self._active_mask = mask
        self._mapped_indices = sorted(mapped_indices)

        return result, mask

    def _enforce_vector_size(self, X: np.ndarray) -> np.ndarray:
        """Ensure output is exactly vector_size dimensions."""
        n_samples = X.shape[0]
        current_dim = X.shape[1]

        if current_dim == self.vector_size:
            return X

        if current_dim < self.vector_size:
            padding = np.zeros((n_samples, self.vector_size - current_dim), dtype=np.float32)
            return np.hstack([X, padding])

        if current_dim > self.vector_size:
            if self._pca is None:
                self._pca = PCA(n_components=self.vector_size, random_state=42)
                X_reduced = self._pca.fit_transform(X)
            else:
                X_reduced = self._pca.transform(X)
            logger.info(f"PCA reduction: {current_dim} -> {self.vector_size} dimensions")
            return X_reduced

        return X

    def save(self, path: str):
        """Save the fitted vectorizer to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'pipeline': self._pipeline,
            'is_fitted': self._is_fitted,
            'vector_size': self.vector_size,
            'scaler_type': self.scaler_type,
            'feature_dim_before_vector': self._feature_dim_before_vector,
            'numeric_columns': self._numeric_columns,
            'categorical_columns': self._categorical_columns,
            'column_to_global_index': self._column_to_global_index,
            'pca': self._pca,
            'mapping_summary': self._mapping_summary,
            'active_mask': self._active_mask,
            'mapped_indices': self._mapped_indices,
            'categorical_encoders': self._categorical_encoders,
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Vectorizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DynamicVectorizer":
        """Load a fitted vectorizer from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        vectorizer = cls(
            vector_size=state['vector_size'],
            scaler_type=state['scaler_type']
        )
        vectorizer._pipeline = state['pipeline']
        vectorizer._is_fitted = state['is_fitted']
        vectorizer._feature_dim_before_vector = state['feature_dim_before_vector']
        vectorizer._numeric_columns = state['numeric_columns']
        vectorizer._categorical_columns = state['categorical_columns']
        vectorizer._column_to_global_index = state['column_to_global_index']
        vectorizer._pca = state['pca']
        vectorizer._mapping_summary = state['mapping_summary']
        vectorizer._active_mask = state.get('active_mask')
        vectorizer._mapped_indices = state.get('mapped_indices', [])
        vectorizer._categorical_encoders = state.get('categorical_encoders', {})

        logger.info(f"Vectorizer loaded from {path}")

        return vectorizer

    def get_feature_dim(self) -> int:
        return self.vector_size

    def get_mapping_summary(self) -> Dict[str, Any]:
        return self._mapping_summary

    def get_active_mask(self) -> Optional[np.ndarray]:
        """Return the boolean mask of active (non-padded) feature indices."""
        return self._active_mask