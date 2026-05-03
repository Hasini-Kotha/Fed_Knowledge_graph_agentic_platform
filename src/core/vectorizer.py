"""Dynamic Vectorizer — Transforms raw DataFrames into fixed-size PyTorch tensors.

This is the Universal Transformer of the platform. It takes any DataFrame and a
MetadataMapper, then produces a torch.Tensor of shape (batch_size, vector_size).

Key features:
- Automatic feature alignment to global indices
- Numeric: StandardScaler with configurable imputation
- Categorical: OneHotEncoder with configurable imputation
- Zero-padding for missing global indices (gap logic)
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
from src.core.contract import VectorContract

logger = logging.getLogger(__name__)


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
        X_tensor, y = vectorizer.fit_transform(df, mapper)
        # X_tensor.shape = (n_samples, 128)
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
    
    def _build_pipeline(self, mapper: MetadataMapper) -> ColumnTransformer:
        """Build the sklearn preprocessing pipeline based on the mapper.
        
        Args:
            mapper: MetadataMapper instance with feature definitions
            
        Returns:
            Fitted ColumnTransformer pipeline
        """
        numeric_cols = mapper.get_numeric_columns()
        categorical_cols = mapper.get_categorical_columns()
        
        self._numeric_columns = numeric_cols
        self._categorical_columns = categorical_cols
        self._column_to_global_index = mapper.get_local_to_index()
        
        transformers = []
        
        if numeric_cols:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numeric', numeric_transformer, numeric_cols))
        
        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('categorical', categorical_transformer, categorical_cols))
        
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
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Fit the vectorizer on training data and transform it.
        
        Args:
            df: Training DataFrame with features and target
            mapper: MetadataMapper for this client
            
        Returns:
            (X_tensor, y_array) where X_tensor.shape = (n_samples, vector_size)
        """
        if not self._is_fitted:
            self._pipeline = self._build_pipeline(mapper)
            X_processed = self._pipeline.fit_transform(df)
            self._is_fitted = True
        else:
            X_processed = self._pipeline.transform(df)
        
        self._feature_dim_before_vector = X_processed.shape[1]
        
        X_aligned = self._align_to_global_indices(X_processed, mapper)
        X_vector = self._enforce_vector_size(X_aligned)
        
        y = df[mapper.get_target_column()].values.astype(np.float32)
        X_tensor = torch.tensor(X_vector, dtype=torch.float32)
        
        self._mapping_summary = mapper.summary()
        self._mapping_summary['processed_dim'] = self._feature_dim_before_vector
        self._mapping_summary['output_dim'] = self.vector_size
        
        logger.info(
            f"Vectorizer: {X_processed.shape} -> {X_tensor.shape} "
            f"(y: {y.shape})"
        )
        
        return X_tensor, y
    
    def transform(
        self,
        df: pd.DataFrame,
        mapper: Optional[MetadataMapper] = None
    ) -> torch.Tensor:
        """Transform unseen data using the fitted vectorizer.
        
        Args:
            df: DataFrame to transform
            mapper: Optional mapper (uses the one from fit_transform if not provided)
            
        Returns:
            X_tensor of shape (n_samples, vector_size)
        """
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Vectorizer must be fitted before transform. Call fit_transform() first.")
        
        if mapper is None:
            X_processed = self._pipeline.transform(df)
        else:
            X_processed = self._pipeline.transform(df)
        
        X_aligned = self._align_to_global_indices(X_processed, mapper if mapper else None)
        X_vector = self._enforce_vector_size(X_aligned)
        
        return torch.tensor(X_vector, dtype=torch.float32)
    
    def _align_to_global_indices(
        self,
        X_processed: np.ndarray,
        mapper: Optional[MetadataMapper]
    ) -> np.ndarray:
        """Align processed features to their global index positions.
        
        If the mapper defines global indices that don't match sequential 0,1,2...,
        this method places each feature at its correct global index position,
        leaving zeros for unmapped indices (gap logic).
        
        Args:
            X_processed: Output from ColumnTransformer
            mapper: MetadataMapper (can be None for sequential alignment)
            
        Returns:
            Aligned feature matrix
        """
        if mapper is None:
            return X_processed
        
        feature_mapping = mapper.feature_mappings
        has_categorical = len(mapper.get_categorical_columns()) > 0
        
        if not has_categorical:
            return self._align_numeric_only(X_processed, feature_mapping)
        
        numeric_features = [f for f in feature_mapping if f.feature_type == FeatureType.NUMERIC]
        categorical_features = [f for f in feature_mapping if f.feature_type == FeatureType.CATEGORICAL]
        
        n_samples = X_processed.shape[0]
        n_numerics = len(numeric_features)
        n_cats_before_onehot = len(categorical_features)
        
        ohe = self._pipeline.named_transformers_.get('categorical')
        if ohe is not None:
            onehot = ohe.named_steps['onehot']
            cat_feature_names = onehot.get_feature_names_out(categorical_features)
        else:
            cat_feature_names = []
        
        result = np.zeros((n_samples, self.vector_size), dtype=np.float32)
        
        col_offset = 0
        for i, f in enumerate(numeric_features):
            result[:, f.global_index] = X_processed[:, i]
            col_offset += 1
        
        for i, f in enumerate(categorical_features):
            cat_start = n_numerics + i
            start_name = f"{f.local_name}_"
            cat_indices = [
                j for j, name in enumerate(cat_feature_names)
                if name.startswith(start_name)
            ]
            for j, cat_idx in enumerate(cat_indices):
                global_pos = f.global_index + j
                if global_pos < self.vector_size:
                    result[:, global_pos] = X_processed[:, n_numerics + cat_idx]
        
        return result
    
    def _align_numeric_only(
        self,
        X_processed: np.ndarray,
        feature_mapping
    ) -> np.ndarray:
        """Align numeric-only features to global indices."""
        n_samples = X_processed.shape[0]
        result = np.zeros((n_samples, self.vector_size), dtype=np.float32)
        
        for i, f in enumerate(feature_mapping):
            if i < X_processed.shape[1] and f.global_index < self.vector_size:
                result[:, f.global_index] = X_processed[:, i]
        
        return result
    
    def _enforce_vector_size(self, X: np.ndarray) -> np.ndarray:
        """Ensure output is exactly vector_size dimensions.
        
        - If fewer: zero-pad
        - If more: apply PCA reduction
        - If exact: return as-is
        
        Args:
            X: Input feature matrix
            
        Returns:
            Feature matrix of shape (n_samples, vector_size)
        """
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
        """Save the fitted vectorizer to disk.
        
        Args:
            path: File path for the pickle file
        """
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
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Vectorizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DynamicVectorizer":
        """Load a fitted vectorizer from disk.
        
        Args:
            path: File path of the pickle file
            
        Returns:
            Fitted DynamicVectorizer instance
        """
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
        
        logger.info(f"Vectorizer loaded from {path}")
        
        return vectorizer
    
    def get_feature_dim(self) -> int:
        """Return the output vector dimension."""
        return self.vector_size
    
    def get_mapping_summary(self) -> Dict[str, Any]:
        """Return summary of the mapping used during fitting."""
        return self._mapping_summary