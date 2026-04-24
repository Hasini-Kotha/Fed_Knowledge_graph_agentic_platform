"""
Local preprocessing module for domain-agnostic federated learning platform.
Provides the ClientPreprocessor class to handle imputation and scaling for tabular data.
"""

import logging
import pickle
import pathlib
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class ClientPreprocessor:
    """
    Preprocessor for client tabular data in a federated learning setting.
    Handles missing value imputation and feature scaling using an sklearn Pipeline.
    
    Usage example:
        preprocessor = ClientPreprocessor(numeric_cols=['feat1', 'feat2'], label_col='target')
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)
        preprocessor.save('preprocessor.pkl')
    """
    
    def __init__(self, numeric_cols: list[str], label_col: str = "Class", scaler_type: str = "robust"):
        """
        Initialize the preprocessor.
        
        Args:
            numeric_cols: List of column names to apply scaling to.
            label_col: Name of the target label column.
            scaler_type: Type of scaler to use ('robust' or 'standard'). Defaults to 'robust'.
        """
        self.numeric_cols = numeric_cols
        self.label_col = label_col
        self.scaler_type = scaler_type
        
        if scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Use 'robust' or 'standard'.")
            
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        self.feature_names_ = None
        self.is_fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits the preprocessing pipeline on the input data and transforms it.
        
        Args:
            df: Training data DataFrame.
            
        Returns:
            Tuple of (X_scaled, y) as numpy arrays.
        """
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataframe.")
            
        # Ensure consistent column ordering based on provided numeric_cols
        missing_cols = [col for col in self.numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing numeric columns in dataframe: {missing_cols}")
            
        self.feature_names_ = list(self.numeric_cols)
        X_df = df[self.feature_names_]
        y = df[self.label_col].values
        
        logger.info(f"Fitting preprocessor on {len(X_df)} samples with {len(self.feature_names_)} features.")
        X_scaled = self.pipeline.fit_transform(X_df)
        self.is_fitted = True
        
        return X_scaled, y
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the already-fitted preprocessing pipeline to the input data.
        
        Args:
            df: Data DataFrame to transform.
            
        Returns:
            Tuple of (X_scaled, y) as numpy arrays.
            
        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")
            
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in dataframe.")
            
        X_df = df[self.feature_names_]
        y = df[self.label_col].values
        
        logger.info(f"Transforming {len(X_df)} samples using fitted preprocessor.")
        X_scaled = self.pipeline.transform(X_df)
        
        return X_scaled, y
        
    def save(self, path: str) -> None:
        """
        Saves the fitted preprocessor to a pickle file.
        
        Args:
            path: File path to save the preprocessor.
        """
        if not self.is_fitted:
            logger.warning("Saving an unfitted preprocessor.")
            
        path_obj = pathlib.Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'wb') as f:
            pickle.dump(self, f)
            
        logger.info(f"Preprocessor saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'ClientPreprocessor':
        """
        Loads a saved preprocessor from a pickle file.
        
        Args:
            path: File path to load the preprocessor from.
            
        Returns:
            ClientPreprocessor instance.
        """
        path_obj = pathlib.Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Preprocessor file not found at {path}")
            
        with open(path_obj, 'rb') as f:
            obj = pickle.load(f)
            
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
            
        logger.info(f"Preprocessor loaded from {path}")
        return obj
        
    def get_feature_dim(self) -> int:
        """
        Returns the number of output features.
        
        Returns:
            Integer representing the number of features.
            
        Raises:
            RuntimeError: If called before fit_transform.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")
        return len(self.feature_names_)
        
    def __repr__(self) -> str:
        status = "Fitted" if self.is_fitted else "Unfitted"
        feature_count = len(self.feature_names_) if self.is_fitted else len(self.numeric_cols)
        return f"ClientPreprocessor(scaler_type='{self.scaler_type}', status='{status}', features={feature_count})"
