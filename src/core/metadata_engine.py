"""Metadata Engine — The Universal Translator for the Federated Learning Platform.

This module provides a metadata-driven abstraction layer that allows the platform
to process any tabular dataset without hard-coded column names. Each organization
provides a mapping.json that translates local column names to a global feature index.

Usage:
    mapper = MetadataMapper("configs/neobank_a_mapping.json")
    mapper.validate_local_data(df)
    ordered_columns = mapper.get_feature_order()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import pandas as pd

from src.core.contract import FeatureType, ImputationStrategy

logger = logging.getLogger(__name__)


@dataclass
class FeatureMapping:
    """Represents a single feature's mapping from local to global space."""
    global_index: int
    local_name: str
    feature_type: FeatureType
    imputation: ImputationStrategy
    description: str = ""
    default_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_index": self.global_index,
            "local_name": self.local_name,
            "type": self.feature_type.value,
            "impute": self.imputation.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureMapping":
        return cls(
            global_index=data["global_index"],
            local_name=data["local_name"],
            feature_type=FeatureType(data.get("type", "numeric")),
            imputation=ImputationStrategy(data.get("impute", "median")),
            description=data.get("description", ""),
            default_value=data.get("default_value", 0.0),
        )


class MetadataMapper:
    """Loads and manages the mapping.json contract for a specific client.
    
    The mapping.json defines how local column names map to global feature indices.
    This enables the platform to be domain-agnostic — the same code processes
    fraud data, credit risk data, or cybersecurity data by simply changing the map.
    
    Args:
        mapping_path: Path to the mapping.json file
        
    Example mapping.json structure:
    {
        "client_id": "neobank_a",
        "domain": "fintech_credit_risk",
        "vector_size": 128,
        "target_column": "is_default",
        "feature_mapping": [
            {"global_index": 0, "local_name": "fico_score", "type": "numeric", "impute": "median"},
            {"global_index": 1, "local_name": "annual_inc", "type": "numeric", "impute": "mean"},
        ]
    }
    """
    
    def __init__(self, mapping_path: str):
        self.mapping_path = Path(mapping_path)
        self.raw_config: Dict[str, Any] = {}
        self.client_id: str = ""
        self.domain: str = ""
        self.vector_size: int = 128
        self.target_column: str = "target"
        self.feature_mappings: List[FeatureMapping] = []
        self._numeric_features: List[FeatureMapping] = []
        self._categorical_features: List[FeatureMapping] = []
        
        self._load_mapping()
        self._index_mappings()
        
        logger.info(f"MetadataMapper loaded: {self.client_id} ({len(self.feature_mappings)} features)")
    
    def _load_mapping(self):
        """Load and parse the mapping.json file."""
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_path}")
        
        with open(self.mapping_path, 'r') as f:
            self.raw_config = json.load(f)
        
        self.client_id = self.raw_config.get("client_id", "unknown")
        self.domain = self.raw_config.get("domain", "unknown")
        self.vector_size = self.raw_config.get("vector_size", 128)
        self.target_column = self.raw_config.get("target_column", "target")
        
        raw_mappings = self.raw_config.get("feature_mapping", [])
        self.feature_mappings = [
            FeatureMapping.from_dict(m) for m in raw_mappings
        ]
    
    def _index_mappings(self):
        """Separate features by type for preprocessing."""
        self._numeric_features = sorted(
            [f for f in self.feature_mappings if f.feature_type == FeatureType.NUMERIC],
            key=lambda x: x.global_index
        )
        self._categorical_features = sorted(
            [f for f in self.feature_mappings if f.feature_type == FeatureType.CATEGORICAL],
            key=lambda x: x.global_index
        )
        
        logger.debug(
            f"Features: {len(self._numeric_features)} numeric, "
            f"{len(self._categorical_features)} categorical"
        )
    
    def get_feature_order(self) -> List[str]:
        """Return local column names sorted by their global index.
        
        Returns:
            List of column names in the order they should appear in the feature vector.
        """
        return [f.local_name for f in sorted(self.feature_mappings, key=lambda x: x.global_index)]
    
    def get_numeric_columns(self) -> List[str]:
        """Return list of numeric column names."""
        return [f.local_name for f in self._numeric_features]
    
    def get_categorical_columns(self) -> List[str]:
        """Return list of categorical column names."""
        return [f.local_name for f in self._categorical_features]
    
    def get_target_column(self) -> str:
        """Return the target/label column name."""
        return self.target_column
    
    def validate_local_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that the DataFrame contains all mapped columns.
        
        Args:
            df: Local DataFrame to validate
            
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        if self.target_column not in df.columns:
            issues.append(f"Missing target column: '{self.target_column}'")
        
        for feature in self.feature_mappings:
            if feature.local_name not in df.columns:
                issues.append(
                    f"Missing mapped feature '{feature.local_name}' "
                    f"(global_index={feature.global_index})"
                )
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"Validation passed for {self.client_id}: all {len(self.feature_mappings)} features found")
        else:
            logger.warning(f"Validation failed for {self.client_id}: {len(issues)} issues")
        
        return is_valid, issues
    
    def get_index_to_local_name(self) -> Dict[int, str]:
        """Return mapping from global index to local column name."""
        return {f.global_index: f.local_name for f in self.feature_mappings}
    
    def get_local_to_index(self) -> Dict[str, int]:
        """Return mapping from local column name to global index."""
        return {f.local_name: f.global_index for f in self.feature_mappings}
    
    def get_imputation_strategy(self, column_name: str) -> ImputationStrategy:
        """Get imputation strategy for a specific column."""
        for f in self.feature_mappings:
            if f.local_name == column_name:
                return f.imputation
        return ImputationStrategy.MEDIAN
    
    def get_feature_type(self, column_name: str) -> FeatureType:
        """Get feature type for a specific column."""
        for f in self.feature_mappings:
            if f.local_name == column_name:
                return f.feature_type
        return FeatureType.NUMERIC
    
    def get_missing_indices(self) -> List[int]:
        """Return global indices that have no mapping (gaps in the feature vector)."""
        mapped_indices = {f.global_index for f in self.feature_mappings}
        return [i for i in range(self.vector_size) if i not in mapped_indices]
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the mapping configuration."""
        return {
            "client_id": self.client_id,
            "domain": self.domain,
            "vector_size": self.vector_size,
            "target_column": self.target_column,
            "total_features": len(self.feature_mappings),
            "numeric_features": len(self._numeric_features),
            "categorical_features": len(self._categorical_features),
            "missing_indices": len(self.get_missing_indices()),
        }


def create_default_mapping(
    client_id: str,
    columns: List[str],
    target_column: str = "Class",
    domain: str = "fraud_detection",
    vector_size: int = 128,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create a default mapping.json for a dataset with standard column names.
    
    Useful for bootstrapping new clients or when mapping.json is not provided.
    Maps columns to global indices 0, 1, 2, ... sequentially.
    
    Args:
        client_id: Client identifier
        columns: List of column names (excluding target)
        target_column: Name of target/label column
        domain: Domain identifier
        vector_size: Target vector size
        output_path: If provided, save the mapping to this path
        
    Returns:
        The mapping configuration dictionary
    """
    feature_mapping = []
    for i, col in enumerate(columns):
        feature_mapping.append({
            "global_index": i,
            "local_name": col,
            "type": "numeric",
            "impute": "median",
            "description": f"Auto-mapped feature {i}",
        })
    
    mapping = {
        "client_id": client_id,
        "domain": domain,
        "vector_size": vector_size,
        "target_column": target_column,
        "feature_mapping": feature_mapping,
    }
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"Default mapping saved to {output_path}")
    
    return mapping