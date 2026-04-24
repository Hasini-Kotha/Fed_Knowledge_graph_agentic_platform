import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation utilities for tabular datasets.

    Validates:
    - Missing columns
    - Label errors
    - Duplicate columns
    - Empty outputs
    - Data type consistency
    """

    def __init__(self, schema=None):
        self.schema = schema
        self.errors = []
        self.warnings = []

    def validate(self, df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
        """Validate DataFrame against schema.

        Args:
            df: Input DataFrame
            strict: If True, fail on any missing required column

        Returns:
            (is_valid, list_of_issues)
        """
        self.errors = []
        self.warnings = []

        if df is None or df.empty:
            self.errors.append("DataFrame is None or empty")
            return False, self.errors

        self._check_required_columns(df)
        self._check_label_column(df)
        self._check_duplicates(df)
        self._check_data_types(df)
        self._check_class_distribution(df)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors + self.warnings

    def _check_required_columns(self, df: pd.DataFrame):
        """Check if all required columns are present."""
        if self.schema and self.schema.required_columns:
            for col in self.schema.required_columns:
                if col not in df.columns:
                    self.errors.append(f"Missing required column: {col}")
        else:
            logger.warning("No schema or required_columns defined")

    def _check_label_column(self, df: pd.DataFrame):
        """Validate label column exists and has valid values."""
        label_col = self.schema.label_column if self.schema else "Class"

        if label_col not in df.columns:
            self.errors.append(f"Label column '{label_col}' not found")
            return

        unique_labels = df[label_col].unique()

        if self.schema:
            expected = {self.schema.label_positive, self.schema.label_negative}
            actual = set(unique_labels)
            if not actual.issubset(expected):
                self.warnings.append(
                    f"Label column contains unexpected values: {actual - expected}. "
                    f"Expected {expected}"
                )

    def _check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate rows."""
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            self.warnings.append(f"Found {dup_count} duplicate rows")

    def _check_data_types(self, df: pd.DataFrame):
        """Check data types for numeric columns."""
        if self.schema and self.schema.numeric_columns:
            for col in self.schema.numeric_columns:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        self.warnings.append(f"Column '{col}' is not numeric type")

    def _check_class_distribution(self, df: pd.DataFrame):
        """Check class distribution for validity."""
        label_col = self.schema.label_column if self.schema else "Class"

        if label_col not in df.columns:
            return

        try:
            pos_count = (df[label_col] == 1).sum()
            neg_count = (df[label_col] == 0).sum()
            total = len(df)

            if total == 0:
                self.errors.append("DataFrame has zero rows")
                return

            pos_ratio = pos_count / total

            logger.info(
                f"Class distribution: {pos_count} positive ({pos_ratio:.4f}), {neg_count} negative"
            )

            if pos_ratio == 0:
                self.warnings.append("No positive class samples found")
            elif pos_ratio == 1:
                self.warnings.append("No negative class samples found")

        except Exception as e:
            self.warnings.append(f"Could not compute class distribution: {e}")


class DataValidationReport:
    """Comprehensive validation report for datasets."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.validations = []
        self.timestamp = pd.Timestamp.now().isoformat()

    def add_validation(self, name: str, passed: bool, issues: List[str]):
        self.validations.append({"name": name, "passed": passed, "issues": issues})

    def is_valid(self) -> bool:
        return all(v["passed"] for v in self.validations)

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "is_valid": self.is_valid(),
            "total_checks": len(self.validations),
            "passed_checks": sum(1 for v in self.validations if v["passed"]),
            "failed_checks": sum(1 for v in self.validations if not v["passed"]),
            "validations": self.validations,
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)


def validate_dataset(
    df: pd.DataFrame, schema, dataset_name: str = "dataset"
) -> Tuple[bool, Dict[str, Any]]:
    """Validate a dataset and return comprehensive report.

    Args:
        df: Input DataFrame
        schema: Schema contract
        dataset_name: Name for reporting

    Returns:
        (is_valid, report_dict)
    """
    validator = DataValidator(schema)
    is_valid, issues = validator.validate(df)

    report = {
        "dataset_name": dataset_name,
        "is_valid": is_valid,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "issues": issues,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if schema:
        label_col = schema.label_column
        if label_col in df.columns:
            pos = (df[label_col] == schema.label_positive).sum()
            neg = (df[label_col] == schema.label_negative).sum()
            report["class_distribution"] = {
                "positive": int(pos),
                "negative": int(neg),
                "positive_ratio": float(pos / len(df)) if len(df) > 0 else 0,
            }

    logger.info(
        f"Validation {'PASSED' if is_valid else 'FAILED'} for {dataset_name}: {len(issues)} issues"
    )

    return is_valid, report


def check_label_distribution(
    df: pd.DataFrame, label_col: str = "Class", pos_label: int = 1
) -> Dict[str, Any]:
    """Calculate label distribution for a dataset.

    Args:
        df: Input DataFrame
        label_col: Name of label column
        pos_label: Positive class value

    Returns:
        Dictionary with distribution stats
    """
    total = len(df)

    if total == 0:
        return {"total": 0, "positive": 0, "negative": 0, "positive_ratio": 0.0}

    positive = (df[label_col] == pos_label).sum()
    negative = total - positive
    positive_ratio = positive / total

    return {
        "total": int(total),
        "positive": int(positive),
        "negative": int(negative),
        "positive_ratio": float(positive_ratio),
    }


def verify_client_split(
    client_dfs: Dict[str, pd.DataFrame],
    global_test_df: pd.DataFrame,
    schema,
    original_total: int,
) -> Dict[str, Any]:
    """Verify that client splits preserve valid fraud labels and cover full dataset.

    Args:
        client_dfs: Dictionary of client_id -> DataFrame
        global_test_df: Global test DataFrame
        schema: Schema contract
        original_total: Original dataset size

    Returns:
        Verification report
    """
    results = {
        "original_total": original_total,
        "clients": {},
        "global_test": {},
        "total_accounted": 0,
        "is_valid": True,
        "issues": [],
    }

    total_in_clients = sum(len(df) for df in client_dfs.values())
    total_in_test = len(global_test_df)
    results["total_accounted"] = total_in_clients + total_in_test

    if results["total_accounted"] != original_total:
        results["issues"].append(
            f"Row count mismatch: original={original_total}, "
            f"clients+test={results['total_accounted']}"
        )
        results["is_valid"] = False

    for client_id, df in client_dfs.items():
        dist = check_label_distribution(df, schema.label_column, schema.label_positive)
        results["clients"][client_id] = dist

        if dist["positive"] == 0:
            results["issues"].append(f"Client {client_id} has no positive samples")
            results["is_valid"] = False

    global_dist = check_label_distribution(
        global_test_df, schema.label_column, schema.label_positive
    )
    results["global_test"] = global_dist

    return results
