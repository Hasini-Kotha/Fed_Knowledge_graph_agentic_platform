import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """Load raw tabular datasets with validation and sanity analysis.

    Supported loaders:
    - load_kaggle_fraud()         (legacy, Kaggle MLG-ULB creditcard.csv)
    - load_simulated_fraud()      (legacy, fraudTrain/fraudTest CSVs)
    - load_tabular_dataset(path)  (domain-agnostic, any CSV)
    """

    def __init__(self, data_dir: str = "Sample_datasets"):
        self.data_dir = Path(data_dir)
        self.loaded_data = None
        self.dataset_info = {}

    def load_kaggle_fraud(
        self, filename: str = "credit-card-1/creditcard.csv"
    ) -> pd.DataFrame:
        """Load Kaggle MLG-ULB credit card fraud dataset.

        Args:
            filename: Path to creditcard.csv

        Returns:
            DataFrame with raw data
        """
        path = self.data_dir / filename

        logger.info(f"Loading Kaggle fraud dataset from {path}")

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

        self.loaded_data = df
        self.dataset_info = {
            "source": "kaggle_mlg_ulb",
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }

        return df

    def load_simulated_fraud(
        self,
        train_file: str = "credit-card-2/fraudTrain.csv",
        test_file: str = "credit-card-2/fraudTest.csv",
        merge: bool = False,
    ) -> pd.DataFrame:
        """Load simulated bank fraud dataset.

        Args:
            train_file: Path to fraudTrain.csv
            test_file: Path to fraudTest.csv
            merge: If True, concatenate train and test

        Returns:
            DataFrame with raw data
        """
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file

        logger.info(f"Loading simulated fraud dataset")

        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_path}")

        df_train = pd.read_csv(train_path, index_col=0)

        if merge and test_path.exists():
            df_test = pd.read_csv(test_path, index_col=0)
            df = pd.concat([df_train, df_test], ignore_index=True)
            logger.info(f"Merged train+test: {len(df)} rows")
        else:
            df = df_train
            logger.info(f"Loaded train: {len(df)} rows")

        self.loaded_data = df
        self.dataset_info = {
            "source": "simulated_bank",
            "train_file": train_file,
            "test_file": test_file if test_path.exists() else None,
            "merge": merge,
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }

        return df

    def load_by_name(self, name: str) -> pd.DataFrame:
        """Load dataset by name shorthand.

        Args:
            name: 'kaggle', 'simulated', 'simulated_merged', 'p2p', or 'creditcard_2023'

        Returns:
            Loaded DataFrame
        """
        if name.lower() in ["kaggle", "kaggle_fraud", "creditcard"]:
            return self.load_kaggle_fraud()
        elif name.lower() in ["simulated", "simulated_fraud"]:
            return self.load_simulated_fraud()
        elif name.lower() == "simulated_merged":
            return self.load_simulated_fraud(merge=True)
        elif name.lower() in ["p2p", "p2p_fraud"]:
            return self.load_p2p_fraud()
        elif name.lower() in ["creditcard_2023", "credit-card-3"]:
            return self.load_creditcard_2023()
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def load_p2p_fraud(
        self,
        train_file: str = "credit-card-2/fraudTrain.csv",
        test_file: str = "credit-card-2/fraudTest.csv",
    ) -> pd.DataFrame:
        """Load P2P transaction fraud dataset (merged train+test)."""
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file

        logger.info(f"Loading P2P fraud dataset")

        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_path}")

        df_train = pd.read_csv(train_path)

        if test_path.exists():
            df_test = pd.read_csv(test_path)
            df = pd.concat([df_train, df_test], ignore_index=True)
            logger.info(f"Merged train+test: {len(df)} rows")
        else:
            df = df_train
            logger.info(f"Loaded train: {len(df)} rows")

        self.loaded_data = df
        self.dataset_info = {
            "source": "p2p_fraud",
            "train_file": train_file,
            "test_file": test_file if test_path.exists() else None,
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }

        return df

    def load_creditcard_2023(
        self, filename: str = "credit-card-3/creditcard_2023.csv"
    ) -> pd.DataFrame:
        """Load 2023 credit card fraud dataset."""
        path = self.data_dir / filename

        logger.info(f"Loading creditcard_2023 from {path}")

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)

        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")

        self.loaded_data = df
        self.dataset_info = {
            "source": "creditcard_2023",
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }

        return df

    def get_info(self) -> Dict[str, Any]:
        """Get loaded dataset information."""
        if not self.dataset_info:
            return {"loaded": False}
        return self.dataset_info

    def analyze(
        self,
        df: Optional[pd.DataFrame] = None,
        label_col: Optional[str] = None,
        label_positive: int = 1,
    ) -> Dict[str, Any]:
        """Perform sanity analysis on a dataset.

        Args:
            df: DataFrame to analyse (uses last loaded if not provided).
            label_col: Name of the label column. Auto-detected if None.
            label_positive: Value of the positive class (default 1).

        Returns:
            Analysis report dict.
        """
        if df is None:
            df = self.loaded_data
        if df is None:
            return {"error": "No data loaded"}

        analysis: Dict[str, Any] = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        # Auto-detect label column if not provided
        if label_col is None:
            for candidate in ["Class", "is_fraud", "label", "target"]:
                if candidate in df.columns:
                    label_col = candidate
                    break

        if label_col and label_col in df.columns:
            pos = (df[label_col] == label_positive).sum()
            neg = len(df) - pos
            analysis["class_distribution"] = {
                "label_column": label_col,
                "total": len(df),
                "positive": int(pos),
                "negative": int(neg),
                "positive_ratio": float(pos / len(df)),
            }
            logger.info(
                "Analysis: %d rows, positive_ratio=%.4f (col='%s')",
                len(df), float(pos / len(df)), label_col,
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()

        return analysis

    def load_tabular_dataset(
        self, path: str, index_col: Optional[int] = None
    ) -> pd.DataFrame:
        """Domain-agnostic CSV loader. Works with any tabular dataset.

        Args:
            path: Absolute or relative path to a CSV file.
            index_col: Passed to pd.read_csv (None by default).

        Returns:
            Loaded DataFrame.
        """
        full_path = self.data_dir / path if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
        logger.info("Loading tabular dataset from %s", full_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {full_path}")
        df = pd.read_csv(full_path, index_col=index_col)
        self.loaded_data = df
        self.dataset_info = {
            "source": str(full_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_list": list(df.columns),
        }
        logger.info("Loaded: %d rows, %d columns", len(df), len(df.columns))
        return df


def load_raw_dataset(
    source: str = "kaggle", data_dir: str = "Sample_datasets"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function to load raw dataset.

    Args:
        source: 'kaggle', 'simulated', or 'simulated_merged'
        data_dir: Directory containing datasets

    Returns:
        (DataFrame, analysis_report)
    """
    loader = DataLoader(data_dir)
    df = loader.load_by_name(source)
    analysis = loader.analyze(df)
    return df, analysis


def load_and_analyze(
    path: str, config_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and analyze a dataset from path.

    Args:
        path: Path to CSV file
        config_path: Optional path to YAML config

    Returns:
        (DataFrame, analysis_report)
    """
    logger.info(f"Loading dataset from {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    analysis = {
        "source_path": str(path),
        "shape": df.shape,
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().sum(),
    }

    label_col = "Class" if "Class" in df.columns else "is_fraud"
    if label_col in df.columns:
        pos = (df[label_col] == 1).sum()
        analysis["class_distribution"] = {
            "total": len(df),
            "positive": int(pos),
            "negative": int(len(df) - pos),
            "positive_ratio": float(pos / len(df)),
        }

    return df, analysis


def prepare_data_directory(base_dir: str = "data") -> Path:
    """Ensure data directories exist.

    Args:
        base_dir: Base data directory

    Returns:
        Path object
    """
    base_path = Path(base_dir)

    for subdir in ["raw", "processed", "splits"]:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Data directories prepared: {base_path}")

    return base_path
 
