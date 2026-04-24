import pandas as pd
import numpy as np
import logging
import json
import random
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClientSplitter:
    """Split data into 3-client simulation with train/val/test splits.

    Splitting strategy:
    - 15% global holdout test set (untouched, for final evaluation)
    - 85% remaining split across 3 simulated clients
    - Each client gets 80/20 train/val split

    Assumptions:
    - Data represents multiple banks' transactions
    - Split is IID by default (random shuffle)
    - Can be made non-IID by sorting before split if needed
    """

    def __init__(
        self,
        num_clients: int = 3,
        test_ratio: float = 0.15,
        val_ratio: float = 0.20,
        random_seed: int = 42,
    ):
        self.num_clients = num_clients
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        np.random.seed(random_seed)
        random.seed(random_seed)

    def split(
        self,
        df: pd.DataFrame,
        label_col: str = "Class",
        non_iid: bool = False,
        sort_by: Optional[str] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Split dataset into clients with train/val splits.

        Args:
            df: Input DataFrame
            label_col: Name of label column
            non_iid: If True, sort by column before split for non-IID distribution
            sort_by: Column to sort by for non-IID (e.g., 'Amount', 'Time')

        Returns:
            Dict of {client_id: {'train': df, 'val': df}}
        """
        logger.info(f"Splitting dataset: {len(df)} rows, {self.num_clients} clients")
        logger.info(f"Test ratio: {self.test_ratio}, Val ratio: {self.val_ratio}")

        df_work = df.copy()

        if non_iid and sort_by:
            logger.info(f"Non-IID split: sorting by {sort_by}")
            df_work = df_work.sort_values(sort_by).reset_index(drop=True)

        indices = np.arange(len(df_work))
        np.random.shuffle(indices)
        df_work = df_work.iloc[indices].reset_index(drop=True)

        test_size = int(len(df_work) * self.test_ratio)

        df_test = df_work.iloc[:test_size]
        df_remaining = df_work.iloc[test_size:]

        logger.info(f"Global test set: {len(df_test)} rows")
        logger.info(f"Remaining for clients: {len(df_remaining)} rows")

        per_client = len(df_remaining) // self.num_clients

        client_splits = {}
        client_ids = ["client_a", "client_b", "client_c"][: self.num_clients]

        for i, client_id in enumerate(client_ids):
            start_idx = i * per_client
            end_idx = (
                start_idx + per_client
                if i < self.num_clients - 1
                else len(df_remaining)
            )

            client_data = df_remaining.iloc[start_idx:end_idx]

            train_idx, val_idx = train_test_split(
                np.arange(len(client_data)),
                test_size=self.val_ratio,
                random_state=self.random_seed,
            )

            df_train = client_data.iloc[train_idx]
            df_val = client_data.iloc[val_idx]

            client_splits[client_id] = {"train": df_train, "val": df_val}

            logger.info(f"{client_id}: train={len(df_train)}, val={len(df_val)}")

        client_splits["_global_test"] = df_test

        return client_splits

    def split_with_client_ids(
        self,
        df: pd.DataFrame,
        client_ids: List[str],
        label_col: str = "Class",
        stratify: bool = True,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Split with explicit client IDs and optional stratification.

        Args:
            df: Input DataFrame
            client_ids: List of client ID strings
            label_col: Label column name
            stratify: Whether to maintain class balance

        Returns:
            Dictionary of client splits
        """
        logger.info(f"Splitting with {len(client_ids)} explicit client IDs")

        self.num_clients = len(client_ids)

        return self.split(df, label_col, non_iid=not stratify)

    def create_client_datasets(
        self, df: pd.DataFrame, label_col: str = "Class", non_iid: bool = False
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
        """Create all client datasets in one call.

        Returns:
            (client_trains, client_vals, global_test)
        """
        splits = self.split(df, label_col, non_iid=non_iid)

        client_trains = {}
        client_vals = {}

        for client_id in splits:
            if client_id == "_global_test":
                continue
            client_trains[client_id] = splits[client_id]["train"]
            client_vals[client_id] = splits[client_id]["val"]

        global_test = splits["_global_test"]

        return client_trains, client_vals, global_test


def compute_split_statistics(
    client_splits: Dict[str, Dict[str, pd.DataFrame]],
    label_col: str,
    label_positive: int = 1,
) -> Dict[str, Any]:
    """Compute statistics for each client split.

    Args:
        client_splits: Output from ClientSplitter.split()
        label_col: Label column name
        label_positive: Positive class value (1 for fraud)

    Returns:
        Statistics dictionary
    """
    stats = {"clients": {}, "global_test": {}, "summary": {}}

    total_train = 0
    total_val = 0
    total_positive = 0

    for client_id in client_splits:
        if client_id == "_global_test":
            continue

        train_df = client_splits[client_id]["train"]
        val_df = client_splits[client_id]["val"]

        train_pos = (train_df[label_col] == label_positive).sum()
        val_pos = (val_df[label_col] == label_positive).sum()

        client_stats = {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_positive": int(train_pos),
            "val_positive": int(val_pos),
            "train_positive_ratio": float(train_pos / len(train_df))
            if len(train_df) > 0
            else 0,
            "val_positive_ratio": float(val_pos / len(val_df))
            if len(val_df) > 0
            else 0,
        }

        stats["clients"][client_id] = client_stats

        total_train += len(train_df)
        total_val += len(val_df)
        total_positive += train_pos + val_pos

    global_df = client_splits["_global_test"]
    global_pos = (global_df[label_col] == label_positive).sum()

    stats["global_test"] = {
        "rows": len(global_df),
        "positive": int(global_pos),
        "positive_ratio": float(global_pos / len(global_df))
        if len(global_df) > 0
        else 0,
    }

    stats["summary"] = {
        "total_train": total_train,
        "total_val": total_val,
        "total_test": len(global_df),
        "total_positive": int(total_positive + global_pos),
        "average_positive_ratio": float(total_positive / (total_train + total_val)),
    }

    return stats


def save_client_splits(
    client_splits: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str = "data/splits",
    prefix: str = "",
) -> Dict[str, str]:
    """Save all client splits to CSV files.

    Args:
        client_splits: Output from ClientSplitter.split()
        output_dir: Output directory path
        prefix: Optional filename prefix

    Returns:
        Dictionary of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    for client_id in client_splits:
        if client_id == "_global_test":
            continue

        for split_type in ["train", "val"]:
            df = client_splits[client_id][split_type]

            filename = (
                f"{prefix}{client_id}_{split_type}.csv"
                if prefix
                else f"{client_id}_{split_type}.csv"
            )
            filepath = output_path / filename

            df.to_csv(filepath, index=False)
            saved_files[f"{client_id}_{split_type}"] = str(filepath)

            logger.info(f"Saved {filename}: {len(df)} rows")

    global_filename = f"{prefix}global_test.csv" if prefix else "global_test.csv"
    global_filepath = output_path / global_filename
    client_splits["_global_test"].to_csv(global_filepath, index=False)
    saved_files["global_test"] = str(global_filepath)

    logger.info(f"Saved global test: {len(client_splits['_global_test'])} rows")

    return saved_files


def save_split_summary(
    stats: Dict[str, Any],
    output_dir: str = "data/splits",
    filename: str = "split_summary.json",
) -> str:
    """Save split summary metadata.

    Args:
        stats: Statistics from compute_split_statistics()
        output_dir: Output directory
        filename: Summary filename

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved split summary: {filepath}")

    return str(filepath)


def run_full_split(
    df: pd.DataFrame,
    output_dir: str = "data/splits",
    label_col: str = "Class",
    label_positive: int = 1,
    test_ratio: float = 0.15,
    val_ratio: float = 0.20,
    random_seed: int = 42,
    non_iid: bool = False,
    sort_by: Optional[str] = None,
    prefix: str = "",
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Run full split pipeline and save all outputs.

    Args:
        df: Input DataFrame
        output_dir: Output directory
        label_col: Label column name
        label_positive: Positive class value
        test_ratio: Global test ratio
        val_ratio: Client validation ratio
        random_seed: Random seed
        non_iid: Whether to create non-IID split
        sort_by: Column to sort by for non-IID
        prefix: Filename prefix

    Returns:
        (saved_files_dict, statistics_dict)
    """
    logger.info("=" * 50)
    logger.info("Running full client split pipeline")
    logger.info("=" * 50)

    splitter = ClientSplitter(
        num_clients=3,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
    )

    client_splits = splitter.split(
        df, label_col=label_col, non_iid=non_iid, sort_by=sort_by
    )

    stats = compute_split_statistics(
        client_splits, label_col=label_col, label_positive=label_positive
    )

    saved_files = save_client_splits(
        client_splits, output_dir=output_dir, prefix=prefix
    )

    save_split_summary(stats, output_dir=output_dir)

    logger.info("=" * 50)
    logger.info("Split pipeline complete")
    logger.info("=" * 50)

    return saved_files, stats


def create_fraud_split_summary_note(
    stats: Dict[str, Any],
    is_non_iid: bool = False,
    additional_notes: Optional[Dict[str, str]] = None,
) -> str:
    """Create a summary note about the split.

    Args:
        stats: Split statistics
        is_non_iid: Whether the split is non-IID
        additional_notes: Additional notes to include

    Returns:
        Summary string
    """
    note_lines = [
        "=" * 60,
        "CLIENT SPLIT SUMMARY",
        "=" * 60,
        "",
        f"Split Type: {'NON-IID' if is_non_iid else 'IID (Independent Identically Distributed)'}",
        "",
        f"Total Samples: {stats['summary']['total_train'] + stats['summary']['total_val'] + stats['global_test']['rows']}",
        f"  - Training: {stats['summary']['total_train']}",
        f"  - Validation: {stats['summary']['total_val']}",
        f"  - Global Test: {stats['global_test']['rows']}",
        "",
        "Fraud Ratios:",
    ]

    for client_id, client_stats in stats["clients"].items():
        note_lines.append(
            f"  {client_id}: "
            f"train={client_stats['train_positive_ratio']:.4f}, "
            f"val={client_stats['val_positive_ratio']:.4f}"
        )

    note_lines.extend(
        [
            f"  global_test: {stats['global_test']['positive_ratio']:.4f}",
            "",
            "Average Fraud Ratio: {:.4f}".format(
                stats["summary"]["average_positive_ratio"]
            ),
            "",
        ]
    )

    if additional_notes:
        note_lines.extend(["Additional Notes:", ""])
        for key, value in additional_notes.items():
            note_lines.append(f"  {key}: {value}")
        note_lines.append("")

    note_lines.append("=" * 60)

    return "\n".join(note_lines)
