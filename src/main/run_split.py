import sys
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import DataLoader
from src.data.schema import create_fraud_schema
from src.data.split_clients import (
    ClientSplitter,
    compute_split_statistics,
    save_client_splits,
    save_split_summary,
    create_fraud_split_summary_note,
    run_full_split,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_split(
    source: str = "kaggle",
    data_dir: str = "Sample_datasets",
    output_dir: str = "data/splits",
    test_ratio: float = 0.15,
    val_ratio: float = 0.20,
    random_seed: int = 42,
    non_iid: bool = False,
    sort_by: str = None,
    prefix: str = "",
):
    """Run full client split pipeline.

    Args:
        source: Dataset source ('kaggle', 'simulated')
        data_dir: Directory containing datasets
        output_dir: Directory for split outputs
        test_ratio: Global test set ratio
        val_ratio: Client validation split ratio
        random_seed: Random seed
        non_iid: Whether to create non-IID split
        sort_by: Column to sort by for non-IID
        prefix: Filename prefix
    """
    logger.info("=" * 60)
    logger.info("CLIENT SPLIT PIPELINE")
    logger.info("=" * 60)

    logger.info(f"Loading dataset: {source}")
    loader = DataLoader(data_dir)
    df = loader.load_by_name(source)

    schema = create_fraud_schema()
    logger.info(f"Schema: {schema.name}")
    logger.info(f"Label column: {schema.label_column}")

    logger.info("-" * 40)
    logger.info("Split parameters:")
    logger.info(f"  Test ratio: {test_ratio}")
    logger.info(f"  Val ratio: {val_ratio}")
    logger.info(f"  Random seed: {random_seed}")
    logger.info(f"  Non-IID: {non_iid}")
    if sort_by:
        logger.info(f"  Sort by: {sort_by}")

    saved_files, stats = run_full_split(
        df=df,
        output_dir=output_dir,
        label_col=schema.label_column,
        label_positive=schema.label_positive,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
        non_iid=non_iid,
        sort_by=sort_by,
        prefix=prefix,
    )

    logger.info("-" * 40)
    logger.info("OUTPUT FILES:")
    logger.info("-" * 40)
    for name, path in saved_files.items():
        logger.info(f"  {name}: {path}")

    logger.info("-" * 40)
    logger.info("SPLIT STATISTICS:")
    logger.info("-" * 40)

    for client_id, client_stats in stats["clients"].items():
        logger.info(f"{client_id}:")
        logger.info(
            f"  Train: {client_stats['train_rows']} rows, "
            f"{client_stats['train_positive']} fraud ({client_stats['train_positive_ratio']:.4f})"
        )
        logger.info(
            f"  Val: {client_stats['val_rows']} rows, "
            f"{client_stats['val_positive']} fraud ({client_stats['val_positive_ratio']:.4f})"
        )

    logger.info(f"Global test:")
    logger.info(
        f"  {stats['global_test']['rows']} rows, "
        f"{stats['global_test']['positive']} fraud ({stats['global_test']['positive_ratio']:.4f})"
    )

    logger.info("-" * 40)
    summary_note = create_fraud_split_summary_note(stats, non_iid)
    logger.info("\n" + summary_note)

    logger.info("=" * 60)
    logger.info("SPLIT PIPELINE COMPLETE")
    logger.info("=" * 60)

    return saved_files, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run client split pipeline")
    parser.add_argument("--source", default="kaggle", help="Dataset source")
    parser.add_argument("--data-dir", default="Sample_datasets", help="Data directory")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument(
        "--val-ratio", type=float, default=0.20, help="Validation ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--non-iid", action="store_true", help="Create non-IID split")
    parser.add_argument("--sort-by", default=None, help="Column to sort by for non-IID")
    parser.add_argument("--prefix", default="", help="Filename prefix")

    args = parser.parse_args()

    run_split(
        args.source,
        args.data_dir,
        args.output_dir,
        args.test_ratio,
        args.val_ratio,
        args.seed,
        args.non_iid,
        args.sort_by,
        args.prefix,
    )
