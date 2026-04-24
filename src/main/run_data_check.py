import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_data import DataLoader
from src.data.schema import get_schema, create_fraud_schema
from src.data.validate import validate_dataset, check_label_distribution

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_data_check(
    source: str = "kaggle",
    data_dir: str = "Sample_datasets",
    output_dir: str = "data/processed",
):
    """Run data validation and analysis.

    Args:
        source: Dataset source ('kaggle', 'simulated')
        data_dir: Directory containing datasets
        output_dir: Directory for validation report
    """
    logger.info("=" * 60)
    logger.info("DATA VALIDATION AND ANALYSIS")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(data_dir)

    logger.info(f"Loading dataset: {source}")
    df = loader.load_by_name(source)

    schema = create_fraud_schema()
    logger.info(f"Using schema: {schema.name}")

    is_valid, issues = validate_dataset(df, schema, source)

    logger.info("-" * 40)
    logger.info("VALIDATION RESULTS:")
    logger.info("-" * 40)

    if is_valid:
        logger.info("STATUS: PASSED")
    else:
        logger.info("STATUS: FAILED")

    for issue in issues:
        logger.info(f"  - {issue}")

    analysis = loader.analyze(df)

    logger.info("-" * 40)
    logger.info("DATASET ANALYSIS:")
    logger.info("-" * 40)
    logger.info(f"  Shape: {analysis['shape']}")
    logger.info(f"  Memory: {analysis['memory_mb']:.2f} MB")

    if "class_distribution" in analysis:
        dist = analysis["class_distribution"]
        logger.info(f"  Total: {dist['total']}")
        logger.info(f"  Fraud: {dist['fraud']} ({dist['fraud_ratio']:.4f})")
        logger.info(f"  Legit: {dist['legit']}")

    logger.info("-" * 40)
    logger.info("COLUMN INFO:")
    logger.info("-" * 40)
    for col in analysis["columns"]:
        if col in [
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
        ]:
            logger.info(f"  {col}: numeric (PCA feature)")
        elif col in ["Time", "Amount"]:
            logger.info(f"  {col}: raw numeric")
        elif col == "Class":
            logger.info(f"  {col}: label (0=legit, 1=fraud)")

    report = {
        "source": source,
        "data_dir": data_dir,
        "is_valid": is_valid,
        "issues": issues,
        "analysis": analysis,
        "schema": schema.to_dict(),
    }

    report_path = output_path / "validation_report.json"
    import json

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("-" * 40)
    logger.info(f"Report saved: {report_path}")
    logger.info("=" * 60)

    return is_valid, report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data validation")
    parser.add_argument("--source", default="kaggle", help="Dataset source")
    parser.add_argument("--data-dir", default="Sample_datasets", help="Data directory")
    parser.add_argument(
        "--output-dir", default="data/processed", help="Output directory"
    )

    args = parser.parse_args()

    run_data_check(args.source, args.data_dir, args.output_dir)
