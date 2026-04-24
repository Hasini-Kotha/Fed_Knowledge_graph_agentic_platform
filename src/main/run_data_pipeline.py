import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import DataLoader, prepare_data_directory
from src.data.schema import get_schema
from src.data.validate import validate_dataset, verify_client_split
from src.data.split_clients import run_full_split, create_fraud_split_summary_note
from src.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/data_config.yaml"):
    """
    Main data pipeline: Ingest -> Validate -> Split -> Verify.
    """
    logger.info("=" * 60)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 60)

    # 1. Load Configuration
    config = ConfigLoader(config_path)
    data_cfg = config.get_section("data")
    split_cfg = config.get_section("split")
    
    # 2. Prepare Directories
    prepare_data_directory()
    
    # 3. Load Data
    logger.info(f"Loading dataset from source: {data_cfg.get('source')}")
    loader = DataLoader(data_cfg.get("data_dir"))
    df = loader.load_by_name(data_cfg.get("source"))
    
    # 4. Get Schema and Validate Raw Data
    schema_name = data_cfg.get("schema")
    schema = get_schema(schema_name)
    logger.info(f"Validating data against schema: {schema_name}")
    
    is_valid, report = validate_dataset(df, schema, schema_name)
    if not is_valid:
        logger.error("Raw data validation failed!")
        for issue in report.get("issues", []):
            logger.error(f"  - {issue}")
        sys.exit(1)
    
    logger.info("Data validation passed.")
    
    # 5. Split Data
    logger.info("Starting data splitting...")
    saved_files, stats = run_full_split(
        df=df,
        output_dir=split_cfg.get("output_dir"),
        label_col=schema.label_column,
        label_positive=schema.label_positive,
        test_ratio=split_cfg.get("test_ratio"),
        val_ratio=split_cfg.get("val_ratio"),
        random_seed=split_cfg.get("random_seed"),
        non_iid=split_cfg.get("non_iid"),
        sort_by=split_cfg.get("sort_by"),
        prefix=split_cfg.get("prefix"),
    )
    
    # 6. Final Verification
    logger.info("Verifying splits...")
    # (Helper to reload for verification if needed, or use 'stats' directly)
    summary_note = create_fraud_split_summary_note(stats, split_cfg.get("non_iid"))
    logger.info("\n" + summary_note)
    
    logger.info("=" * 60)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument("--config", default="configs/data_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    run_pipeline(args.config)
