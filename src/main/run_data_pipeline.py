import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.load_data import DataLoader, prepare_data_directory
from src.data.schema import get_schema
from src.data.validate import validate_dataset, verify_client_split
from src.data.split_clients import ClientSplitter, run_full_split, create_split_summary_note
from src.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str = "configs/data_config.yaml"):
    """Main data pipeline: Ingest → Validate → Split → Save Global Preprocessor.

    The global preprocessor is fitted on the non-test portion of the dataset
    (server-side, before FL begins) and saved to
    artifacts/preprocessors/global_preprocessor.pkl. This ensures that the
    global holdout evaluation uses a scaler that represents the full training
    distribution — not a single client's local scaler.
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
    logger.info("Loading dataset from source: %s", data_cfg.get("source"))
    loader = DataLoader(data_cfg.get("data_dir"))
    df = loader.load_by_name(data_cfg.get("source"))

    # 4. Get Schema and Validate Raw Data
    schema_name = data_cfg.get("schema")
    schema = get_schema(schema_name)
    logger.info("Validating data against schema: %s", schema_name)

    is_valid, report = validate_dataset(df, schema, schema_name)
    if not is_valid:
        logger.error("Raw data validation failed!")
        for issue in report.get("issues", []):
            logger.error("  - %s", issue)
        sys.exit(1)

    logger.info("Data validation passed.")

    # 5. Split Data
    logger.info("Starting data splitting...")
    random_seed = split_cfg.get("random_seed", 42)
    test_ratio = split_cfg.get("test_ratio", 0.15)

    saved_files, stats = run_full_split(
        df=df,
        output_dir=split_cfg.get("output_dir"),
        label_col=schema.label_column,
        label_positive=schema.label_positive,
        test_ratio=test_ratio,
        val_ratio=split_cfg.get("val_ratio", 0.20),
        random_seed=random_seed,
        non_iid=split_cfg.get("non_iid", False),
        sort_by=split_cfg.get("sort_by"),
        prefix=split_cfg.get("prefix", ""),
    )

    # 6. Fit & Save Global Preprocessor on the non-test portion
    #    This scaler is used ONLY for global holdout evaluation — never for FL training.
    feature_cols = schema.get_feature_columns(df)
    if feature_cols:
        try:
            from src.data.preprocess import ClientPreprocessor

            # Reproduce the same 85/15 split deterministically to get non-test rows
            import numpy as np
            rng = np.random.default_rng(random_seed)
            all_indices = np.arange(len(df))
            rng.shuffle(all_indices)
            n_test = int(len(df) * test_ratio)
            non_test_indices = all_indices[n_test:]
            df_non_test = df.iloc[non_test_indices]

            global_preprocessor = ClientPreprocessor(
                numeric_cols=feature_cols,
                label_col=schema.label_column,
            )
            global_preprocessor.fit_transform(df_non_test)

            prep_dir = Path("artifacts/preprocessors")
            prep_dir.mkdir(parents=True, exist_ok=True)
            global_preprocessor.save(str(prep_dir / "global_preprocessor.pkl"))
            logger.info(
                "Global preprocessor saved → artifacts/preprocessors/global_preprocessor.pkl"
            )
        except Exception as exc:
            logger.warning("Could not save global preprocessor: %s", exc)
    else:
        logger.warning(
            "Schema '%s' has no feature_columns defined — skipping global preprocessor.",
            schema_name,
        )

    # 7. Final Verification Summary
    summary_note = create_split_summary_note(
        stats,
        is_non_iid=split_cfg.get("non_iid", False),
        label_positive_name=schema.label_column,
    )
    logger.info("\n%s", summary_note)

    logger.info("=" * 60)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("=" * 60)

    return saved_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preparation pipeline")
    parser.add_argument(
        "--config",
        default="configs/data_config.yaml",
        help="Path to data config file",
    )
    args = parser.parse_args()
    run_pipeline(args.config)
