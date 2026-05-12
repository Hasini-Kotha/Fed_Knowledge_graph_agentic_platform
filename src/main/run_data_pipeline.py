"""Step 1: Data Pipeline — Ingest, validate, split, and fit global vectorizer.

This is the entry point for preparing data for federated learning.
It loads the raw dataset, validates it against the schema, splits it into
3 client datasets + global test set, fits the DynamicVectorizer on all
non-test data, and saves everything.

Usage:
    python src/main/run_data_pipeline.py
"""

import sys
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.metadata_engine import MetadataMapper, create_default_mapping
from src.core.vectorizer import DynamicVectorizer
from src.data.load_data import DataLoader
from src.data.schema import create_fraud_schema, create_schema_from_mapping
from src.data.validate import DataValidator
from src.data.split_clients import ClientSplitter, compute_split_statistics, save_client_splits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline(
    source: str = "kaggle",
    data_dir: str = "Sample_datasets",
    mapping_path: str = "configs/mapping.json",
    output_dir: str = "data/splits",
    artifacts_dir: str = "artifacts",
    test_ratio: float = 0.15,
    val_ratio: float = 0.20,
    random_seed: int = 42
):
    """Execute the full data preparation pipeline.
    
    Args:
        source: Dataset source ('kaggle', 'simulated', 'p2p')
        data_dir: Directory containing raw datasets
        mapping_path: Path to mapping.json
        output_dir: Directory for split outputs
        artifacts_dir: Directory for model artifacts
        test_ratio: Global holdout ratio
        val_ratio: Client validation ratio
        random_seed: Random seed
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PIPELINE")
    logger.info("=" * 60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("--- Phase 1: Load Dataset ---")
    loader = DataLoader(data_dir)
    df = loader.load_by_name(source)
    logger.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    logger.info("--- Phase 2: Load/Validate Mapping ---")
    try:
        mapper = MetadataMapper(mapping_path)
        summary = mapper.summary()
        logger.info(f"Mapping: {summary['client_id']} ({summary['total_features']} features)")
    except Exception as e:
        logger.warning(f"Mapping not found at {mapping_path}, creating default...")
        schema = create_fraud_schema()
        feature_cols = [c for c in df.columns if c != schema.label_column]
        create_default_mapping(
            client_id="default",
            columns=feature_cols,
            target_column=schema.label_column,
            domain="fraud_detection",
            vector_size=128,
            output_path=mapping_path
        )
        mapper = MetadataMapper(mapping_path)
        logger.info(f"Default mapping created: {len(mapper.feature_mappings)} features")
    
    is_valid_map, map_issues = mapper.validate_local_data(df)
    if is_valid_map:
        logger.info("Mapping validation: PASSED")
    else:
        logger.warning(f"Mapping validation issues: {map_issues}")
    
    logger.info("--- Phase 3: Validate Data (schema derived from mapping) ---")
    schema = create_schema_from_mapping(mapping_path)
    validator = DataValidator(schema)
    is_valid, issues = validator.validate(df)
    logger.info(f"Validation: {'PASSED' if is_valid else 'FAILED'} ({len(issues)} issues)")
    for issue in issues:
        logger.info(f"  - {issue}")
    
    if not is_valid:
        logger.error("Validation failed. Aborting pipeline.")
        return None
    
    logger.info("--- Phase 4: Split into Clients ---")
    splitter = ClientSplitter(
        num_clients=3,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )
    
    client_splits = splitter.split(
        df,
        label_col=schema.label_column,
        non_iid=False
    )
    
    stats = compute_split_statistics(
        client_splits,
        label_col=schema.label_column,
        label_positive=schema.label_positive
    )
    
    saved_files = save_client_splits(client_splits, output_dir=output_dir)
    logger.info(f"Saved {len(saved_files)} files to {output_dir}")
    
    for client_id, s in stats['clients'].items():
        logger.info(f"  {client_id}: train={s['train_rows']:,}, val={s['val_rows']:,}, "
                  f"fraud_ratio={s['train_positive_ratio']:.4f}")
    logger.info(f"  global_test: {stats['global_test']['rows']:,} rows")
    
    logger.info("--- Phase 5: Fit Global Vectorizer ---")
    vectorizer = DynamicVectorizer(vector_size=mapper.vector_size)
    
    all_train_data = []
    for client_id in ['client_a', 'client_b', 'client_c']:
        train_df = client_splits[client_id]['train']
        all_train_data.append(train_df)
    
    combined_train = None
    import pandas as pd
    combined_train = pd.concat(all_train_data, ignore_index=True)
    
    result = vectorizer.fit_transform(combined_train, mapper)
    X_tensor = result["data"]
    logger.info(f"Global vectorizer fitted: input={combined_train.shape}, output={X_tensor.shape}")
    
    source_tag = source.replace("-", "_").replace("/", "_")
    vectorizer_path = f"{artifacts_dir}/global_vectorizer_{source_tag}.pkl"
    vectorizer.save(vectorizer_path)
    logger.info(f"Global vectorizer saved: {vectorizer_path}")
    
    summary_report = {
        "pipeline": "data_pipeline",
        "source": source,
        "total_rows": len(df),
        "split_stats": stats,
        "vectorizer": {
            "vector_size": mapper.vector_size,
            "output_dim": vectorizer.get_feature_dim(),
            "mapping_summary": vectorizer.get_mapping_summary(),
        },
        "saved_files": saved_files,
        "vectorizer_path": vectorizer_path,
    }
    
    summary_path = f"{output_dir}/pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    logger.info(f"Pipeline summary saved: {summary_path}")
    logger.info("=" * 60)
    logger.info("STEP 1 COMPLETE")
    logger.info("=" * 60)
    
    return summary_report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--source", default="kaggle", help="Dataset source")
    parser.add_argument("--data-dir", default="Sample_datasets", help="Data directory")
    parser.add_argument("--mapping-path", default="configs/mapping.json", help="Mapping JSON path")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_data_pipeline(
        args.source,
        args.data_dir,
        args.mapping_path,
        args.output_dir,
        args.artifacts_dir,
        args.test_ratio,
        args.val_ratio,
        args.seed
    )
