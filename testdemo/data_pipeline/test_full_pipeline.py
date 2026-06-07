import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.main.run_data_pipeline import run_data_pipeline

print("=" * 60)
print("RUNNING FULL DATA PIPELINE")
print("=" * 60)

result = run_data_pipeline(
    source="kaggle",
    data_dir="Sample_datasets",
    mapping_path="configs/mapping.json",
    output_dir="data/splits",
    artifacts_dir="artifacts",
    test_ratio=0.15,
    val_ratio=0.20,
    random_seed=42
)

if result:
    print("\nPipeline completed successfully!")
    print(f"Vectorizer: {result['vectorizer_path']}")
    print(f"Files saved: {len(result['saved_files'])}")
else:
    print("\nPipeline failed!")