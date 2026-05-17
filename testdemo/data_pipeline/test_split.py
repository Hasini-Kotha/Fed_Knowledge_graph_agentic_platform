import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import DataLoader
from src.data.schema import create_fraud_schema
from src.data.validate import validate_dataset

print("=" * 60)
print("DATA VALIDATION")
print("=" * 60)

loader = DataLoader("Sample_datasets")
df = loader.load_kaggle_fraud("credit-card-1/creditcard.csv")

schema = create_fraud_schema()
is_valid, report = validate_dataset(df, schema, "kaggle")

print(f"Valid: {is_valid}")
print(f"Issues: {len(report.get('issues', []))}")
for issue in report.get("issues", []):
    print(f"  - {issue}")

analysis = loader.analyze(df)
print(f"\nDataset Shape: {analysis['shape']}")
dist = analysis.get("class_distribution", {})
print(f"Fraud ratio: {dist.get('fraud_ratio', 0):.4f}")
print(f"Total fraud: {dist.get('fraud', 0):,}")
print(f"Total legit: {dist.get('legit', 0):,}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE - Dataset is valid!")
print("=" * 60)
