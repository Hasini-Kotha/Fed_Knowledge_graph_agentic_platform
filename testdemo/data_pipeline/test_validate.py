import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.load_data import DataLoader
from src.data.schema import create_fraud_schema
from src.data.validate import DataValidator

print("=" * 60)
print("DATA VALIDATION - STEP 1")
print("=" * 60)

loader = DataLoader("Sample_datasets")
df = loader.load_kaggle_fraud("credit-card-1/creditcard.csv")

schema = create_fraud_schema()
print(f"Schema: {schema.name}")

validator = DataValidator(schema)
is_valid, issues = validator.validate(df)

print(f"\nDataset: Kaggle MLG-ULB Credit Card Fraud")
print(f"Shape: {df.shape}")
print(f"Valid: {is_valid}")

print("\nValidation Issues:")
if issues:
    for issue in issues:
        print(f"  - {issue}")
else:
    print("  None")

dist = (df[schema.label_column] == 1).sum()
total = len(df)
print(f"\nClass Distribution:")
print(f"  Fraud: {dist:,} ({dist / total:.4f})")
print(f"  Legit: {total - dist:,} ({(total - dist) / total:.4f})")

print("\n" + "=" * 60)
print("STEP 1 COMPLETE - Dataset validated!")
print("=" * 60)
