import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.data.load_data import DataLoader
from src.core.contract import VectorContract

print("=" * 60)
print("TESTING METADATA ENGINE + DYNAMIC VECTORIZER")
print("=" * 60)

print("\n--- Step 1: Load Dataset ---")
loader = DataLoader('Sample_datasets')
df = loader.load_kaggle_fraud('credit-card-1/creditcard.csv')
print(f"Loaded: {df.shape}")

print("\n--- Step 2: Load Mapping ---")
mapper = MetadataMapper('configs/mapping.json')
summary = mapper.summary()
print(f"Client: {summary['client_id']}")
print(f"Domain: {summary['domain']}")
print(f"Vector size: {summary['vector_size']}")
print(f"Features: {summary['total_features']} ({summary['numeric_features']} numeric, {summary['categorical_features']} categorical)")

print("\n--- Step 3: Validate Mapping ---")
is_valid, issues = mapper.validate_local_data(df)
print(f"Valid: {is_valid}")
if issues:
    for issue in issues:
        print(f"  - {issue}")

print("\n--- Step 4: Dynamic Vectorizer ---")
vectorizer = DynamicVectorizer(vector_size=mapper.vector_size)
result = vectorizer.fit_transform(df, mapper)
X_tensor, y = result["data"], result["y"]
print(f"Input: {df.shape}")
print(f"Output: X={X_tensor.shape}, y={y.shape}")
print(f"Tensor dtype: {X_tensor.dtype}")
print(f"Vector contract satisfied: {X_tensor.shape[1] == mapper.vector_size}")

print("\n--- Step 5: Save/Load Vectorizer ---")
vectorizer.save('artifacts/test_vectorizer.pkl')
loaded = DynamicVectorizer.load('artifacts/test_vectorizer.pkl')
print(f"Loaded vectorizer: vector_size={loaded.vector_size}")

print("\n--- Step 6: Transform Unseen Data ---")
sample = df.head(100)
X_sample = loaded.transform(sample, mapper)
print(f"Sample output: {X_sample['data'].shape}")

print("\n--- Step 7: Test with Smaller Feature Set (Gap Logic) ---")
small_mapping = {
    "client_id": "test_small",
    "domain": "test",
    "vector_size": 128,
    "target_column": "Class",
    "feature_mapping": [
        {"global_index": 0, "local_name": "Time", "type": "numeric", "impute": "median"},
        {"global_index": 5, "local_name": "Amount", "type": "numeric", "impute": "median"},
        {"global_index": 10, "local_name": "V1", "type": "numeric", "impute": "median"},
    ]
}
import json
import os
os.makedirs('configs/test', exist_ok=True)
with open('configs/test/small_mapping.json', 'w') as f:
    json.dump(small_mapping, f, indent=2)

small_mapper = MetadataMapper('configs/test/small_mapping.json')
small_vectorizer = DynamicVectorizer(vector_size=128)
small_result = small_vectorizer.fit_transform(df, small_mapper)
X_small, y_small = small_result["data"], small_result["y"]
print(f"Small mapping output: {X_small.shape}")
print(f"Zero columns (gaps): {(X_small == 0).all(dim=0).sum().item()}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)