"""Architectural audit: check P2P categorical cardinality and index collisions."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

# Check P2P categorical cardinalities
df = pd.read_csv('Sample_datasets/credit-card-2/fraudTrain.csv', nrows=100000)
print("=== P2P Categorical Cardinalities (first 100k rows) ===")
for col in ['category', 'gender', 'state', 'job']:
    nunique = df[col].nunique()
    print(f"  {col}: {nunique} unique values")

# Check if there are collisions in the current mapping
# Mapping: category=7, gender=9, state=10, job=11
# If category has ~14 values, it occupies indices 7-20
# gender at 9 would OVERLAP with category's positions 9-20!
print("\n=== Index Collision Analysis ===")
print("Current mapping:")
print("  category: global_index=7 (will occupy 7, 8, 9, ... up to 7+cardinality-1)")
print("  gender:   global_index=9")
print("  state:    global_index=10")
print("  job:      global_index=11")
print("")
print("If category has > 2 unique values, gender/state/job will OVERWRITE category's positions!")
print("This is a KNOWN BUG in the current P2P mapping configuration.")
print("")
print("Fix: Assign non-overlapping global indices based on cardinality:")
print("  e.g., category=7, gender=7+cat_cardinality, state=..., job=...")

# Demonstrate the collision
print("\n=== Collision Demonstration ===")
from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.data.load_data import DataLoader

mapper = MetadataMapper('configs/mapping_p2p.json')
df_small = pd.read_csv('Sample_datasets/credit-card-2/fraudTrain.csv', nrows=1000)

vectorizer = DynamicVectorizer(vector_size=128)
result = vectorizer.fit_transform(df_small, mapper)

# Check if gender/state/job overwrote category positions
cat_nunique = df_small['category'].nunique()
print(f"  Category unique values in sample: {cat_nunique}")
print(f"  Category occupies global indices: 7 to {7 + cat_nunique - 1}")
print(f"  Gender at index 9: {'OVERLAPS' if 9 < 7 + cat_nunique else 'OK'}")
print(f"  State at index 10: {'OVERLAPS' if 10 < 7 + cat_nunique else 'OK'}")
print(f"  Job at index 11: {'OVERLAPS' if 11 < 7 + cat_nunique else 'OK'}")

# Verify by checking if position 9 and 10 have non-zero values
# (they should be category one-hot encodings, not gender/state)
print(f"\n  Position 9 mean value: {result['data'][:, 9].mean().item():.4f} (should be category one-hot)")
print(f"  Position 10 mean value: {result['data'][:, 10].mean().item():.4f} (should be category one-hot)")
print(f"  Position 11 mean value: {result['data'][:, 11].mean().item():.4f} (could be category or job)")

print("\n=== AUDIT CONCLUSION ===")
print("The P2P mapping has index collisions. The vectorizer's categorical")
print("alignment logic correctly places one-hot features starting at global_index,")
print("but the mapping config doesn't account for the expanded positions.")
print("")
print("IMPACT: gender, state, and job features are overwriting category one-hot")
print("positions. Only category and numeric features are correctly placed.")
print("")
print("This does NOT affect Kaggle data (no categoricals).")
print("This only affects the P2P dataset mapping configuration.")
