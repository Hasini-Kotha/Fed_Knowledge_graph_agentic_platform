"""Test P2P vectorization with fixed non-overlapping mapping."""
import sys
sys.path.insert(0, '.')
import pandas as pd
from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer

mapper = MetadataMapper('configs/mapping_p2p.json')
df = pd.read_csv('Sample_datasets/credit-card-2/fraudTrain.csv', nrows=5000)
vectorizer = DynamicVectorizer(vector_size=128)
result = vectorizer.fit_transform(df, mapper)
print(f"Output shape: {result['data'].shape}")
print(f"Active mask: {result['mask'].sum().item()}/128")
mask = result['mask']
active = mask.nonzero(as_tuple=False).squeeze().tolist()
if isinstance(active, int):
    active = [active]
print(f"Active indices: {active[:20]}... ({len(active)} total)")
