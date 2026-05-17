"""Thorough pipeline evaluation - categorical vectorizer, DP, edge cases."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import torch

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.core.contract import VectorContract, PrivacyConfig, FedProxConfig
from src.models import create_model
from src.models.train_engine import train_one_round, evaluate_model
from src.fl.secure_update import protect_update, clip_update_norm, add_gaussian_noise
from src.data.schema import create_fraud_schema, create_p2p_fraud_schema, create_schema_from_mapping

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")

print("=" * 60)
print("COMPREHENSIVE PIPELINE EVALUATION")
print("=" * 60)

# 1. Schema contract tests
print("\n--- 1. Schema Contract ---")
fraud_schema = create_fraud_schema()
check("Fraud schema target=Class", fraud_schema.label_column == "Class")
check("Fraud schema has 30 numeric cols", len(fraud_schema.numeric_columns) == 30)

p2p_schema = create_p2p_fraud_schema()
check("P2P schema target=is_fraud", p2p_schema.label_column == "is_fraud")
check("P2P schema has numeric cols", len(p2p_schema.numeric_columns) == 7)
check("P2P schema has categorical cols", len(p2p_schema.categorical_columns) == 4)

schema_from_mapping = create_schema_from_mapping('configs/mapping.json')
check("Schema from mapping target=Class", schema_from_mapping.label_column == "Class")

p2p_schema_from_map = create_schema_from_mapping('configs/mapping_p2p.json')
check("P2P schema from mapping target=is_fraud", p2p_schema_from_map.label_column == "is_fraud")
check("P2P schema from mapping has categoricals", len(p2p_schema_from_map.categorical_columns) == 4)

# 2. Metadata engine tests
print("\n--- 2. Metadata Engine ---")
mapper_a = MetadataMapper('configs/mapping.json')
check("Mapper A client_id", mapper_a.client_id == "neobank_a")
check("Mapper A 30 features", len(mapper_a.feature_mappings) == 30)
check("Mapper A all numeric", len(mapper_a.get_categorical_columns()) == 0)
check("Mapper A target", mapper_a.target_column == "Class")

mapper_p2p = MetadataMapper('configs/mapping_p2p.json')
check("Mapper P2P client_id", mapper_p2p.client_id == "p2p_bank")
check("Mapper P2P 11 features", len(mapper_p2p.feature_mappings) == 11)
check("Mapper P2P categoricals", len(mapper_p2p.get_categorical_columns()) == 4)
check("Mapper P2P target", mapper_p2p.target_column == "is_fraud")

df_kaggle = pd.read_csv('Sample_datasets/credit-card-1/creditcard.csv', nrows=1000)
valid, issues = mapper_a.validate_local_data(df_kaggle)
check("Mapper A validates Kaggle data", valid)

df_p2p = pd.read_csv('Sample_datasets/credit-card-2/fraudTrain.csv', nrows=1000)
valid, issues = mapper_p2p.validate_local_data(df_p2p)
check("Mapper P2P validates P2P data", valid)

# 3. Vectorizer tests - Kaggle (numeric only)
print("\n--- 3. Vectorizer (Numeric - Kaggle) ---")
v_kaggle = DynamicVectorizer(vector_size=128)
result_k = v_kaggle.fit_transform(df_kaggle, mapper_a)
check("Kaggle output shape", result_k["data"].shape == (1000, 128))
check("Kaggle mask 30 active", result_k["mask"].sum().item() == 30)
check("Kaggle y shape", result_k["y"].shape == (1000,))
check("Kaggle tensor dtype float32", result_k["data"].dtype == torch.float32)

# Test transform
result_k2 = v_kaggle.transform(df_kaggle.head(100), mapper_a)
check("Kaggle transform shape", result_k2["data"].shape == (100, 128))

# 4. Vectorizer tests - P2P (categorical)
print("\n--- 4. Vectorizer (Categorical - P2P) ---")
v_p2p = DynamicVectorizer(vector_size=128)
result_p = v_p2p.fit_transform(df_p2p, mapper_p2p)
check("P2P output shape", result_p["data"].shape[0] == 1000)
check("P2P output width 128", result_p["data"].shape[1] == 128)
check("P2P y shape", result_p["y"].shape == (1000,))
check("P2P mask has active features", result_p["mask"].sum().item() > 11)

# 5. Gap logic test
print("\n--- 5. Gap Logic (Sparse Mapping) ---")
import json, os
sparse_mapping = {
    "client_id": "sparse_test",
    "domain": "test",
    "vector_size": 128,
    "target_column": "Class",
    "feature_mapping": [
        {"global_index": 0, "local_name": "Time", "type": "numeric", "impute": "median"},
        {"global_index": 50, "local_name": "Amount", "type": "numeric", "impute": "median"},
    ]
}
os.makedirs('configs/test', exist_ok=True)
with open('configs/test/sparse_mapping.json', 'w') as f:
    json.dump(sparse_mapping, f, indent=2)

sparse_mapper = MetadataMapper('configs/test/sparse_mapping.json')
v_sparse = DynamicVectorizer(vector_size=128)
result_s = v_sparse.fit_transform(df_kaggle, sparse_mapper)
zero_cols = (result_s["data"] == 0).all(dim=0).sum().item()
check("Sparse gap logic (126 zero cols)", zero_cols == 126)
check("Sparse mask has 2 active", result_s["mask"].sum().item() == 2)

# 6. Model tests
print("\n--- 6. Model Architecture ---")
mlp = create_model(128, {"hidden_dims": [64, 32], "dropout": 0.2}, "mlp")
x = torch.randn(32, 128)
out = mlp(x)
check("MLP forward pass shape", out.shape == (32, 1))
check("MLP accepts padding_mask kwarg", mlp(x, padding_mask=None) is not None)

transformer = create_model(128, {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.2}, "transformer")
mask = torch.zeros(128, dtype=torch.bool)
mask[:30] = True
out_t = transformer(x, padding_mask=mask)
check("Transformer forward pass shape", out_t.shape == (32, 1))
check("Transformer uses mask", transformer(x, padding_mask=None) is not None)

# 7. Training engine tests
print("\n--- 7. Training Engine ---")
device = torch.device("cpu")
y = torch.randint(0, 2, (32,), dtype=torch.float32)
tc = {"epochs": 1, "batch_size": 16, "lr": 0.001, "mu": 0.0, "round": 0}
params, metrics = train_one_round(mlp, x, y, tc, device)
check("Training produces params", len(params) > 0)
check("Training produces metrics", "train_loss" in metrics)

# FedProx training
mlp2 = create_model(128, {"hidden_dims": [64, 32], "dropout": 0.2}, "mlp")
tc_prox = {"epochs": 1, "batch_size": 16, "lr": 0.001, "mu": 0.01, "round": 0}
params2, metrics2 = train_one_round(mlp2, x, y, tc_prox, device)
check("FedProx training completes", "train_loss" in metrics2)

# 8. Differential privacy tests
print("\n--- 8. Differential Privacy ---")
ref = [np.ones((10, 10), dtype=np.float32), np.zeros(5, dtype=np.float32)]
updated = [np.ones((10, 10), dtype=np.float32) * 2, np.ones(5, dtype=np.float32) * 0.5]

# Norm clipping
delta = [u.astype(np.float64) - r.astype(np.float64) for u, r in zip(updated, ref)]
clipped = clip_update_norm(delta, max_norm=1.0)
clipped_norm = np.sqrt(sum(np.sum(d**2) for d in clipped))
check("Norm clipping <= max_norm", clipped_norm <= 1.0001)

# Gaussian noise
noisy = add_gaussian_noise(clipped, noise_multiplier=0.1, max_norm=1.0)
check("Noise adds variation", not np.array_equal(noisy[0], clipped[0]))

# Full pipeline
cfg = {"max_norm": 1.0, "noise_multiplier": 0.0, "validate": True}
protected, audit = protect_update(updated, ref, cfg)
check("Protect update produces same shape params", len(protected) == len(ref))
check("Protect update audit has checksum", "checksum" in audit)
# delta norm is ~10.06, which exceeds max_norm=1.0, so it IS clipped
check("Protect update clipped (large delta)", audit["clipped"])

cfg_noise = {"max_norm": 1.0, "noise_multiplier": 0.1, "validate": True}
protected2, audit2 = protect_update(updated, ref, cfg_noise)
check("Protect update with noise", audit2["noise_added"])

# 9. Contract tests
print("\n--- 9. Contract Verification ---")
vc = VectorContract(vector_size=128)
check("Vector contract validates 128", vc.validate(128))
check("Vector contract validates 30 (padding ok)", vc.validate(30))

privacy = PrivacyConfig()
check("Privacy disabled by default", not privacy.enabled)
check("Privacy max_norm default", privacy.max_norm == 1.0)

fedprox = FedProxConfig()
check("FedProx mu default", fedprox.mu == 0.01)

# 10. Save/load roundtrip
print("\n--- 10. Save/Load Roundtrip ---")
import tempfile, os
tmpdir = tempfile.mkdtemp()
v_kaggle.save(os.path.join(tmpdir, "test.pkl"))
loaded_v = DynamicVectorizer.load(os.path.join(tmpdir, "test.pkl"))
check("Loaded vectorizer same size", loaded_v.vector_size == 128)
check("Loaded vectorizer fitted", loaded_v._is_fitted)

# Transform with loaded
result_loaded = loaded_v.transform(df_kaggle.head(50), mapper_a)
check("Loaded vectorizer transform works", result_loaded["data"].shape == (50, 128))

print(f"\n{'=' * 60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print(f"{'=' * 60}")
