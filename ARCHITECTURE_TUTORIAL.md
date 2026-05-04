# Federated Learning Platform — Complete Architecture Tutorial

## The Core Design Principle

This codebase solves one problem: **how to train a single global AI model across multiple organizations that refuse to share raw data.**

The solution uses three architectural ideas:
1. **Metadata-driven processing** — no column names are hard-coded anywhere
2. **Fixed-size vector contract** — all data converges to 128-dimensional tensors
3. **Federated aggregation** — only weight updates (not data) leave each organization

---

## How to Read This Tutorial

Each folder is explained from **why it exists** to **how it connects to everything else**. If you can understand the flow from `configs/mapping.json` → `src/core/vectorizer.py` → `src/fl/manual_loop.py` → `artifacts/global_model/`, you understand the entire platform.

---

## 1. The Contract Layer — `src/core/`

### `src/core/contract.py` — The Rulebook

This file defines the immutable rules every component must follow. It contains:

- **`VectorContract(vector_size=128)`** — Every dataset, regardless of source, must produce tensors of exactly 128 dimensions. If a bank has 30 features, the remaining 98 are zero-padded. If another has 150, PCA reduces to 128.
- **`FeatureType`** enum — `numeric`, `categorical`, `timestamp`, `boolean`. These tell the vectorizer how to preprocess each column.
- **`ImputationStrategy`** enum — `median`, `mean`, `mode`, `constant`, `drop`. Missing values are handled per the mapping config.
- **`PrivacyConfig`** — DP parameters (enabled/disabled, max_norm, noise_multiplier).
- **`FedProxConfig`** — The proximal term μ=0.01 that prevents Non-IID clients from drifting apart.

**Why this file exists:** Without a contract, every team would make different assumptions about tensor shapes, feature types, and privacy settings. This file is the single source of truth.

### `src/core/metadata_engine.py` — The Universal Translator

This is the most important file in the entire codebase. It turns a JSON configuration into a working feature map:

```json
{"global_index": 0, "local_name": "Time", "type": "numeric", "impute": "median"}
{"global_index": 1, "local_name": "V1", "type": "numeric", "impute": "median"}
```

The `MetadataMapper` class:
1. Loads `mapping.json` from the configs directory
2. Validates that the CSV has every column listed in the mapping
3. Separates numeric vs categorical columns for preprocessing
4. Provides `get_feature_order()` — columns sorted by global index
5. Provides `get_local_to_index()` — reverse lookup from column name to position

**Why this file exists:** Bank A calls their column `fico_score`. Bank B calls theirs `credit_rating`. Bank C doesn't have this column at all. The `MetadataMapper` translates each bank's local vocabulary into a shared global index space. The model never sees "fico_score" or "credit_rating" — it only sees "feature at index 4".

**How it handles dynamic datasets:** Add a new `mapping_newbank.json` that maps the new bank's columns to global indices. No Python code changes. The `MetadataMapper` is completely data-agnostic.

### `src/core/vectorizer.py` — The Feature Assembler

The `DynamicVectorizer` takes a DataFrame and a `MetadataMapper` and produces:

```python
{
    "data": torch.Tensor(batch_size, 128),   # Features at their global indices
    "mask": torch.Tensor(128),               # True=active feature, False=zero-padded
    "y": numpy.array(batch_size,)            # Labels
}
```

**How it works (the critical pipeline):**

1. **Builds an sklearn ColumnTransformer** from the mapper:
   - Numeric columns → `SimpleImputer(median)` → `StandardScaler()`
   - Categorical columns → `SimpleImputer(mode)` → `OneHotEncoder()`

2. **Processes the DataFrame** through this pipeline → gets a matrix of shape `(rows, processed_features)`

3. **Aligns to global indices:**
   ```python
   result = zeros(rows, 128)
   for i, feature in enumerate(features):
       result[:, feature.global_index] = processed[:, i]
   ```
   This is where magic happens. If Bank B's `annual_inc` maps to global index 1, its standardized values land at position 1 in every row.

4. **Generates the boolean mask:**
   ```python
   mask = zeros(128, dtype=bool)
   for feature in features:
       mask[feature.global_index] = True
   ```

5. **Handles categorical expansion:** OneHot encoding turns `category` (14 values) into 14 binary columns. These occupy positions `global_index` through `global_index + cardinality - 1`.

6. **PCA fallback:** If a bank has more than 128 features after OneHot expansion, PCA reduces to exactly 128.

**Why this file exists:** It's the bridge between raw CSV data and the neural network. Without it, you'd need a custom preprocessing script for every dataset. With it, any CSV + any mapping.json = valid input tensor.

---

## 2. The Data Layer — `src/data/`

### `src/data/schema.py` — The Data Contract

Defines `SchemaContract` — a dataclass that specifies:
- Which column is the label (`Class` vs `is_fraud` vs `loan_status`)
- What values represent positive/negative class (1/0)
- Which columns are required, numeric, categorical, identifiers, timestamps

**Critical function: `create_schema_from_mapping(mapping_path)`**

This derives the schema entirely from the mapping.json file. It reads the mapping, extracts numeric and categorical column names, identifies the target column, and builds a `SchemaContract`. This means **the schema is not hard-coded** — it's generated from the same configuration that drives the vectorizer.

### `src/data/load_data.py` — The Data Loader

`DataLoader` class with methods for each known dataset:
- `load_kaggle_fraud()` — credit-card-1/creditcard.csv
- `load_p2p_fraud()` — credit-card-2/fraudTrain.csv + fraudTest.csv (merged)
- `load_creditcard_2023()` — credit-card-3/creditcard_2023.csv
- `load_by_name(name)` — dispatches to the correct loader by string name

Each loader returns a plain `pd.DataFrame`. The loaders are the **only** place where dataset-specific paths exist.

### `src/data/validate.py` — The Quality Gate

`DataValidator` checks:
1. All required columns present
2. Label column has only valid values (0 and 1)
3. No duplicate column names
4. Class distribution is reasonable (not all one class)
5. Data types match schema expectations

**Why this file exists:** Before spending 30 minutes training a model, verify the data is valid in 2 seconds. Catches "your CSV has 'Class' spelled 'class'" before it becomes a silent bug.

### `src/data/split_clients.py` — The Federation Simulator

`ClientSplitter` takes one big dataset and splits it into 3 simulated banks:

```
284,807 rows
├── 15% global test (42,721 rows) — held out, never touched during training
└── 85% remaining (242,086 rows)
    ├── Client A: 80,695 → 64,556 train + 16,139 val
    ├── Client B: 80,695 → 64,556 train + 16,139 val
    └── Client C: 80,696 → 64,556 train + 16,140 val
```

Supports both **IID** (random shuffle) and **Non-IID** (sort by a column, then split sequentially) distributions.

**Why this file exists:** In production, each bank already has its own data. For simulation, we split one dataset to mimic this. The `non_iid=True` mode lets us test how the system handles distribution shift.

---

## 3. The Model Layer — `src/models/`

### `src/models/tab_transformer.py` — Two Architectures, One Interface

**TabularMLP** (the workhorse):
```
Input(128) → Linear(64) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(1) → Sigmoid
```
10,561 parameters. Fast, reliable, production-ready.

**TabularTransformer** (the advanced option):
```
Input(128) → FeatureEmbedding(1→64 per feature)
           → TransformerEncoder(2 layers, 4 heads, d_model=64)
           → Mean Pooling → Linear(64→32) → Linear(32→1) → Sigmoid
```
77,441 parameters. Learns feature interactions through attention.

**The critical design:** Both models implement the same interface:
- `forward(x, padding_mask=None)` → logits
- `predict_proba(x, padding_mask=None)` → probabilities
- `get_parameters()` → list of weight tensors (for FL)
- `set_parameters(params)` → load weights from another model

The `padding_mask` parameter is what enables heterogeneous clients. When Client B has only 12 active features out of 128, the mask tells the Transformer to ignore the other 116 zero-padded positions:

```python
key_padding_mask = ~padding_mask  # Invert: True = ignore
output = transformer(embedded, src_key_padding_mask=key_padding_mask)
# Positions where mask is False get -inf attention → 0 after softmax
```

**Why this file exists:** The `create_model()` factory lets you switch between MLP and Transformer by changing one config value. Both conform to the same interface, so the FL loop doesn't care which one you use.

### `src/models/train_engine.py` — The Training Loop

Contains everything needed to train and evaluate:

1. **`train_one_epoch()`** — Single pass through data with:
   - BCE loss with logits
   - FedProx proximal term (if μ > 0): penalizes drift from global weights
   - Gradient clipping at max_norm=1.0

2. **`evaluate_model()`** — Computes ROC-AUC, PR-AUC, F1, precision, recall, accuracy

3. **`train_one_round()`** — Multiple epochs with:
   - AdamW optimizer (weight decay 1e-4)
   - Early stopping on validation PR-AUC (patience=5)
   - Returns (parameters, metrics)

4. **`save_local_checkpoint()` / `load_local_checkpoint()`** — PyTorch checkpoint save/load

**Why this file exists:** It's the shared training engine used by both local baseline training and federated client training. Single source of truth for how models are trained.

---

## 4. The Federated Layer — `src/fl/`

### `src/fl/secure_update.py` — The Privacy Shield

The protection pipeline applied to every weight update before it leaves a client:

```
Step 1: delta = local_weights - global_weights
Step 2: clip L2 norm of delta to max_norm (default 1.0)
Step 3: add Gaussian noise N(0, noise_multiplier × max_norm)
Step 4: protected = global_weights + noisy_delta
Step 5: SHA-256 checksum for audit trail
```

**Why delta, not absolute weights?** Clipping absolute weights would destroy learned representations. Clipping the delta limits how much any single client can influence the global model. This is the standard DP-SGD approach.

### `src/fl/strategy.py` — The Aggregation Rules

Three strategies for combining client weights:

1. **WeightedFedAvg** — Standard FedAvg: `ω_global = Σ(n_i/N) × ω_i`
2. **FedProxStrategy** — Same as FedAvg, but local training includes proximal term
3. **TrimmedMeanStrategy** — Sort weights, trim top/bottom β%, average the rest. Filters out malicious or corrupted updates.

### `src/fl/manual_loop.py` — The FL Orchestrator

This is the main execution engine. Flower's built-in simulation uses Ray, which doesn't work on Windows/Python 3.13. This manual loop implements the same logic:

```
for round in 1..20:
    for each client:
        create local model
        set global weights
        train locally (FedProx, 3 epochs)
        apply DP protection if enabled
    aggregate weights (weighted average or trimmed mean)
    evaluate all clients on new global model
    save checkpoint with SHA-256 checksum
```

**Why this file exists:** It's the production execution path for this platform. No Flower, no Ray — just clean synchronous FL that works everywhere.

---

## 5. The Evaluation Layer — `src/evaluation/`

### `src/evaluation/metrics.py` — Metric Utilities

- `compute_optimal_threshold()` — Scans thresholds 0.1–0.9 to find the one maximizing F1
- `print_metrics_report()` — Formatted console output
- `aggregate_client_metrics()` — Weighted average across clients

### `src/evaluation/evaluate_global.py` — Final Assessment

1. Scans all `round_*_checkpoint.pt` files
2. Picks the best by PR-AUC
3. Evaluates on the global holdout test set (never seen during training)
4. Generates JSON + text reports comparing FL validation vs holdout

---

## 6. The Entry Points — `src/main/`

### `src/main/run_data_pipeline.py` — Step 1: Data Preparation

The 5-phase pipeline:
1. **Load** raw CSV via `DataLoader`
2. **Validate mapping** — ensure mapping.json columns exist in the CSV
3. **Validate data** — schema checks, class distribution, duplicates
4. **Split** — 15% test + 3 clients with 80/20 train/val
5. **Fit vectorizer** — learn scaler parameters on combined training data, save to pickle

**Usage:**
```bash
python src/main/run_data_pipeline.py --source kaggle --mapping-path configs/mapping.json
python src/main/run_data_pipeline.py --source p2p --mapping-path configs/mapping_p2p.json
```

### `src/main/run_full_pipeline.py` — The One-Click Run

Phases: FL simulation (20 rounds) → global evaluation → report generation.

This is the file you run to get end-to-end results.

---

## 7. The Configuration Layer — `configs/`

### `configs/mapping.json` — The Feature Map

This is **the single file you change** to support a new dataset. Example for Kaggle fraud:

```json
{
  "client_id": "neobank_a",
  "domain": "fintech_fraud_detection",
  "vector_size": 128,
  "target_column": "Class",
  "feature_mapping": [
    {"global_index": 0, "local_name": "Time", "type": "numeric", "impute": "median"},
    {"global_index": 1, "local_name": "V1", "type": "numeric", "impute": "median"},
    ...
    {"global_index": 29, "local_name": "Amount", "type": "numeric", "impute": "median"}
  ]
}
```

### `configs/mapping_p2p.json` — P2P Mapping

Same structure, different columns. Categorical features with non-overlapping global indices:
- `category` at 7 (occupies 7–20, 14 values)
- `gender` at 21 (occupies 21–22, 2 values)
- `state` at 23 (occupies 23–72, 50 values)
- `job` at 73 (occupies 73–127, top-54 values)

### `configs/model_config.yaml` — Hyperparameters
### `configs/fl_config.yaml` — FL Settings
### `configs/data_config.yaml` — Pipeline Defaults

---

## 8. The Data Flow — End to End

Here is the complete journey of data through the platform:

```
Sample_datasets/credit-card-1/creditcard.csv
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. DataLoader.load_kaggle_fraud()       │
│    → pd.DataFrame(284807, 31)           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. DataValidator.validate(df)           │
│    → Check labels, types, distribution  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. MetadataMapper("mapping.json")       │
│    → 30 features → global indices 0-29  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 4. ClientSplitter.split(df)             │
│    → 3 clients + global test set        │
│    → Saved as data/splits/*.csv         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 5. DynamicVectorizer.fit_transform()    │
│    → DataFrame → torch.Tensor(N, 128)   │
│    → + boolean mask (30 True, 98 False) │
│    → Saved as artifacts/global_vectorizer.pkl
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 6. Manual FL Loop (20 rounds)           │
│    For each client:                     │
│      - Load CSV → vectorizer.transform()│
│      - Create model → train (FedProx)   │
│      - Clip delta + add noise (DP)      │
│    Aggregate: weighted average          │
│    Save: round_001_checkpoint.pt        │
│          ...                            │
│          round_020_checkpoint.pt        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 7. evaluate_global.py                   │
│    - Pick best checkpoint by PR-AUC     │
│    - Evaluate on global_test.csv        │
│    - Generate report                    │
│    - artifacts/reports/*.json + .txt    │
└─────────────────────────────────────────┘
```

---

## 9. How Dynamic Datasets Work

The platform supports **any** tabular binary classification dataset. Here's exactly what you need to do to add a new one:

### Step 1: Place the CSV

```
Sample_datasets/my_dataset/data.csv
```

### Step 2: Create a mapping

```json
{
  "client_id": "my_org",
  "domain": "my_domain",
  "vector_size": 128,
  "target_column": "my_label",
  "feature_mapping": [
    {"global_index": 0, "local_name": "feature_a", "type": "numeric", "impute": "median"},
    {"global_index": 1, "local_name": "feature_b", "type": "categorical", "impute": "mode"},
    ...
  ]
}
```

**Rules for global indices:**
- Numeric features occupy exactly 1 position each
- Categorical features at index N occupy positions N, N+1, ..., N+cardinality-1
- No overlaps between features
- All indices must be < 128

### Step 3: Add a loader (optional)

If you want `load_by_name("my_dataset")` to work, add a method to `DataLoader`. Otherwise, use `load_tabular_dataset("my_dataset/data.csv")`.

### Step 4: Run

```bash
python src/main/run_data_pipeline.py --source my_dataset --mapping-path configs/mapping_my_dataset.json
python src/main/run_full_pipeline.py
```

**Zero Python code changes.** The entire platform is driven by the mapping.json file.

---

## 10. Code Quality Principles

### Readability

- **Every file has a docstring** explaining its purpose in 1-2 sentences
- **Every function has type hints** and docstrings with Args/Returns
- **No magic numbers** — vector_size=128, mu=0.01 are in contract.py or config files
- **Descriptive variable names** — `X_tensor`, `padding_mask`, `global_params`, not `x`, `m`, `g`

### Modularity

```
src/core/      ← Domain-agnostic engine (no data knowledge)
src/data/      ← Data loading, validation, splitting
src/models/    ← Neural network architectures
src/fl/        ← Federated orchestration and security
src/evaluation/← Metrics and reporting
src/main/      ← Entry points (thin wrappers)
src/utils/     ← General utilities
```

Each layer depends only on the layers below it. `src/fl/` imports from `src/models/` and `src/core/`, but never from `src/data/` or `src/main/`.

### Maintainability

- **Single source of truth** for each concept:
  - Feature types → `contract.py`
  - Schema → `schema.py` (derived from mapping)
  - Training logic → `train_engine.py`
  - Aggregation → `strategy.py`
- **Configuration over code** — change behavior via JSON/YAML, not Python
- **Tests for every layer** — `test_metadata_engine.py`, `test_model.py`, `eval_pipeline.py`
- **Lazy imports** — `src/fl/__init__.py` uses `__getattr__` to avoid requiring Flower when only using the manual loop

---

## 11. Current Results Summary

| Dataset | Rows | Features | Rounds | ROC-AUC (holdout) | PR-AUC (holdout) |
|---------|------|----------|--------|-------------------|------------------|
| Kaggle Fraud | 284,807 | 30 numeric | 20 | 0.980 | 0.756 |

**Model:** TabularMLP (10,561 params), FedProx (μ=0.01), 3 clients, IID split
**DP:** Disabled (noise_multiplier=0.0)
**Threshold:** Optimal at 0.44 (vs default 0.50)

---

## 12. File Inventory at a Glance

| File | Purpose | Lines |
|------|---------|-------|
| `src/core/contract.py` | Global constants and enums | 85 |
| `src/core/metadata_engine.py` | JSON → feature mapping | 269 |
| `src/core/vectorizer.py` | DataFrame → Tensor(128) | 345 |
| `src/data/schema.py` | Data contracts | 321 |
| `src/data/load_data.py` | CSV loaders | 350 |
| `src/data/validate.py` | Data quality checks | 283 |
| `src/data/split_clients.py` | 3-client federation split | 430 |
| `src/models/tab_transformer.py` | MLP + Transformer models | 281 |
| `src/models/train_engine.py` | Training loop + evaluation | 330 |
| `src/fl/secure_update.py` | DP weight protection | 188 |
| `src/fl/strategy.py` | FedAvg/FedProx/TrimmedMean | 396 |
| `src/fl/manual_loop.py` | FL orchestration | 273 |
| `src/fl/client.py` | Flower NumPyClient wrapper | 144 |
| `src/evaluation/metrics.py` | F1, AUC, threshold optimization | 100 |
| `src/evaluation/evaluate_global.py` | Holdout evaluation + reports | 136 |
| `src/main/run_data_pipeline.py` | Step 1: data preparation | 203 |
| `src/main/run_full_pipeline.py` | End-to-end execution | 99 |
| `configs/mapping.json` | Kaggle feature map | 38 |
| `configs/mapping_p2p.json` | P2P feature map | 20 |
| `configs/mapping_lendingclub.json` | LendingClub feature map | 20 |

**Total:** 20 core files, ~4,250 lines of production code.

---

## 13. What to Change, What Not to Touch

### ✅ Safe to change
- `configs/mapping*.json` — add new datasets
- `configs/*.yaml` — tune hyperparameters
- `src/main/*.py` — create new entry points
- `Sample_datasets/` — add raw data

### ⚠️ Change with tests
- `src/core/vectorizer.py` — the most complex file; test with `eval_pipeline.py`
- `src/fl/manual_loop.py` — core orchestration; test with `test_fl_simulation.py`
- `src/models/tab_transformer.py` — model architecture; test with `test_model.py`

### ❌ Don't change without good reason
- `src/core/contract.py` — changing the contract breaks everything downstream
- `src/data/schema.py` — schema changes require validation updates
- `src/fl/secure_update.py` — DP correctness is critical

---

*This platform is production-grade for the FL layer (Layers 1-2). The Knowledge Graph (Layer 3), Explainability (Layer 4), and Agentic Engine (Layer 5) are designed but not yet implemented.*
