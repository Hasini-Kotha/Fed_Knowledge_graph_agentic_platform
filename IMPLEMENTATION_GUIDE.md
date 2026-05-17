# Federated Learning Platform — Technical Implementation Guide

## Executive Summary

This document provides a complete technical walkthrough of the **Federated Knowledge Graph-Enhanced Agentic AI Platform for Secure Explainable Cross-Enterprise Intelligence**. We cover the implementation from raw data ingestion through the global model, including the metadata-driven architecture, Tabular Transformer model, FedProx strategy, differential privacy, and Byzantine fault tolerance.

---

## 1. Architecture Overview

### 1.1 Five-Layer Pipeline (Current Scope: Layers 1-4)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Federated Learning Layer (CURRENT SCOPE)          │
│   - Data Ingestion → Validation → Client Splitting          │
│   - Metadata Engine → Dynamic Vectorization                 │
│   - Local Training → FedProx Aggregation → Global Model     │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: Prediction Layer                                  │
│   - Calibrated risk scores from global model                │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: Knowledge Graph Layer (Future)                    │
│   - Neo4j/NetworkX EAV schema for entity relationships      │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: Explainability Layer (Future)                     │
│   - Llama 3 via Ollama for natural language audit trails    │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: Agentic Engine (Future)                           │
│   - LangGraph + ReAct for autonomous response               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Folder Structure

```
src/
├── core/                    ← THE AGNOSTIC BRAIN
│   ├── contract.py          # Global constants: VectorContract(128), FedProxConfig, PrivacyConfig
│   ├── metadata_engine.py   # MetadataMapper: mapping.json → local-to-global index translation
│   └── vectorizer.py        # DynamicVectorizer: DataFrame → torch.Tensor(128) + boolean mask
├── models/                  ← THE PREDICTION LAYER
│   ├── tab_transformer.py   # TabularMLP + TabularTransformer (with attention masking)
│   └── train_engine.py      # Training loop: FedProx, AdamW, early stopping, evaluation
├── fl/                      ← THE FEDERATED LAYER
│   ├── client.py            # Flower NumPyClient with DP protection pipeline
│   ├── strategy.py          # FedProxStrategy, TrimmedMeanStrategy, WeightedFedAvg
│   ├── manual_loop.py       # Ray-free synchronous FL simulation
│   └── secure_update.py     # Norm clipping + Gaussian noise + SHA-256 audit
├── data/                    ← THE DATA LAYER
│   ├── schema.py            # SchemaContract for tabular binary classification
│   ├── validate.py          # DataValidator: column checks, label validation, duplicates
│   ├── load_data.py         # DataLoader: Kaggle, simulated, arbitrary CSV loaders
│   └── split_clients.py     # ClientSplitter: 15% test + 3 clients + 80/20 train/val
├── evaluation/              ← THE EVALUATION LAYER
│   ├── metrics.py           # Confusion matrix, optimal threshold, metric aggregation
│   └── evaluate_global.py   # Load best checkpoint → evaluate on global test set
├── main/                    ← ENTRY POINTS
│   └── run_data_pipeline.py # Step 1: Full data pipeline
├── config/
│   └── data_config.yaml     # Pipeline configuration
└── utils/
    └── config_loader.py     # YAML/JSON config reader

configs/
├── mapping.json             # IEEE-CIS Fraud: 30 features → global indices 0-29
├── mapping_lendingclub.json # LendingClub: 12 features (numeric + categorical)
├── mapping_p2p.json         # P2P Transactions: 12 features (numeric + categorical + timestamp)
├── data_config.yaml         # Data pipeline settings
├── model_config.yaml        # Model architecture + training hyperparameters
└── fl_config.yaml           # FL strategy + round settings

data/splits/                 # Generated: client CSVs (gitignored)
artifacts/                   # Generated: checkpoints, reports (gitignored)
Sample_datasets/             # Raw source data (gitignored)
```

---

## 2. The Metadata-Driven Architecture

### 2.1 The Problem with Hard-Coded Columns

In a hard-coded system, the model architecture references specific column names like `V1`, `Amount`, `fico_score`. This means:
- Bank A (uses `fico_score`) and Bank B (uses `credit_rating`) cannot collaborate without code changes
- Adding a new dataset requires modifying every file in the codebase
- The system is locked to one domain (fraud detection)

### 2.2 The Solution: Global Feature Index Contract

We define a **fixed-size vector of 128 dimensions**. Every neobank maps its local columns to global indices:

| Global Index | Feature Description | Bank A (IEEE-CIS) | Bank B (LendingClub) | Bank C (P2P) |
|-------------|--------------------|--------------------|----------------------|---------------|
| 0 | Time/Transaction | `Time` | `unix_time` | `unix_time` |
| 1 | Amount/Credit | `V1` | `loan_amnt` | `amt` |
| 2 | Income/Risk | `V2` | `annual_inc` | `city_pop` |
| 3 | Debt/Burden | `V3` | `int_rate` | `lat` |
| 4 | Credit Score | `V4` | `dti` | `long` |
| ... | ... | ... | ... | ... |
| 8 | Employment | — | `term` (categorical) | — |
| 9 | Ownership | — | `home_ownership` (categorical) | — |

**Critical rule:** If Bank B has no mapping for index 5, the vector at position 5 is **zero-padded**. The model's attention mechanism uses a **boolean mask** to ignore these padded positions.

### 2.3 mapping.json Structure

```json
{
  "client_id": "neobank_a",
  "domain": "fintech_fraud_detection",
  "vector_size": 128,
  "target_column": "Class",
  "feature_mapping": [
    {
      "global_index": 0,
      "local_name": "Time",
      "type": "numeric",
      "impute": "median",
      "description": "Seconds elapsed from first transaction"
    },
    ...
  ]
}
```

### 2.4 MetadataMapper (src/core/metadata_engine.py)

```python
mapper = MetadataMapper("configs/mapping.json")

# Validate that the CSV has all required columns
is_valid, issues = mapper.validate_local_data(df)

# Get columns in global index order
ordered_columns = mapper.get_feature_order()  # ['Time', 'V1', 'V2', ...]

# Get numeric vs categorical columns
numeric_cols = mapper.get_numeric_columns()    # ['Time', 'V1', ...]
categorical_cols = mapper.get_categorical_columns()  # [] for IEEE-CIS
```

### 2.5 DynamicVectorizer (src/core/vectorizer.py)

The vectorizer takes a DataFrame and a MetadataMapper and produces:

```python
vectorizer = DynamicVectorizer(vector_size=128)
result = vectorizer.fit_transform(df, mapper)

result["data"]   # torch.Tensor(284807, 128) — standardized features at global indices
result["mask"]   # torch.Tensor(128,) — True for 30 active features, False for 98 padded
result["y"]      # numpy.array(284807,) — labels
```

**How it works:**
1. Builds sklearn ColumnTransformer from the mapper (StandardScaler for numeric, OneHotEncoder for categorical)
2. Applies the pipeline to get processed features
3. Places each feature at its global index position in a 128-dimensional zero array
4. Generates a boolean mask: True where a feature was placed, False where padding remains
5. If the bank has more than 128 features, PCA reduction is applied

**Save/load for inference consistency:**
```python
vectorizer.save("artifacts/global_vectorizer_kaggle.pkl")
loaded = DynamicVectorizer.load("artifacts/global_vectorizer_kaggle.pkl")
```

---

## 3. The Prediction Layer: Tabular Transformer

### 3.1 Why Tabular Transformer over MLP

MLPs treat each feature independently. Transformers use **self-attention** to learn which features interact — for example, how `Amount` and `Time` together indicate fraud patterns that neither reveals alone.

### 3.2 Architecture

```
Input (128-dim)
    │
    ▼
FeatureEmbedding
    ├── Numeric: Linear(1 → 64) per feature
    └── Categorical: nn.Embedding(cardinality → 64)
    │
    ▼
LayerNorm(64)
    │
    ▼
TransformerEncoder × 2 layers
    ├── MultiHeadAttention(nhead=4, d_model=64)
    │   └── key_padding_mask: ignores zero-padded features
    └── FeedForward(64 → 128 → 64)
    │
    ▼
Mean Pooling (over features)
    │
    ▼
MLP Head: Linear(64 → 32) → BatchNorm → ReLU → Dropout(0.2) → Linear(32 → 1)
    │
    ▼
Sigmoid → Probability
```

### 3.3 Attention Masking — The Key Innovation

When Bank A has 30 features and Bank B has 12 features, their vectors have different numbers of active positions. Without masking, the Transformer would treat Bank B's 98 zeros as "features with value 0" and learn incorrect patterns.

**Solution:** We pass a `key_padding_mask` to PyTorch's `MultiheadAttention`:

```python
def forward(self, x, padding_mask=None):
    embedded = self.feature_embedding(x)
    # padding_mask: [True, True, ..., False, False] (30 True, 98 False)

    key_padding_mask = ~padding_mask  # Invert: True = ignore
    output = self.transformer(embedded, src_key_padding_mask=key_padding_mask)
    # Padded positions get -inf attention weight → 0 after softmax
```

This means the Transformer **only attends to features that actually exist** for each client. When weights are aggregated on the server, the global model learns from the union of all features.

### 3.4 TabularMLP (Fallback)

For simpler use cases or when compute is limited:

```python
MLP: Input(128) → Linear(64) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(1) → Sigmoid
```

### 3.5 Model Factory

```python
from src.models.tab_transformer import create_model

# Transformer
model = create_model(input_dim=128, config={
    "d_model": 64, "nhead": 4, "num_layers": 2,
    "dim_feedforward": 128, "dropout": 0.2
}, model_type="transformer")

# MLP
model = create_model(input_dim=128, config={
    "hidden_dims": [64, 32], "dropout": 0.2
}, model_type="mlp")
```

---

## 4. The Training Engine

### 4.1 FedProx Loss Function

Standard cross-entropy + proximal term:

```python
loss = BCE_with_logits(y_pred, y_true)

if mu > 0:  # FedProx
    proximal_term = sum(
        ||local_param - global_param||^2
        for local_param, global_param in zip(model.parameters(), global_params)
    )
    loss = loss + (mu / 2) * proximal_term
```

The proximal term penalizes the local model for drifting too far from the global consensus. This is critical when Bank C has a very different data distribution (Non-IID) — without it, Bank C's local training could produce weights that are incompatible with Banks A and B.

### 4.2 AdamW Optimizer

AdamW (Adam with decoupled weight decay) is used instead of Adam for the Transformer:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### 4.3 Early Stopping

Each client monitors validation PR-AUC. If it doesn't improve for 5 epochs, training stops and the best weights are restored:

```python
early_stopping = EarlyStopping(patience=5, metric="pr_auc")
for epoch in range(epochs):
    train(...)
    val_metrics = evaluate(...)
    if early_stopping.step(val_metrics["pr_auc"], model):
        early_stopping.load_best(model)  # Restore best weights
        break
```

### 4.4 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents exploding gradients, especially important for Transformer architectures.

---

## 5. Federated Orchestration

### 5.1 Flower Client (src/fl/client.py)

Each client is a `fl.client.NumPyClient` that:

1. **fit()**: Receives global weights from server → sets them on local model → trains locally → clips + adds noise to weight delta → returns protected weights
2. **evaluate()**: Receives global weights → evaluates on validation data → returns loss + metrics

### 5.2 Secure Update Pipeline (src/fl/secure_update.py)

**Before sending weights to the server:**

```
Step 1: Compute delta = local_weights - global_weights
Step 2: Clip L2 norm of delta to max_norm (default: 1.0)
Step 3: Add Gaussian noise: N(0, noise_multiplier × max_norm)
Step 4: Reconstruct: protected = global_weights + noisy_delta
Step 5: Compute SHA-256 checksum for audit trail
```

**Why delta, not absolute weights?** Clipping absolute weights would destroy the model's learned representations. Clipping the delta limits how much any single client can change the global model.

**Differential Privacy guarantee:** With noise_multiplier > 0, the system provides (epsilon, delta)-DP. The noise prevents "model inversion" attacks where an adversary tries to reconstruct training data from model weights.

### 5.3 Aggregation Strategies (src/fl/strategy.py)

**FedProxStrategy:**
```
ω_global = Σ (n_i / N) × ω_i
where n_i = client i's sample size, N = total samples
```
The proximal term μ is applied during local training, not at aggregation.

**TrimmedMeanStrategy (Byzantine Fault Tolerance):**
```
1. Sort weight updates from all clients
2. Trim top β% and bottom β%
3. Average the remaining updates
```
This filters out malicious or corrupted weight updates. If one of the 3 banks is compromised, their outlier weights are trimmed before aggregation.

**WeightedFedAvg:**
Standard FedAvg with weighted metrics logging and per-round checkpointing.

### 5.4 Manual FL Loop (src/fl/manual_loop.py)

For Windows environments where Flower's Ray backend is unavailable:

```
for round in 1..20:
    for each client:
        train locally with global weights (FedProx)
        apply DP protection if enabled
    aggregate weights (weighted average or trimmed mean)
    evaluate all clients on new global model
    aggregate metrics
    save checkpoint + SHA-256 checksum
```

---

## 6. Data Layer

### 6.1 Schema Contract (src/data/schema.py)

Defines the structure for any tabular binary classification dataset:

```python
schema = SchemaContract(
    name="kaggle_mlg_ulb_fraud",
    label_column="Class",
    label_positive=1,
    label_negative=0,
    required_columns=["Class", "Amount"],
    numeric_columns=["Time", "V1", ..., "V28", "Amount"],
)
```

### 6.2 Data Validator (src/data/validate.py)

Checks:
- All required columns present
- Label column has valid values (only 0 and 1)
- No duplicate columns
- Class distribution is reasonable (not all fraud or all legit)
- Data types are consistent

### 6.3 Client Splitter (src/data/split_clients.py)

```
Total: 284,807 rows
├── Global Test (15%): 42,721 rows — untouched, for final evaluation
└── Remaining (85%): 242,086 rows
    ├── Client A (1/3): 80,695 rows → 64,556 train + 16,139 val
    ├── Client B (1/3): 80,695 rows → 64,556 train + 16,139 val
    └── Client C (1/3): 80,696 rows → 64,556 train + 16,140 val
```

Each client split preserves the fraud ratio (~0.17%) because we use random shuffling (IID split). For Non-IID, sort by a column (e.g., `Time`) before splitting.

---

## 7. Evaluation Layer

### 7.1 Metrics (src/evaluation/metrics.py)

- **ROC-AUC**: Area under ROC curve (threshold-independent)
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)
- **F1**: Harmonic mean of precision and recall
- **Optimal Threshold**: Scans 0.1-0.9 to find threshold maximizing F1

### 7.2 Global Evaluation (src/evaluation/evaluate_global.py)

1. Scans all `round_*_checkpoint.pt` files
2. Picks best checkpoint by PR-AUC
3. Loads it and evaluates on `global_test.csv`
4. Generates text + JSON reports comparing FL validation vs global holdout

---

## 8. How to Use the Platform

### 8.1 Where to Add Your Datasets

Place raw CSV files in `Sample_datasets/`:

```
Sample_datasets/
├── credit-card-1/creditcard.csv          # IEEE-CIS Fraud (already present)
├── credit-card-2/fraudTrain.csv          # P2P Transactions (already present)
├── credit-card-2/fraudTest.csv           # P2P test set (already present)
└── lendingclub/                          # Add LendingClub here
    └── loans.csv
```

### 8.2 Step 1: Run the Data Pipeline

```bash
python src/main/run_data_pipeline.py \
  --source kaggle \
  --data-dir Sample_datasets \
  --mapping-path configs/mapping.json \
  --output-dir data/splits \
  --artifacts-dir artifacts
```

This produces:
- `data/splits/client_a_train.csv`, `client_a_val.csv`
- `data/splits/client_b_train.csv`, `client_b_val.csv`
- `data/splits/client_c_train.csv`, `client_c_val.csv`
- `data/splits/global_test.csv`
- `artifacts/global_vectorizer_kaggle.pkl`
- `artifacts/global_vectorizer_p2p.pkl`

### 8.3 Step 2: Run FL Simulation

```bash
python -c "
from src.fl.manual_loop import run_manual_simulation

history = run_manual_simulation(
    client_ids=['client_a', 'client_b', 'client_c'],
    data_dir='data/splits',
    mapping_path='configs/mapping.json',
    vectorizer_path='artifacts/global_vectorizer_kaggle.pkl',
    model_config={'model_type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2},
    train_config={'epochs': 3, 'batch_size': 256, 'lr': 0.001},
    fl_config={'num_rounds': 10, 'strategy': 'fedprox', 'fedprox_mu': 0.01},
    privacy_config={'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 0.1},
    artifacts_dir='artifacts/global_model',
    strategy_type='fedprox',
)
"
```

### 8.4 Step 3: Evaluate Global Model

```bash
python -c "
from src.evaluation.evaluate_global import load_best_checkpoint, evaluate_global_model, generate_evaluation_report
from src.core.metadata_engine import MetadataMapper

checkpoint = load_best_checkpoint('artifacts/global_model', metric='pr_auc')
mapper = MetadataMapper('configs/mapping.json')

metrics = evaluate_global_model(
    checkpoint=checkpoint,
    global_test_csv='data/splits/global_test.csv',
    mapping_path='configs/mapping.json',
    vectorizer_path='artifacts/global_vectorizer_kaggle.pkl',
    model_config={'model_type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2}
)

generate_evaluation_report(
    round_number=checkpoint['round'],
    round_metrics=checkpoint['metrics'],
    global_metrics=metrics,
    output_path='artifacts/reports'
)
"
```

### 8.5 Using Different Datasets

To switch from IEEE-CIS to LendingClub:

```bash
python src/main/run_data_pipeline.py \
  --mapping-path configs/mapping_lendingclub.json \
  --data-dir Sample_datasets/lendingclub
```

No code changes — only the mapping.json changes.

---

## 9. Security & Privacy Guarantees

### 9.1 Differential Privacy

When `privacy_config.enabled = True`:
- **Norm Clipping**: Each client's weight delta is clipped to `max_norm` (default 1.0). This limits the influence of any single client.
- **Gaussian Noise**: Calibrated noise `N(0, σ)` where `σ = noise_multiplier × max_norm` is added to the clipped delta. This prevents model inversion attacks.

```python
privacy_config = {
    "enabled": True,
    "max_norm": 1.0,          # Clip threshold
    "noise_multiplier": 0.1,  # DP noise scale
}
```

### 9.2 Audit Trail

Every weight update is hashed:
```python
checksum = SHA-256(protected_weights)
```
The checksum is stored in each checkpoint, enabling verification that weights were not tampered with during transmission.

### 9.3 Byzantine Fault Tolerance

Using `TrimmedMeanStrategy`:
```python
fl_config = {"strategy": "trimmed_mean", "beta": 0.1}
```
Trims 10% from each end of the weight distribution before averaging, filtering out outlier (potentially malicious) updates.

---

## 10. Collaboration Across Competing Neobanks

### 10.1 The Problem

Bank A has tech-savvy users (online transactions, digital wallets). Bank B has traditional users (branch visits, paper checks). Bank C has high-volume micro-transactions. Each bank's data has different statistical properties (Non-IID).

### 10.2 The Solution

1. **FedProx**: The proximal term `μ = 0.01` prevents Bank C's micro-transaction data from causing the model to drift too far from the global consensus. Each bank trains locally but stays anchored to the shared model.

2. **Attention Masking**: Each bank only contributes features it actually has. The global model learns from the union of all features without any bank needing to share its raw data.

3. **DP Protection**: Even the weight updates are protected. Banks cannot reverse-engineer each other's customer data from the shared weights.

4. **Trimmed Mean**: If one bank's data is corrupted or if a malicious actor infiltrates the federation, the Trimmed Mean strategy filters out their outlier updates.

### 10.3 The Result

After 10-20 rounds of federated learning:
- Bank B can now detect fraud patterns typical of tech-savvy users (learned from Bank A)
- Bank A benefits from Bank B's traditional user patterns
- Bank C's high-volume patterns improve detection for all
- The global model achieves higher PR-AUC and ROC-AUC than any single bank's local model
- **No raw data ever leaves any bank's infrastructure**

---

## 11. Current Results

The platform has been verified end-to-end with 20 rounds of FedProx (μ=0.01):

| Component | Status | Verification |
|-----------|--------|-------------|
| Metadata Engine | ✅ Complete | 30 features mapped → 128-dim vector with 98 zero-padded slots |
| Dynamic Vectorizer | ✅ Complete | DataFrame → Tensor(128) + boolean mask, save/load verified |
| High-Cardinality Handling | ✅ Complete | TopKEncoder with "other" bucket for oversized categoricals |
| DP Accountant | ✅ Complete | RDP-based (ε, δ)-DP tracking with compositional bounds |
| TabularMLP | ✅ Complete | 10,561 parameters, trains with FedProx |
| TabularTransformer | ✅ Complete | 77,441 parameters, attention masking implemented |
| Training Engine | ✅ Complete | FedProx + AdamW + early stopping verified |
| FL Client | ✅ Complete | Secure update pipeline (clipping + noise + checksum) |
| FL Strategies | ✅ Complete | FedProx, TrimmedMean, WeightedFedAvg |
| Manual Loop | ✅ Complete | 20 rounds with DP accounting integration |
| Global Evaluation | ✅ Complete | Checkpoint loading + optimal threshold |

### 20-Round FL Results (IID, Kaggle Fraud, 284,807 rows, No DP)

| Metric | Round 1 | Round 10 | Round 20 | Best |
|--------|---------|----------|----------|------|
| ROC-AUC | 0.883 | 0.968 | 0.967 | 0.982 (holdout) |
| PR-AUC | 0.655 | 0.778 | 0.795 | 0.795 |
| F1 | 0.655 | 0.770 | 0.765 | 0.800 (R18) |
| Precision | 0.835 | 0.906 | 0.894 | 0.894 |
| Recall | 0.541 | 0.671 | 0.671 | 0.741 (R6) |

### Global Holdout Evaluation (Best: Round 20, PR-AUC=0.795)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.980 |
| PR-AUC | 0.756 |
| F1 | 0.734 |
| Precision | 0.870 |
| Recall | 0.635 |
| Accuracy | 0.999 |
| Optimal Threshold | 0.44 |

### P2P Dataset Experiments (50K sampled, 10 rounds, TabularMLP)

| Experiment | ROC-AUC | PR-AUC | F1 | DP Epsilon |
|------------|---------|--------|----|------------|
| p2p_iid_no_dp | 0.9259 | 0.8123 | 0.4714 | N/A |
| p2p_iid_dp_sigma1 | 0.6886 | 0.3140 | 0.4365 | 20.17 |
| p2p_noniid_amt_dp1 | 0.5605 | 0.2602 | 0.2973 | 20.17 |

**Key observations:**
- **DP privacy-utility tradeoff**: sigma=1.0 with full participation gives ε=20.17 at δ=1e-5 after 10 rounds. PR-AUC drops 0.81→0.31 but F1 stays reasonable (0.47→0.44).
- **Non-IID impact**: Sorting by transaction amount creates distribution shift. Combined with DP, PR-AUC drops further to 0.26, F1 to 0.30.
- **Recommendation**: Use sigma=0.5 or subsampled clients (q<1) to reduce ε while maintaining utility.

**Note:** Kaggle results are from IID splits. P2P results use 50K sampled rows with all fraud preserved.

---

## 12. What's Next (Future Work)

- [ ] Layer 3: Knowledge Graph (Neo4j EAV schema for entity relationships)
- [ ] Layer 4: Explainability (Llama 3 via Ollama for audit trails)
- [ ] Layer 5: Agentic Engine (LangGraph + ReAct for autonomous response)
- [ ] Secure Aggregation via canceling masks (Phase 2 of secure_update.py)
- [ ] Homomorphic encryption for weight transmission (research)
- [ ] Subsampled DP (q<1) for tighter privacy-utility tradeoff
- [ ] Production deployment with Flower Server (beyond simulation)
- [ ] Full-dataset P2P run (1.85M rows) with optimized training

---

## 13. Quick Reference

| Parameter | Value | Where |
|-----------|-------|-------|
| Vector Size | 128 | `configs/mapping.json` |
| FedProx μ | 0.01 | `configs/fl_config.yaml` |
| FL Rounds | 20 | `src/main/run_full_pipeline.py` |
| Local Epochs | 3 | `src/main/run_full_pipeline.py` |
| Batch Size | 256 | `configs/model_config.yaml` |
| Learning Rate | 0.001 | `configs/model_config.yaml` |
| Transformer Heads | 4 | `src/models/tab_transformer.py` |
| Transformer Layers | 2 | `src/models/tab_transformer.py` |
| Dropout | 0.2 | `configs/model_config.yaml` |
| Max Norm (DP) | 1.0 | `configs/fl_config.yaml` |
| Noise Multiplier | 0.0 (disabled) | `configs/fl_config.yaml` |
| Trimmed Mean β | 0.1 | `configs/fl_config.yaml` |

---

*Document generated: May 2026. Platform version: v1.0 (FL layer complete, KG/LLM/Agentic layers in design phase).*