# CLIENT SPLIT SUMMARY NOTE
# ====================

## Split Type: IID (Independent Identically Distributed)

The split is **IID** because we use random shuffling before distributing data to clients.
This means each client receives a random sample of the overall data distribution.

## Dataset Summary

- **Source**: Kaggle MLG-ULB Credit Card Fraud Dataset
- **Total Rows**: 284,807
- **Features**: 30 (Time + V1-V28 + Amount)
- **Label**: Class (0=legitimate, 1=fraud)
- **Fraud Ratio**: 0.0017 (0.17%)

## Fraud Ratio Per Client

| Client | Train Fraud Ratio | Val Fraud Ratio |
|--------|-----------------|-----------------|
| client_a | 0.00170 (0.17%) | 0.00173 (0.17%) |
| client_b | 0.00160 (0.16%) | 0.00180 (0.18%) |
| client_c | 0.00186 (0.19%) | 0.00173 (0.17%) |
| Global Test | - | 0.00173 (0.17%) |

## Dataset Shape and Class Distribution

| Split | Rows | Fraud | Legitimate | Fraud Ratio |
|-------|------|-------|----------|-----------|
| client_a_train | 64,556 | 110 | 64,446 | 0.00170 |
| client_a_val | 16,139 | 28 | 16,111 | 0.00173 |
| client_b_train | 64,556 | 103 | 64,453 | 0.00160 |
| client_b_val | 16,139 | 29 | 16,110 | 0.00180 |
| client_c_train | 64,556 | 120 | 64,436 | 0.00186 |
| client_c_val | 16,140 | 28 | 16,112 | 0.00173 |
| global_test | 42,721 | 74 | 42,647 | 0.00173 |
| **Total** | **284,807** | **492** | **284,315** | **0.00173** |

## Split Strategy

1. **15% Global Holdout Test**: 42,721 rows (untouched, for final evaluation)
2. **85% Remaining Data**: 242,086 rows split across 3 clients
3. **Per-Client Split**: 80% train, 20% validation

## Assumptions and Limitations

1. **Random Seed**: Fixed at 42 for reproducibility
2. **No stratification by Time**: Could create non-IID by sorting by Time column
3. **Class imbalance preserved**: Each client has similar fraud ratio (~0.17%)
4. **No data leakage**: Global test set is completely untouched
5. **Duplicate rows**: 1,081 duplicates found in original (removed or preserved - not deduplicated)

## Files Generated

- `data/splits/client_a_train.csv` - 64,556 rows
- `data/splits/client_a_val.csv` - 16,139 rows
- `data/splits/client_b_train.csv` - 64,556 rows
- `data/splits/client_b_val.csv` - 16,139 rows
- `data/splits/client_c_train.csv` - 64,556 rows
- `data/splits/client_c_val.csv` - 16,140 rows
- `data/splits/global_test.csv` - 42,721 rows
- `data/splits/split_summary.json` - Statistics

## Non-IID Option

To create a **non-IID** split (realistic when banks have different fraud patterns):

```python
from src.data.split_clients import ClientSplitter
splitter = ClientSplitter(num_clients=3, test_ratio=0.15, val_ratio=0.20, random_seed=42)
client_splits = splitter.split(df, label_col='Class', non_iid=True, sort_by='Time')
```

This sorts by transaction time before splitting, so earlier transactions go to client_a,
middle to client_b, and later to client_c - simulating different time periods.

## Next Steps for Person 2 (Training) and Person 3 (Federated)

Load data:
```python
import pandas as pd
client_a_train = pd.read_csv('data/splits/client_a_train.csv')
client_a_val = pd.read_csv('data/splits/client_a_val.csv')
# etc.
```