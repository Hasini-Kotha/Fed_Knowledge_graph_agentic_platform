"""Non-IID Experiment Runner — P2P and LendingClub with varying distribution shift.

Runs federated learning experiments with Non-IID data distributions.
Supports:
- IID baseline (random shuffle)
- Non-IID by fraud amount (sort by transaction amount, then split)
- Non-IID by category (sort by merchant category, then split)

Usage:
    python experiments/run_noniid_experiments.py
"""

import sys
import logging
import json
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.models.tab_transformer import create_model
from src.models.train_engine import train_one_round, evaluate_client
from src.fl.manual_loop import run_manual_simulation, _weighted_aggregate
from src.fl.secure_update import protect_update
from src.fl.dp_accountant import RDPAccountant
from src.data.split_clients import ClientSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def split_non_iid(
    df: pd.DataFrame,
    sort_column: str,
    output_dir: str,
    num_clients: int = 3,
    test_ratio: float = 0.15,
    val_ratio: float = 0.20,
    random_seed: int = 42
):
    """Split dataset into Non-IID clients by sorting on a column."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(random_seed)
    test_size = int(len(df) * test_ratio)
    test_idx = rng.choice(len(df), test_size, replace=False)
    test_mask = np.zeros(len(df), dtype=bool)
    test_mask[test_idx] = True

    df_test = df[test_mask].copy()
    df_remaining = df[~test_mask].copy()

    df_remaining = df_remaining.sort_values(sort_column).reset_index(drop=True)

    client_size = len(df_remaining) // num_clients
    splits = {'global_test': df_test}

    for i in range(num_clients):
        start = i * client_size
        end = start + client_size if i < num_clients - 1 else len(df_remaining)
        client_df = df_remaining.iloc[start:end].copy()

        val_size = int(len(client_df) * val_ratio)
        val_idx = rng.choice(len(client_df), val_size, replace=False)
        val_mask = np.zeros(len(client_df), dtype=bool)
        val_mask[val_idx] = True

        splits[f'client_{chr(97 + i)}_train'] = client_df[~val_mask].copy()
        splits[f'client_{chr(97 + i)}_val'] = client_df[val_mask].copy()

    saved = []
    for name, df_split in splits.items():
        path = Path(output_dir) / f"{name}.csv"
        df_split.to_csv(path, index=False)
        saved.append(str(path))
        logger.info("Saved %s: %d rows", name, len(df_split))

    return splits, saved


def run_experiment(
    experiment_name: str,
    source: str,
    data_dir: str,
    mapping_path: str,
    split_dir: str,
    vectorizer_path: str,
    output_dir: str,
    model_config: dict,
    train_config: dict,
    fl_config: dict,
    privacy_config: dict,
    strategy_type: str = "fedprox",
):
    """Run a single FL experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info("=" * 60)

    start_time = time.time()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    history = run_manual_simulation(
        client_ids=['client_a', 'client_b', 'client_c'],
        data_dir=split_dir,
        mapping_path=mapping_path,
        vectorizer_path=vectorizer_path,
        model_config=model_config,
        train_config=train_config,
        fl_config=fl_config,
        privacy_config=privacy_config,
        artifacts_dir=output_dir,
        strategy_type=strategy_type,
    )

    elapsed = time.time() - start_time

    best = max(history, key=lambda h: h.get('pr_auc', 0))
    result = {
        "experiment": experiment_name,
        "source": source,
        "strategy": strategy_type,
        "dp_enabled": privacy_config.get("enabled", False),
        "best_round": best['round'],
        "best_roc_auc": best.get('roc_auc', 0),
        "best_pr_auc": best.get('pr_auc', 0),
        "best_f1": best.get('f1', 0),
        "final_roc_auc": history[-1].get('roc_auc', 0),
        "final_pr_auc": history[-1].get('pr_auc', 0),
        "final_f1": history[-1].get('f1', 0),
        "elapsed_seconds": round(elapsed, 1),
        "num_rounds": len(history),
    }

    if privacy_config.get("enabled", False) and history[-1].get("dp_epsilon") is not None:
        result["final_dp_epsilon"] = history[-1]["dp_epsilon"]
        result["final_dp_delta"] = history[-1]["dp_delta"]

    result_path = Path(output_dir) / "experiment_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("Results: %s", json.dumps(result, indent=2))
    logger.info("Experiment completed in %.1f seconds", elapsed)

    return result


def main():
    """Run all Non-IID experiments."""
    experiments = []

    model_config = {
        'model_type': 'mlp',
        'hidden_dims': [64, 32],
        'dropout': 0.2,
    }
    train_config = {
        'epochs': 3,
        'batch_size': 256,
        'lr': 0.001,
    }
    fl_config = {
        'num_rounds': 20,
        'strategy': 'fedprox',
        'fedprox_mu': 0.01,
    }

    base_dir = "experiments/output"
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=== P2P Dataset Experiments ===")

    p2p_mapping = "configs/mapping_p2p.json"
    p2p_vectorizer = "artifacts/global_vectorizer_p2p.pkl"

    splits_dir_iid = "data/splits_p2p"

    result_iid = run_experiment(
        experiment_name="p2p_iid",
        source="p2p",
        data_dir="Sample_datasets",
        mapping_path=p2p_mapping,
        split_dir=splits_dir_iid,
        vectorizer_path=p2p_vectorizer,
        output_dir=f"{base_dir}/p2p_iid",
        model_config=model_config,
        train_config=train_config,
        fl_config=fl_config,
        privacy_config={'enabled': False, 'max_norm': 1.0, 'noise_multiplier': 0.0},
        strategy_type='fedprox',
    )
    experiments.append(result_iid)

    result_dp = run_experiment(
        experiment_name="p2p_iid_dp_sigma1",
        source="p2p",
        data_dir="Sample_datasets",
        mapping_path=p2p_mapping,
        split_dir=splits_dir_iid,
        vectorizer_path=p2p_vectorizer,
        output_dir=f"{base_dir}/p2p_iid_dp_sigma1",
        model_config=model_config,
        train_config=train_config,
        fl_config=fl_config,
        privacy_config={'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 1.0, 'delta': 1e-5},
        strategy_type='fedprox',
    )
    experiments.append(result_dp)

    splits_dir_amt = f"{base_dir}/p2p_noniid_amt_splits"
    df_p2p_train = pd.read_csv(f"{splits_dir_iid}/client_a_train.csv")
    df_p2p_val = pd.read_csv(f"{splits_dir_iid}/client_a_val.csv")
    df_p2p_test = pd.read_csv(f"{splits_dir_iid}/global_test.csv")
    df_p2p_all = pd.concat([df_p2p_train, df_p2p_val], ignore_index=True)

    split_non_iid(df_p2p_all, sort_column="amt", output_dir=splits_dir_amt)

    mapper_p2p = MetadataMapper(p2p_mapping)
    vectorizer_p2p = DynamicVectorizer(vector_size=128)
    train_dfs = []
    for cid in ['client_a', 'client_b', 'client_c']:
        train_dfs.append(pd.read_csv(f"{splits_dir_amt}/{cid}_train.csv"))
    combined = pd.concat(train_dfs, ignore_index=True)
    vectorizer_p2p_noniid = DynamicVectorizer(vector_size=128)
    vectorizer_p2p_noniid.fit_transform(combined, mapper_p2p)
    vectorizer_p2p_noniid.save(f"{base_dir}/global_vectorizer_p2p_noniid.pkl")

    result_noniid_amt = run_experiment(
        experiment_name="p2p_noniid_amount",
        source="p2p",
        data_dir="Sample_datasets",
        mapping_path=p2p_mapping,
        split_dir=splits_dir_amt,
        vectorizer_path=f"{base_dir}/global_vectorizer_p2p_noniid.pkl",
        output_dir=f"{base_dir}/p2p_noniid_amt",
        model_config=model_config,
        train_config=train_config,
        fl_config=fl_config,
        privacy_config={'enabled': False, 'max_norm': 1.0, 'noise_multiplier': 0.0},
        strategy_type='fedprox',
    )
    experiments.append(result_noniid_amt)

    logger.info("=== SUMMARY ===")
    for exp in experiments:
        logger.info("%-25s | ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f | DP eps: %s",
                   exp['experiment'], exp['best_roc_auc'], exp['best_pr_auc'],
                   exp['best_f1'], exp.get('final_dp_epsilon', 'N/A'))

    summary = {
        "experiments": experiments,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = Path(base_dir) / "all_experiments.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("All experiments complete. Summary: %s", summary_path)


if __name__ == '__main__':
    main()
