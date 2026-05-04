"""Run P2P FL experiments with sampled data for faster iteration.

Uses a representative sample of the full P2P dataset to run experiments
quickly while preserving the data distribution characteristics.

Usage:
    python experiments/run_p2p_experiments.py
"""

import sys
import logging
import json
import time
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.core.metadata_engine import MetadataMapper
from src.core.vectorizer import DynamicVectorizer
from src.fl.manual_loop import run_manual_simulation
from src.data.split_clients import ClientSplitter
from src.data.schema import create_schema_from_mapping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_sampled_splits(
    mapping_path: str,
    raw_dir: str = "Sample_datasets",
    output_dir: str = "data/splits_p2p_sampled",
    sample_size: int = 200000,
    test_ratio: float = 0.15,
    val_ratio: float = 0.20,
    random_seed: int = 42,
    non_iid: bool = False,
    non_iid_column: str = None,
):
    """Load P2P data, sample it, and split into clients."""
    logger.info("Loading P2P data and sampling to %d rows...", sample_size)

    train_path = Path(raw_dir) / "credit-card-2/fraudTrain.csv"
    test_path = Path(raw_dir) / "credit-card-2/fraudTest.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df = pd.concat([df_train, df_test], ignore_index=True)
    logger.info("Full dataset: %d rows", len(df))

    rng = np.random.RandomState(random_seed)
    fraud_mask = df["is_fraud"] == 1
    legit_mask = ~fraud_mask

    n_fraud = fraud_mask.sum()
    n_legit = min(sample_size - n_fraud, legit_mask.sum())

    fraud_idx = np.where(fraud_mask)[0]
    legit_idx = np.where(legit_mask)[0]
    sampled_legit = rng.choice(legit_idx, n_legit, replace=False)

    sampled_idx = np.concatenate([fraud_idx, sampled_legit])
    df_sampled = df.iloc[sampled_idx].copy().reset_index(drop=True)

    fraud_ratio = (df_sampled["is_fraud"] == 1).sum() / len(df_sampled)
    logger.info("Sampled dataset: %d rows, fraud ratio: %.4f", len(df_sampled), fraud_ratio)

    mapper = MetadataMapper(mapping_path)
    schema = create_schema_from_mapping(mapping_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splitter = ClientSplitter(
        num_clients=3,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    client_splits = splitter.split(
        df_sampled,
        label_col=schema.label_column,
        non_iid=non_iid,
        sort_by=non_iid_column if non_iid else None,
    )

    saved = []
    for name, split_data in client_splits.items():
        if name == "_global_test":
            path = Path(output_dir) / "global_test.csv"
            split_data.to_csv(path, index=False)
            saved.append(str(path))
            logger.info("Saved global_test: %d rows", len(split_data))
        else:
            for split_name, data in split_data.items():
                fname = f"{name}_{split_name}.csv"
                path = Path(output_dir) / fname
                data.to_csv(path, index=False)
                saved.append(str(path))
                logger.info("Saved %s: %d rows", fname, len(data))

    logger.info("Saved %d split files to %s", len(saved), output_dir)

    combined_train = pd.concat([
        client_splits['client_a']['train'],
        client_splits['client_b']['train'],
        client_splits['client_c']['train'],
    ], ignore_index=True)

    return mapper, combined_train, saved


def run_single_experiment(
    experiment_name: str,
    mapping_path: str,
    split_dir: str,
    vectorizer_path: str,
    output_dir: str,
    privacy_config: dict,
    strategy_type: str = "fedprox",
    num_rounds: int = 10,
):
    """Run a single FL experiment."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT: %s", experiment_name)
    logger.info("=" * 60)

    start_time = time.time()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_config = {
        'model_type': 'mlp',
        'hidden_dims': [64, 32],
        'dropout': 0.2,
    }
    train_config = {
        'epochs': 3,
        'batch_size': 512,
        'lr': 0.001,
    }
    fl_config = {
        'num_rounds': num_rounds,
        'strategy': strategy_type,
        'fedprox_mu': 0.01,
    }

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
        "strategy": strategy_type,
        "dp_enabled": privacy_config.get("enabled", False),
        "noise_multiplier": privacy_config.get("noise_multiplier", 0.0),
        "best_round": best['round'],
        "best_roc_auc": round(best.get('roc_auc', 0), 4),
        "best_pr_auc": round(best.get('pr_auc', 0), 4),
        "best_f1": round(best.get('f1', 0), 4),
        "final_roc_auc": round(history[-1].get('roc_auc', 0), 4),
        "final_pr_auc": round(history[-1].get('pr_auc', 0), 4),
        "final_f1": round(history[-1].get('f1', 0), 4),
        "elapsed_seconds": round(elapsed, 1),
        "num_rounds": len(history),
    }

    if privacy_config.get("enabled", False):
        result["final_dp_epsilon"] = round(history[-1].get("dp_epsilon", 0), 4)
        result["final_dp_delta"] = history[-1].get("dp_delta", 0)

    result_path = Path(output_dir) / "experiment_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("Result: %s", json.dumps(result, indent=2))
    logger.info("Completed in %.1f seconds", elapsed)

    return result


def main():
    """Run all P2P experiments."""
    mapping_path = "configs/mapping_p2p.json"
    base_output = "experiments/output_p2p"

    experiments = []

    logger.info("=== Experiment 1: IID Baseline (no DP) ===")
    splits_iid = "data/splits_p2p_sampled_iid"
    mapper, combined_train, _ = prepare_sampled_splits(
        mapping_path=mapping_path,
        output_dir=splits_iid,
        sample_size=200000,
        non_iid=False,
    )

    vec_iid = DynamicVectorizer(vector_size=128)
    vec_iid.fit_transform(combined_train, mapper)
    vec_path_iid = f"{base_output}/vectorizer_iid.pkl"
    Path(base_output).mkdir(parents=True, exist_ok=True)
    vec_iid.save(vec_path_iid)

    result_iid = run_single_experiment(
        experiment_name="p2p_iid_no_dp",
        mapping_path=mapping_path,
        split_dir=splits_iid,
        vectorizer_path=vec_path_iid,
        output_dir=f"{base_output}/iid_no_dp",
        privacy_config={'enabled': False, 'max_norm': 1.0, 'noise_multiplier': 0.0},
        num_rounds=10,
    )
    experiments.append(result_iid)

    logger.info("=== Experiment 2: IID with DP (sigma=1.0) ===")
    result_dp1 = run_single_experiment(
        experiment_name="p2p_iid_dp_sigma1",
        mapping_path=mapping_path,
        split_dir=splits_iid,
        vectorizer_path=vec_path_iid,
        output_dir=f"{base_output}/iid_dp_sigma1",
        privacy_config={'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 1.0, 'delta': 1e-5},
        num_rounds=10,
    )
    experiments.append(result_dp1)

    logger.info("=== Experiment 3: IID with DP (sigma=2.0) ===")
    result_dp2 = run_single_experiment(
        experiment_name="p2p_iid_dp_sigma2",
        mapping_path=mapping_path,
        split_dir=splits_iid,
        vectorizer_path=vec_path_iid,
        output_dir=f"{base_output}/iid_dp_sigma2",
        privacy_config={'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 2.0, 'delta': 1e-5},
        num_rounds=10,
    )
    experiments.append(result_dp2)

    logger.info("=== Experiment 4: Non-IID (sorted by amount) ===")
    splits_noniid = "data/splits_p2p_sampled_noniid"
    mapper2, combined_train2, _ = prepare_sampled_splits(
        mapping_path=mapping_path,
        output_dir=splits_noniid,
        sample_size=200000,
        non_iid=True,
        non_iid_column="amt",
        random_seed=43,
    )

    vec_noniid = DynamicVectorizer(vector_size=128)
    vec_noniid.fit_transform(combined_train2, mapper2)
    vec_path_noniid = f"{base_output}/vectorizer_noniid.pkl"
    vec_noniid.save(vec_path_noniid)

    result_noniid = run_single_experiment(
        experiment_name="p2p_noniid_amount",
        mapping_path=mapping_path,
        split_dir=splits_noniid,
        vectorizer_path=vec_path_noniid,
        output_dir=f"{base_output}/noniid_amt",
        privacy_config={'enabled': False, 'max_norm': 1.0, 'noise_multiplier': 0.0},
        num_rounds=10,
    )
    experiments.append(result_noniid)

    logger.info("=== Experiment 5: Non-IID with DP ===")
    result_noniid_dp = run_single_experiment(
        experiment_name="p2p_noniid_dp_sigma1",
        mapping_path=mapping_path,
        split_dir=splits_noniid,
        vectorizer_path=vec_path_noniid,
        output_dir=f"{base_output}/noniid_dp_sigma1",
        privacy_config={'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 1.0, 'delta': 1e-5},
        num_rounds=10,
    )
    experiments.append(result_noniid_dp)

    logger.info("=== ALL EXPERIMENTS COMPLETE ===")
    logger.info("")
    header = f"{'Experiment':<30} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'F1':>8} | {'DP eps':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for exp in experiments:
        logger.info(
            "%-30s | %8.4f | %8.4f | %8.4f | %s",
            exp['experiment'],
            exp['best_roc_auc'],
            exp['best_pr_auc'],
            exp['best_f1'],
            f"{exp.get('final_dp_epsilon', 'N/A'):.4f}" if exp.get('final_dp_epsilon') else "N/A",
        )

    summary = {
        "experiments": experiments,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "P2P (sampled)",
        "model": "TabularMLP",
        "strategy": "FedProx (mu=0.01)",
    }
    summary_path = Path(base_output) / "all_experiments.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary saved to %s", summary_path)


if __name__ == '__main__':
    main()
