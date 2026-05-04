"""Streamlined P2P experiments — sampled data, 10 rounds, 3 settings.

Runs 3 experiments sequentially:
1. P2P IID baseline (no DP) — 10 rounds
2. P2P IID + DP (sigma=1.0) — 10 rounds
3. P2P Non-IID (sorted by amount) + DP (sigma=1.0) — 10 rounds

Each uses 50K samples (all fraud + random legit) to run in minutes.
"""

import sys
import logging
import json
import time
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


def prepare_data(mapping_path, raw_dir, output_dir, sample_size, seed, non_iid=False, sort_col=None):
    """Sample P2P data and split into 3 clients."""
    logger.info("=== Preparing data: iid=%s, seed=%d ===", non_iid, seed)

    df_train = pd.read_csv(Path(raw_dir) / "credit-card-2/fraudTrain.csv")
    df_test = pd.read_csv(Path(raw_dir) / "credit-card-2/fraudTest.csv")
    df = pd.concat([df_train, df_test], ignore_index=True)
    logger.info("Full P2P: %d rows", len(df))

    rng = np.random.RandomState(seed)
    fraud_idx = np.where(df["is_fraud"] == 1)[0]
    legit_idx = np.where(df["is_fraud"] == 0)[0]
    n_legit = sample_size - len(fraud_idx)
    sampled_legit = rng.choice(legit_idx, n_legit, replace=False)
    sampled = np.concatenate([fraud_idx, sampled_legit])
    df_sampled = df.iloc[sampled].copy().reset_index(drop=True)
    logger.info("Sampled: %d rows, fraud=%d (%.4f)", len(df_sampled), len(fraud_idx), len(fraud_idx)/len(df_sampled))

    mapper = MetadataMapper(mapping_path)
    schema = create_schema_from_mapping(mapping_path)

    splitter = ClientSplitter(num_clients=3, test_ratio=0.15, val_ratio=0.20, random_seed=seed)
    splits = splitter.split(df_sampled, label_col=schema.label_column, non_iid=non_iid, sort_by=sort_col)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        if name == "_global_test":
            data.to_csv(Path(output_dir) / "global_test.csv", index=False)
        else:
            for stype, d in data.items():
                d.to_csv(Path(output_dir) / f"{name}_{stype}.csv", index=False)

    combined = pd.concat([splits['client_a']['train'], splits['client_b']['train'], splits['client_c']['train']], ignore_index=True)
    return mapper, combined


def run_exp(name, mapping, split_dir, vec_path, out_dir, priv_cfg, rounds=10):
    """Run one FL experiment."""
    logger.info("=== EXPERIMENT: %s ===", name)
    t0 = time.time()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    hist = run_manual_simulation(
        client_ids=['client_a', 'client_b', 'client_c'],
        data_dir=split_dir,
        mapping_path=mapping,
        vectorizer_path=vec_path,
        model_config={'model_type': 'mlp', 'hidden_dims': [64, 32], 'dropout': 0.2},
        train_config={'epochs': 3, 'batch_size': 512, 'lr': 0.001},
        fl_config={'num_rounds': rounds, 'strategy': 'fedprox', 'fedprox_mu': 0.01},
        privacy_config=priv_cfg,
        artifacts_dir=out_dir,
        strategy_type='fedprox',
    )

    elapsed = time.time() - t0
    best = max(hist, key=lambda h: h.get('pr_auc', 0))
    result = {
        "experiment": name,
        "dp_enabled": priv_cfg.get("enabled", False),
        "best_round": best['round'],
        "best_roc_auc": round(best.get('roc_auc', 0), 4),
        "best_pr_auc": round(best.get('pr_auc', 0), 4),
        "best_f1": round(best.get('f1', 0), 4),
        "final_roc_auc": round(hist[-1].get('roc_auc', 0), 4),
        "final_pr_auc": round(hist[-1].get('pr_auc', 0), 4),
        "final_f1": round(hist[-1].get('f1', 0), 4),
        "elapsed_s": round(elapsed, 1),
    }
    if priv_cfg.get("enabled"):
        result["final_dp_epsilon"] = round(hist[-1].get("dp_epsilon", 0), 4)
        result["final_dp_delta"] = hist[-1].get("dp_delta", 0)

    with open(Path(out_dir) / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("DONE: %s | ROC=%.4f PR=%.4f F1=%.4f eps=%s | %.1fs",
               name, result['best_roc_auc'], result['best_pr_auc'], result['best_f1'],
               result.get('final_dp_epsilon', 'N/A'), elapsed)
    return result


def main():
    base = "experiments/output_p2p"
    Path(base).mkdir(parents=True, exist_ok=True)
    mapping = "configs/mapping_p2p.json"
    raw = "Sample_datasets"
    results = []

    # Experiment 1: IID, no DP
    sp1 = "data/splits_p2p_exp_iid"
    m1, c1 = prepare_data(mapping, raw, sp1, sample_size=50000, seed=42)
    vp1 = f"{base}/vec_iid.pkl"
    v1 = DynamicVectorizer(vector_size=128)
    v1.fit_transform(c1, m1)
    v1.save(vp1)
    r1 = run_exp("p2p_iid_no_dp", mapping, sp1, vp1, f"{base}/exp1_iid",
                 {'enabled': False, 'max_norm': 1.0, 'noise_multiplier': 0.0}, rounds=10)
    results.append(r1)

    # Experiment 2: IID, DP sigma=1.0
    r2 = run_exp("p2p_iid_dp_sigma1", mapping, sp1, vp1, f"{base}/exp2_iid_dp1",
                 {'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 1.0, 'delta': 1e-5}, rounds=10)
    results.append(r2)

    # Experiment 3: Non-IID by amount, DP sigma=1.0
    sp3 = "data/splits_p2p_exp_noniid"
    m3, c3 = prepare_data(mapping, raw, sp3, sample_size=50000, seed=43, non_iid=True, sort_col="amt")
    vp3 = f"{base}/vec_noniid.pkl"
    v3 = DynamicVectorizer(vector_size=128)
    v3.fit_transform(c3, m3)
    v3.save(vp3)
    r3 = run_exp("p2p_noniid_amt_dp1", mapping, sp3, vp3, f"{base}/exp3_noniid_dp1",
                 {'enabled': True, 'max_norm': 1.0, 'noise_multiplier': 1.0, 'delta': 1e-5}, rounds=10)
    results.append(r3)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 80)
    for r in results:
        logger.info("%-25s | ROC=%.4f | PR=%.4f | F1=%.4f | DP_eps=%s",
                   r['experiment'], r['best_roc_auc'], r['best_pr_auc'], r['best_f1'],
                   r.get('final_dp_epsilon', 'N/A'))

    with open(Path(base) / "summary.json", 'w') as f:
        json.dump({"experiments": results}, f, indent=2)
    logger.info("Summary: %s/summary.json", base)


if __name__ == '__main__':
    main()
