"""Run P2P FL simulation and evaluation end-to-end."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fl.manual_loop import run_manual_simulation
from src.evaluation.evaluate_global import load_best_checkpoint, evaluate_global_model, generate_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_p2p_fl_pipeline(dp_enabled=False, noise_multiplier=0.0):
    """Run P2P FL simulation + global evaluation."""

    dp_label = "dp_sigma" + str(noise_multiplier) if dp_enabled else "no_dp"

    logger.info("=" * 60)
    logger.info("PHASE 1: P2P Federated Learning Simulation (%s)", dp_label)
    logger.info("=" * 60)

    history = run_manual_simulation(
        client_ids=['client_a', 'client_b', 'client_c'],
        data_dir='data/splits_p2p',
        mapping_path='configs/mapping_p2p.json',
        vectorizer_path='artifacts/global_vectorizer_p2p.pkl',
        model_config={
            'model_type': 'mlp',
            'hidden_dims': [64, 32],
            'dropout': 0.2,
        },
        train_config={
            'epochs': 3,
            'batch_size': 256,
            'lr': 0.001,
        },
        fl_config={
            'num_rounds': 20,
            'strategy': 'fedprox',
            'fedprox_mu': 0.01,
        },
        privacy_config={
            'enabled': dp_enabled,
            'max_norm': 1.0,
            'noise_multiplier': noise_multiplier,
            'delta': 1e-5,
        },
        artifacts_dir=f'artifacts/global_model_p2p_{dp_label}',
        strategy_type='fedprox',
    )

    logger.info("FL simulation complete: %d rounds", len(history))

    best_round = max(history, key=lambda h: h.get('pr_auc', 0))
    logger.info("Best round: %d (PR-AUC=%.4f, ROC-AUC=%.4f)",
               best_round['round'], best_round['pr_auc'], best_round['roc_auc'])

    if dp_enabled:
        dp_eps = best_round.get('dp_epsilon')
        logger.info("DP budget at best round: eps=%.4f", dp_eps if dp_eps else 'N/A')

    return history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run P2P FL pipeline")
    parser.add_argument("--dp", action="store_true", help="Enable differential privacy")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")

    args = parser.parse_args()

    run_p2p_fl_pipeline(dp_enabled=args.dp, noise_multiplier=args.sigma)
