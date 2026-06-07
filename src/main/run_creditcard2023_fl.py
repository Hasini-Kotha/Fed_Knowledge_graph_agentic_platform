"""Run FL simulation for credit-card-3 (2023 dataset) using DynamicVectorizer."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fl.manual_loop import run_manual_simulation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fl():
    """Run FL simulation on credit-card-3 using pre-fitted DynamicVectorizer."""

    # The dynamic vectorizer was already fitted by run_data_pipeline.py
    vectorizer_path = 'artifacts/global_vectorizer_creditcard_2023.pkl'

    logger.info('=' * 60)
    logger.info('FL SIMULATION: credit-card-3 (neobank_2023, DynamicVectorizer)')
    logger.info('=' * 60)

    history = run_manual_simulation(
        client_ids=['client_a', 'client_b', 'client_c'],
        data_dir='data/splits_creditcard_2023',
        mapping_path='configs/mapping_creditcard_2023.json',
        vectorizer_path=vectorizer_path,
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
            'patience': 7,
        },
        privacy_config={
            'enabled': False,
            'max_norm': 1.0,
            'noise_multiplier': 0.0,
        },
        artifacts_dir='artifacts/global_model_creditcard_2023',
        strategy_type='fedprox',
    )

    logger.info('FL simulation complete: %d rounds', len(history))

    best = max(history, key=lambda h: h.get('pr_auc', 0))
    final = history[-1]
    print()
    print('=' * 60)
    print('SUMMARY — credit-card-3 FL (DynamicVectorizer)')
    print('=' * 60)
    print('Rounds completed: %d' % len(history))
    print()
    print('Final round (%d):' % final['round'])
    print('  ROC-AUC:  %.4f' % final.get('roc_auc', 0))
    print('  PR-AUC:   %.4f' % final.get('pr_auc', 0))
    print('  F1:       %.4f' % final.get('f1', 0))
    print()
    print('Best round (%d):' % best['round'])
    print('  ROC-AUC:  %.4f' % best.get('roc_auc', 0))
    print('  PR-AUC:   %.4f' % best.get('pr_auc', 0))
    print('  F1:       %.4f' % best.get('f1', 0))
    print()
    print('Checkpoints: artifacts/global_model_creditcard_2023/')
    print('=' * 60)

    return history


if __name__ == '__main__':
    run_fl()
