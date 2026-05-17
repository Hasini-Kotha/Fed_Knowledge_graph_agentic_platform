"""Run full FL simulation and evaluation end-to-end."""

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


def run_full_fl_pipeline():
    """Run FL simulation + global evaluation."""
    
    # Phase 1: FL Simulation
    logger.info("=" * 60)
    logger.info("PHASE 1: Federated Learning Simulation")
    logger.info("=" * 60)
    
    history = run_manual_simulation(
        client_ids=['client_a', 'client_b', 'client_c'],
        data_dir='data/splits',
        mapping_path='configs/mapping.json',
        vectorizer_path='artifacts/global_vectorizer_kaggle.pkl',
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
            'enabled': False,
            'max_norm': 1.0,
            'noise_multiplier': 0.0,
        },
        artifacts_dir='artifacts/global_model',
        strategy_type='fedprox',
    )
    
    logger.info(f"FL simulation complete: {len(history)} rounds")
    
    # Phase 2: Evaluate Global Model
    logger.info("=" * 60)
    logger.info("PHASE 2: Global Model Evaluation")
    logger.info("=" * 60)
    
    checkpoint = load_best_checkpoint('artifacts/global_model', metric='pr_auc')
    
    if checkpoint is None:
        logger.error("No checkpoints found. Cannot evaluate.")
        return
    
    global_metrics = evaluate_global_model(
        checkpoint=checkpoint,
        global_test_csv='data/splits/global_test.csv',
        mapping_path='configs/mapping.json',
        vectorizer_path='artifacts/global_vectorizer_kaggle.pkl',
        model_config={
            'model_type': 'mlp',
            'hidden_dims': [64, 32],
            'dropout': 0.2,
        }
    )
    
    # Phase 3: Generate Report
    logger.info("=" * 60)
    logger.info("PHASE 3: Generating Evaluation Report")
    logger.info("=" * 60)
    
    generate_evaluation_report(
        round_number=checkpoint['round'],
        round_metrics=checkpoint['metrics'],
        global_metrics=global_metrics,
        output_path='artifacts/reports'
    )
    
    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_full_fl_pipeline()
