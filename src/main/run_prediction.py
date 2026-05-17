"""Layer 2 Runner — Run global model inference on the test set.

Usage:
    python src/main/run_prediction.py
    python src/main/run_prediction.py --data data/splits/global_test.csv
"""

import sys
import logging
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.prediction.predictor import GlobalModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run global model inference.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/splits/global_test.csv",
        help="Path to CSV file to score",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/predictions/scored_global_test.csv",
        help="Output path for scored CSV",
    )
    args = parser.parse_args()

    import pandas as pd

    # --- Load data ---
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    # --- Load predictor ---
    predictor = GlobalModelPredictor.from_artifacts(args.artifacts_dir)
    logger.info("Predictor: %s", predictor)

    # --- Run inference ---
    scored_df = predictor.classify(df)

    # --- Save ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)
    logger.info("Scored CSV saved to %s", output_path)

    # --- Summary ---
    n_total = len(scored_df)
    n_flagged = int(scored_df["predicted_label"].sum())
    n_actual_fraud = int(scored_df.get("Class", pd.Series([0])).sum()) if "Class" in scored_df.columns else "N/A"
    avg_score = float(scored_df["fraud_risk_score"].mean())
    max_score = float(scored_df["fraud_risk_score"].max())
    min_score = float(scored_df["fraud_risk_score"].min())

    summary = {
        "total_transactions": n_total,
        "flagged_as_fraud": n_flagged,
        "actual_fraud_count": n_actual_fraud,
        "threshold_used": predictor.threshold,
        "avg_risk_score": round(avg_score, 4),
        "max_risk_score": round(max_score, 4),
        "min_risk_score": round(min_score, 4),
    }

    summary_path = output_path.parent / "prediction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    banner = f"""
    ============================================
    PREDICTION LAYER (LAYER 2) COMPLETE
    ============================================
    Transactions scored: {n_total}
    Flagged as fraud:    {n_flagged}
    Actual fraud:        {n_actual_fraud}
    Threshold used:      {predictor.threshold}
    Avg risk score:      {avg_score:.4f}
    Max risk score:      {max_score:.4f}
    Scored CSV:          {output_path}
    Summary:             {summary_path}
    ============================================
    """
    print(banner)


if __name__ == "__main__":
    main()
