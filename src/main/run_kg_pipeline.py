"""Layer 3 Runner — End-to-end Knowledge Graph pipeline.

Sequence:
  1. Load global test CSV
  2. Get risk scores from the Prediction Layer (Layer 2)
  3. Build the knowledge graph from schema config
  4. Enrich with risk scores and propagate
  5. Run community detection and centrality
  6. Save graph and reports to artifacts
  7. Query and print evidence bundles for top 5 high-risk transactions

Usage:
    python src/main/run_kg_pipeline.py
    python src/main/run_kg_pipeline.py --data data/splits/global_test.csv
"""

import sys
import json
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kg.kg_schema import KGSchema
from src.kg.kg_builder import KnowledgeGraphBuilder
from src.kg.kg_enricher import KGEnricher
from src.kg.kg_query import KGQueryEngine
from src.kg.kg_analytics import KGAnalytics
from src.prediction.predictor import GlobalModelPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Run the Knowledge Graph pipeline.")
    parser.add_argument("--data", default="data/splits/global_test.csv", help="CSV to process")
    parser.add_argument("--kg-config", default="configs/kg_config.yaml", help="KG config path")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top-risk transactions to display")
    args = parser.parse_args()

    start_time = time.time()

    # ---- Step 1: Load KG schema ----
    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH PIPELINE (LAYER 3)")
    logger.info("=" * 60)

    schema = KGSchema.from_config(args.kg_config)
    logger.info("Schema: %s", schema)

    # ---- Step 2: Load transaction data ----
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    df = pd.read_csv(data_path)
    logger.info("Loaded %d transactions from %s", len(df), data_path)

    # Validate schema against data
    is_valid, issues = schema.validate(df)
    if not is_valid:
        logger.warning("Schema validation issues: %s", issues)

    # ---- Step 3: Get risk scores from Prediction Layer ----
    logger.info("--- Getting risk scores from Prediction Layer ---")
    try:
        predictor = GlobalModelPredictor.from_artifacts(args.artifacts_dir)
        risk_scores = predictor.predict_scores_only(df)
        logger.info("Risk scores: min=%.4f, max=%.4f, mean=%.4f",
                     risk_scores.min(), risk_scores.max(), risk_scores.mean())
    except Exception as e:
        logger.warning("Could not load global model: %s. Using random scores for demo.", e)
        import numpy as np
        np.random.seed(42)
        risk_scores = np.random.beta(0.5, 5, size=len(df))

    # ---- Step 4: Build Knowledge Graph ----
    logger.info("--- Building Knowledge Graph ---")
    builder = KnowledgeGraphBuilder(schema)
    graph = builder.build(df)
    build_stats = builder.get_build_stats()
    logger.info("Build stats: %s", build_stats)

    # ---- Step 5: Enrich with risk scores ----
    logger.info("--- Enriching graph with risk scores ---")
    enricher = KGEnricher(schema)
    graph = enricher.attach_predictions(graph, risk_scores, df.index.values)
    graph = enricher.propagate_risk(graph)
    graph = enricher.label_risk_tiers(graph)
    enrichment_stats = enricher.compute_enrichment_stats(graph)
    logger.info("Enrichment stats: %s", enrichment_stats)

    # ---- Step 6: Run analytics ----
    logger.info("--- Running graph analytics ---")
    analytics = KGAnalytics(graph, schema)
    community_results = analytics.detect_communities()
    centrality_results = analytics.compute_centrality()
    risk_clusters = analytics.find_risk_clusters()
    graph_summary = analytics.get_graph_summary()
    logger.info("Graph summary: %s", graph_summary)
    logger.info("Communities: %d found", community_results["num_communities"])
    logger.info("Risk clusters: %d found", len(risk_clusters))

    # ---- Step 7: Save artifacts ----
    logger.info("--- Saving artifacts ---")
    output_dir = Path(schema.output_config.get("artifacts_dir", "artifacts/knowledge_graph"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save graph
    graph_path = output_dir / "fraud_kg.graphml"
    builder.graph = graph  # Update builder's graph with enriched version
    builder.save(str(graph_path))

    # Save reports
    build_report = {
        "build_stats": build_stats,
        "enrichment_stats": enrichment_stats,
        "graph_summary": graph_summary,
        "community_results": {k: v for k, v in community_results.items() if k != "sizes"},
        "centrality_top_5": centrality_results.get("top_nodes", [])[:5],
        "risk_clusters": risk_clusters[:10],
    }

    report_path = output_dir / "kg_build_report.json"
    with open(report_path, "w") as f:
        json.dump(build_report, f, indent=2, default=str)
    logger.info("Build report saved: %s", report_path)

    # ---- Step 8: Query top-K evidence bundles ----
    logger.info("--- Querying top %d high-risk transactions ---", args.top_k)
    query_engine = KGQueryEngine(graph, schema)
    top_risk = query_engine.get_top_risk_transactions(top_k=args.top_k)

    evidence_bundles = []
    for txn in top_risk:
        bundle = query_engine.generate_evidence_bundle(txn["transaction_id"])
        evidence_bundles.append(bundle)

        print(f"\n{'=' * 60}")
        print(f"Transaction: {bundle['transaction_id']}")
        print(f"Risk Score:  {bundle.get('risk_score', 0):.4f}")
        print(f"Risk Tier:   {bundle.get('risk_tier', 'unknown')}")
        print(f"Prediction:  {bundle.get('model_prediction', 'unknown')}")
        print(f"Ground Truth: {bundle.get('ground_truth', 'N/A')}")
        print(f"Neighborhood Risk: {bundle.get('neighborhood_risk', 0):.4f}")
        print(f"High-Risk Neighbors: {bundle.get('connected_high_risk_count', 0)}")
        if bundle.get("community"):
            comm = bundle["community"]
            print(f"Community:   #{comm.get('community_id', -1)} "
                  f"(size={comm.get('size', 0)}, "
                  f"high-risk ratio={comm.get('high_risk_ratio', 0):.0%})")
        print(f"\nEvidence: {bundle.get('evidence_summary', '')}")
        print("=" * 60)

    # Save evidence bundles
    evidence_path = output_dir / "top_risk_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence_bundles, f, indent=2, default=str)
    logger.info("Evidence bundles saved: %s", evidence_path)

    elapsed = time.time() - start_time

    banner = f"""
    ============================================
    KNOWLEDGE GRAPH LAYER (LAYER 3) COMPLETE
    ============================================
    Transactions processed: {len(df)}
    Graph nodes:            {graph_summary['total_nodes']}
    Graph edges:            {graph_summary['total_edges']}
    Communities detected:   {community_results['num_communities']}
    Risk clusters found:    {len(risk_clusters)}
    
    Artifacts saved to:     {output_dir}
      - fraud_kg.graphml
      - kg_build_report.json
      - top_risk_evidence.json

    Ready for: Layer 4 (Explainability) -> Layer 5 (Agentic Engine)
    Completed in {elapsed:.1f} seconds.
    ============================================
    """
    print(banner)


if __name__ == "__main__":
    main()
