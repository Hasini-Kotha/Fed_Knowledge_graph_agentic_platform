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

    # ── Incremental rolling-window arguments ──────────────────────────────
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help=(
            "Incremental mode: load existing graph from --graph-state, "
            "add new transactions from --data, evict old nodes using dual constraint "
            "(old timestamp AND all SIMILAR_PATTERN neighbors low-risk)."
        ),
    )
    parser.add_argument(
        "--graph-state",
        default="artifacts/knowledge_graph/fraud_kg.graphml",
        help="Path to saved GraphML file. Used as input in --incremental mode and "
             "overwritten with the updated graph after every run.",
    )
    parser.add_argument(
        "--cutoff-timestamp",
        type=float,
        default=None,
        help=(
            "Evict transaction nodes with timestamp < this value, PROVIDED their "
            "SIMILAR_PATTERN neighbors are all low-risk. "
            "Example: pass the min timestamp of the new batch to evict older data."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        help=(
            "Duration-based eviction window in days. Nodes older than this are "
            "evicted (subject to low-risk-neighbor constraint). "
            "Example: --max-age-days 2 evicts transactions ingested > 2 days ago. "
            "Overrides --cutoff-timestamp when both are provided."
        ),
    )
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

    # ---- Step 3: Get risk scores and embeddings from Prediction Layer ----
    logger.info("--- Getting risk scores and embeddings from Prediction Layer ---")
    embedding_matrix = None
    try:
        predictor = GlobalModelPredictor.from_artifacts(args.artifacts_dir)
        risk_scores = predictor.predict_scores_only(df)
        logger.info("Risk scores: min=%.4f, max=%.4f, mean=%.4f",
                     risk_scores.min(), risk_scores.max(), risk_scores.mean())

        # Extract MLP embeddings for semantically meaningful similarity edges
        try:
            import torch
            import pathlib as _pl
            from src.models.Fed_model import create_model

            _artifacts = _pl.Path(args.artifacts_dir)
            _prep_path = _artifacts / "preprocessors" / "client_a_preprocessor.pkl"
            _ckpt_dir  = _artifacts / "global_model"

            # Find the latest round checkpoint
            _ckpts = sorted(_ckpt_dir.glob("round_*_checkpoint.pt"))
            if _ckpts and _prep_path.exists():
                _prep = ClientPreprocessor.load(str(_prep_path))
                _input_dim = _prep.get_feature_dim()
                _ckpt = torch.load(str(_ckpts[-1]), map_location="cpu", weights_only=False)
                _model_cfg = {"hidden_dim": 64, "embedding_dim": 32, "dropout": 0.20}
                _model = create_model(input_dim=64, config=_model_cfg, model_type="lite_fraud_net")
                _params = _ckpt.get("weights", None)
                if _params:
                    _model.set_parameters(_params)
                _model.eval()

                X_proc, _ = _prep.transform(df)
                X_tensor = torch.tensor(X_proc, dtype=torch.float32)

                with torch.no_grad():
                    # Batch-process to avoid OOM on large test sets
                    emb_parts = []
                    _bs = 512
                    for _s in range(0, len(X_tensor), _bs):
                        emb_parts.append(_model.get_embeddings(X_tensor[_s:_s+_bs]))
                    embedding_matrix = torch.cat(emb_parts, dim=0).numpy()

                logger.info(
                    "MLP embeddings extracted: shape=%s (will be used for SIMILAR_PATTERN edges)",
                    embedding_matrix.shape,
                )
            else:
                logger.warning("No checkpoint or preprocessor found — using raw features for similarity")
        except Exception as emb_err:
            logger.warning("Embedding extraction failed (%s) — falling back to raw features", emb_err)
            embedding_matrix = None

    except Exception as e:
        logger.warning("Could not load global model: %s. Using random scores for demo.", e)
        import numpy as np
        np.random.seed(42)
        risk_scores = np.random.beta(0.5, 5, size=len(df))
        embedding_matrix = None

    # ---- Step 4: Build or update the Knowledge Graph ----
    builder = KnowledgeGraphBuilder(schema)

    if args.incremental:
        # ── Incremental mode: load existing graph and add new transactions ──
        graph_state_path = Path(args.graph_state)
        if graph_state_path.exists():
            logger.info("--- Incremental mode: loading existing graph from %s ---", graph_state_path)
            builder = KnowledgeGraphBuilder.load(str(graph_state_path), schema)
            logger.info(
                "Loaded graph: %d nodes, %d edges",
                builder.graph.number_of_nodes(),
                builder.graph.number_of_edges(),
            )
        else:
            logger.warning(
                "Graph state file not found at %s — building fresh graph.",
                graph_state_path,
            )

        # Determine cutoff timestamp
        cutoff_ts = args.cutoff_timestamp
        if cutoff_ts is None and "timestamp" in df.columns:
            # Default: use min timestamp of the NEW batch as the cutoff.
            # This evicts transactions older than the earliest transaction
            # in the current batch, subject to the low-risk-neighbor constraint.
            cutoff_ts = float(df["timestamp"].min()) if "timestamp" in df.columns else None
            # Fallback: use the Time column directly
            if cutoff_ts is None and "Time" in df.columns:
                cutoff_ts = float(df["Time"].min())
        elif cutoff_ts is None and "Time" in df.columns:
            cutoff_ts = float(df["Time"].min())

        logger.info(
            "--- Incremental add: %d new transactions, cutoff_timestamp=%.1f, max_age_days=%s ---",
            len(df), cutoff_ts if cutoff_ts is not None else -1, args.max_age_days,
        )
        max_age_seconds = args.max_age_days * 86400 if args.max_age_days else None
        update_stats = builder.add_transactions(
            new_df=df,
            new_embedding_matrix=embedding_matrix,
            cutoff_timestamp=cutoff_ts,
            max_age_seconds=max_age_seconds,
        )
        logger.info("Update stats: %s", update_stats)
        build_stats = {
            "mode": "incremental",
            "update_stats": update_stats,
            "total_nodes": builder.graph.number_of_nodes(),
            "total_edges": builder.graph.number_of_edges(),
            "embeddings_used": embedding_matrix is not None,
        }
        graph = builder.graph

    else:
        # ── Full rebuild mode (default) ─────────────────────────────────────
        logger.info("--- Full rebuild mode: building Knowledge Graph from scratch ---")
        graph = builder.build(df, embedding_matrix=embedding_matrix)
        build_stats = builder.get_build_stats()

    logger.info("Build stats: %s", build_stats)

    # ---- Step 5: Enrich with risk scores ----
    logger.info("--- Enriching graph with risk scores ---")
    enricher = KGEnricher(schema)
    graph = enricher.attach_predictions(graph, risk_scores, df.index.values)
    graph = enricher.label_risk_tiers(graph)      # tier first so time window can use it
    graph = enricher.enrich_time_windows(graph)   # per-window fraud rates vs baseline
    graph = enricher.propagate_risk(graph)         # SIMILAR_PATTERN BFS only (no hub dilution)
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
    logger.info(
        "Communities: %d found, risk_clusters (similarity-based): %d",
        community_results["num_communities"],
        community_results.get("risk_clusters", 0),
    )
    logger.info("Risk clusters (find_risk_clusters): %d found", len(risk_clusters))

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
