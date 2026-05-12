"""KG Query Demo — Query the evidence bundle for a specific transaction.

Usage:
    python src/main/run_kg_query_demo.py
    python src/main/run_kg_query_demo.py --index 42
    python src/main/run_kg_query_demo.py --id txn_42301
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.kg.kg_schema import KGSchema
from src.kg.kg_builder import KnowledgeGraphBuilder
from src.kg.kg_query import KGQueryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query a transaction from the Knowledge Graph.")
    parser.add_argument("--index", type=int, default=0, help="Row index of the transaction")
    parser.add_argument("--id", type=str, default=None, help="Direct node ID (e.g. txn_42301)")
    parser.add_argument("--kg-config", default="configs/kg_config.yaml", help="KG config path")
    parser.add_argument("--graph-path", default="artifacts/knowledge_graph/fraud_kg.graphml", help="Path to saved graph")
    args = parser.parse_args()

    # Load schema
    schema = KGSchema.from_config(args.kg_config)

    # Load graph
    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        logger.error(
            "Graph file not found: %s. Run 'python src/main/run_kg_pipeline.py' first.",
            graph_path,
        )
        sys.exit(1)

    builder = KnowledgeGraphBuilder.load(str(graph_path), schema)
    graph = builder.graph
    logger.info("Graph loaded: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    # Determine transaction ID
    if args.id:
        txn_id = args.id
    else:
        primary = schema.get_primary_entity()
        prefix = primary.id_prefix or f"{primary.name}_" if primary else "txn_"
        txn_id = f"{prefix}{args.index}"

    logger.info("Querying transaction: %s", txn_id)

    # Query
    query_engine = KGQueryEngine(graph, schema)
    bundle = query_engine.generate_evidence_bundle(txn_id)

    if "error" in bundle:
        logger.error("Query failed: %s", bundle["error"])
        sys.exit(1)

    # Print formatted output
    print(f"\n{'=' * 60}")
    print(f"EVIDENCE BUNDLE FOR: {bundle['transaction_id']}")
    print(f"{'=' * 60}")
    print(json.dumps(bundle, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"EVIDENCE SUMMARY:")
    print(f"{'=' * 60}")
    print(bundle.get("evidence_summary", "No summary available"))
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
