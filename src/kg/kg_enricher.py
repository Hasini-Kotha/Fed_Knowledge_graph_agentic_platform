"""KG Enricher — Attaches risk scores from the global model to the graph and propagates risk.

This is the critical integration point between the FL pipeline (Layer 1/2)
and the Knowledge Graph (Layer 3).  It takes the built graph, adds model
predictions as node attributes, and then propagates risk through neighborhoods.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from src.kg.kg_schema import KGSchema

logger = logging.getLogger(__name__)


class KGEnricher:
    """Enriches a knowledge graph with model predictions and risk propagation.

    Args:
        schema: KGSchema instance with risk thresholds and propagation settings.
    """

    def __init__(self, schema: KGSchema):
        self.schema = schema
        self.risk_config = schema.risk_config
        self.high_threshold = self.risk_config.get("high_risk_threshold", 0.7)
        self.medium_threshold = self.risk_config.get("medium_risk_threshold", 0.4)
        self.propagation_hops = self.risk_config.get("propagation_hops", 2)

    def attach_predictions(
        self, graph: nx.Graph, risk_scores: np.ndarray, df_index: Optional[np.ndarray] = None
    ) -> nx.Graph:
        """Attach model risk scores to transaction nodes.

        Args:
            graph: The knowledge graph with transaction nodes.
            risk_scores: Array of risk scores from predict_proba, one per transaction.
            df_index: DataFrame index values to match nodes. If None, uses 0..N-1.

        Returns:
            The same graph with 'risk_score' attribute on transaction nodes.
        """
        primary = self.schema.get_primary_entity()
        if primary is None:
            logger.error("No primary entity type defined.")
            return graph

        prefix = primary.id_prefix or f"{primary.name}_"
        risk_attr = primary.risk_score_attr

        if df_index is None:
            df_index = np.arange(len(risk_scores))

        attached = 0
        for i, score in enumerate(risk_scores):
            node_id = f"{prefix}{df_index[i]}"
            if graph.has_node(node_id):
                graph.nodes[node_id][risk_attr] = float(score)
                attached += 1

        logger.info("Attached risk scores to %d / %d transaction nodes", attached, len(risk_scores))
        return graph

    def propagate_risk(self, graph: nx.Graph, hops: Optional[int] = None) -> nx.Graph:
        """Propagate risk through graph neighborhoods.

        For each transaction node, computes the average risk score of its
        neighbors within N hops and stores it as 'neighborhood_risk'.

        Args:
            graph: Graph with risk_score attributes on transaction nodes.
            hops: Number of hops to consider. Defaults to config value.

        Returns:
            Graph with 'neighborhood_risk' attribute added.
        """
        if hops is None:
            hops = self.propagation_hops

        primary = self.schema.get_primary_entity()
        if primary is None:
            return graph

        prefix = primary.id_prefix or f"{primary.name}_"
        risk_attr = primary.risk_score_attr

        # Collect transaction nodes
        txn_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("entity_type") == primary.name
        ]

        propagated = 0
        for node_id in txn_nodes:
            try:
                # Get neighbors within N hops
                neighbors_at_distance = nx.single_source_shortest_path_length(
                    graph, node_id, cutoff=hops
                )
            except Exception:
                continue

            # Collect risk scores of neighboring transaction nodes
            neighbor_risks = []
            for neighbor_id, dist in neighbors_at_distance.items():
                if dist == 0:
                    continue  # skip self
                if not neighbor_id.startswith(prefix):
                    continue  # skip non-transaction nodes
                n_risk = graph.nodes[neighbor_id].get(risk_attr, None)
                if n_risk is not None:
                    neighbor_risks.append(n_risk)

            if neighbor_risks:
                graph.nodes[node_id]["neighborhood_risk"] = float(np.mean(neighbor_risks))
                graph.nodes[node_id]["neighbor_count"] = len(neighbor_risks)
            else:
                graph.nodes[node_id]["neighborhood_risk"] = 0.0
                graph.nodes[node_id]["neighbor_count"] = 0

            propagated += 1

        logger.info("Risk propagated for %d transaction nodes (%d hops)", propagated, hops)
        return graph

    def label_risk_tiers(self, graph: nx.Graph) -> nx.Graph:
        """Label each transaction node as high / medium / low risk.

        Uses thresholds from the schema's risk config.

        Args:
            graph: Graph with risk_score attributes.

        Returns:
            Graph with 'risk_tier' attribute added.
        """
        primary = self.schema.get_primary_entity()
        if primary is None:
            return graph

        risk_attr = primary.risk_score_attr
        counts = {"high": 0, "medium": 0, "low": 0}

        for node_id, data in graph.nodes(data=True):
            if data.get("entity_type") != primary.name:
                continue

            score = data.get(risk_attr, 0.0)

            if score >= self.high_threshold:
                tier = "high"
            elif score >= self.medium_threshold:
                tier = "medium"
            else:
                tier = "low"

            graph.nodes[node_id]["risk_tier"] = tier
            counts[tier] += 1

        logger.info(
            "Risk tiers assigned: high=%d, medium=%d, low=%d",
            counts["high"], counts["medium"], counts["low"],
        )
        return graph

    def compute_enrichment_stats(self, graph: nx.Graph) -> Dict[str, Any]:
        """Compute summary statistics of the enriched graph.

        Returns:
            Dict with risk distribution, avg scores, tier counts, etc.
        """
        primary = self.schema.get_primary_entity()
        if primary is None:
            return {}

        risk_attr = primary.risk_score_attr
        scores = []
        tiers = {"high": 0, "medium": 0, "low": 0}
        neighborhood_risks = []

        for node_id, data in graph.nodes(data=True):
            if data.get("entity_type") != primary.name:
                continue

            score = data.get(risk_attr, None)
            if score is not None:
                scores.append(score)

            tier = data.get("risk_tier", "low")
            tiers[tier] = tiers.get(tier, 0) + 1

            n_risk = data.get("neighborhood_risk", None)
            if n_risk is not None:
                neighborhood_risks.append(n_risk)

        stats = {
            "total_transactions": len(scores),
            "avg_risk_score": float(np.mean(scores)) if scores else 0.0,
            "max_risk_score": float(np.max(scores)) if scores else 0.0,
            "min_risk_score": float(np.min(scores)) if scores else 0.0,
            "std_risk_score": float(np.std(scores)) if scores else 0.0,
            "risk_tiers": tiers,
            "avg_neighborhood_risk": float(np.mean(neighborhood_risks)) if neighborhood_risks else 0.0,
        }

        return stats
