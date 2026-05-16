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

    def propagate_risk(
        self,
        graph: nx.Graph,
        hops: Optional[int] = None,
        edge_type: str = "SIMILAR_PATTERN",
    ) -> nx.Graph:
        """Propagate risk through SIMILAR_PATTERN neighborhoods only.

        Uses edge-filtered BFS so risk never crosses hub nodes
        (amount_bucket, time_window).  This ensures neighborhood_risk
        reflects behaviorally similar peers, not calendar/bucket neighbors.

        Args:
            graph: Graph with risk_score attributes on transaction nodes.
            hops: Number of hops. Defaults to config value.
            edge_type: Only traverse edges of this relationship type.

        Returns:
            Graph with 'neighborhood_risk' and 'neighbor_count' attributes.
        """
        if hops is None:
            hops = self.propagation_hops

        primary = self.schema.get_primary_entity()
        if primary is None:
            return graph

        prefix = primary.id_prefix or f"{primary.name}_"
        risk_attr = primary.risk_score_attr

        # Collect all transaction node IDs
        txn_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("entity_type") == primary.name
        ]

        propagated = 0
        for node_id in txn_nodes:
            # Edge-filtered BFS: only SIMILAR_PATTERN edges, only transaction nodes
            visited = {node_id}
            frontier = {node_id}

            for _ in range(hops):
                next_frontier = set()
                for current in frontier:
                    for nbr in graph.neighbors(current):
                        if nbr in visited:
                            continue
                        edge_data = graph.edges[current, nbr]
                        if edge_data.get("relationship") != edge_type:
                            continue  # skip structural edges through hub nodes
                        if not nbr.startswith(prefix):
                            continue  # skip non-transaction nodes
                        next_frontier.add(nbr)
                visited.update(next_frontier)
                frontier = next_frontier

            # visited includes start node; exclude it
            neighbor_ids = visited - {node_id}

            neighbor_risks = [
                graph.nodes[nbr].get(risk_attr, 0.0)
                for nbr in neighbor_ids
                if graph.has_node(nbr) and graph.nodes[nbr].get(risk_attr) is not None
            ]

            if neighbor_risks:
                graph.nodes[node_id]["neighborhood_risk"] = float(np.mean(neighbor_risks))
                graph.nodes[node_id]["neighbor_count"] = len(neighbor_risks)
            else:
                graph.nodes[node_id]["neighborhood_risk"] = 0.0
                graph.nodes[node_id]["neighbor_count"] = 0

            propagated += 1

        logger.info(
            "Risk propagated for %d transaction nodes (%d hops, edge_type=%s)",
            propagated, hops, edge_type,
        )
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

    def enrich_time_windows(self, graph: nx.Graph) -> nx.Graph:
        """Compute per-time-window fraud statistics and attach to window nodes.

        For each time_window node, calculates:
        - fraud_count: number of connected high-risk transactions
        - total_count: total connected transactions
        - fraud_rate: fraud_count / total_count
        - baseline_rate: overall dataset fraud rate
        - elevated: True if fraud_rate > 2x baseline

        This gives the KG evidence layer meaningful temporal context:
        e.g. "Time window 14 has 5x elevated fraud rate vs baseline."

        Args:
            graph: Enriched graph with risk_tier attributes set.

        Returns:
            Graph with fraud statistics on time_window nodes.
        """
        primary = self.schema.get_primary_entity()
        if primary is None:
            return graph

        # Find time_window entity type
        time_et = None
        for et in self.schema.get_derived_entities():
            if "time" in et.name.lower():
                time_et = et
                break

        if time_et is None:
            logger.info("No time_window entity type found, skipping time window enrichment")
            return graph

        # Compute overall baseline fraud rate
        all_txn_nodes = [
            nid for nid, d in graph.nodes(data=True)
            if d.get("entity_type") == primary.name
        ]
        if not all_txn_nodes:
            return graph

        total_txn = len(all_txn_nodes)
        total_fraud = sum(
            1 for nid in all_txn_nodes
            if graph.nodes[nid].get("risk_tier") == "high"
        )
        baseline_rate = total_fraud / max(total_txn, 1)

        # Enrich each time_window node
        window_nodes = [
            nid for nid, d in graph.nodes(data=True)
            if d.get("entity_type") == time_et.name
        ]

        enriched_windows = 0
        for win_id in window_nodes:
            connected_txns = [
                nbr for nbr in graph.neighbors(win_id)
                if graph.nodes[nbr].get("entity_type") == primary.name
            ]
            if not connected_txns:
                continue

            fraud_count = sum(
                1 for nbr in connected_txns
                if graph.nodes[nbr].get("risk_tier") == "high"
            )
            total_count = len(connected_txns)
            fraud_rate = fraud_count / max(total_count, 1)

            graph.nodes[win_id]["fraud_count"] = fraud_count
            graph.nodes[win_id]["total_count"] = total_count
            graph.nodes[win_id]["fraud_rate"] = round(fraud_rate, 6)
            graph.nodes[win_id]["baseline_rate"] = round(baseline_rate, 6)
            graph.nodes[win_id]["elevated"] = fraud_rate > (2.0 * baseline_rate)
            graph.nodes[win_id]["elevation_factor"] = round(
                fraud_rate / max(baseline_rate, 1e-9), 2
            )
            enriched_windows += 1

        logger.info(
            "Time window enrichment: %d windows, baseline_rate=%.4f",
            enriched_windows, baseline_rate,
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
