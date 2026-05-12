"""KG Query Engine — Provides a clean API for downstream layers to query graph evidence.

This is the main output interface of the KG layer.  The Explainability layer
(Layer 4, LLM) and the Agentic Engine (Layer 5, LangGraph) call these methods
to get structured evidence bundles for flagged transactions.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from src.kg.kg_schema import KGSchema

logger = logging.getLogger(__name__)


class KGQueryEngine:
    """Query engine for the enriched knowledge graph.

    Provides methods to extract structured context for individual transactions,
    find similar flagged transactions, and generate evidence bundles for the
    explainability layer.

    Args:
        graph: Enriched nx.Graph with risk scores and tiers.
        schema: KGSchema instance.
    """

    def __init__(self, graph: nx.Graph, schema: KGSchema):
        self.graph = graph
        self.schema = schema

        primary = schema.get_primary_entity()
        self._prefix = primary.id_prefix or f"{primary.name}_" if primary else "txn_"
        self._risk_attr = primary.risk_score_attr if primary else "risk_score"
        self._primary_type = primary.name if primary else "transaction"

    def get_transaction_context(self, transaction_id: str) -> Dict[str, Any]:
        """Get full context for a single transaction node.

        Args:
            transaction_id: Node ID in the graph (e.g. 'txn_42301').

        Returns:
            Dict with risk score, tier, neighborhood stats, connected entities,
            and similar transactions.
        """
        if not self.graph.has_node(transaction_id):
            return {"error": f"Node {transaction_id} not found in graph"}

        node_data = dict(self.graph.nodes[transaction_id])

        # Basic info
        context: Dict[str, Any] = {
            "transaction_id": transaction_id,
            "risk_score": node_data.get(self._risk_attr, 0.0),
            "risk_tier": node_data.get("risk_tier", "unknown"),
            "ground_truth": node_data.get("ground_truth", None),
            "neighborhood_risk": node_data.get("neighborhood_risk", 0.0),
            "neighbor_count": node_data.get("neighbor_count", 0),
        }

        # Add entity attributes
        for attr_def in (self.schema.get_primary_entity().attributes if self.schema.get_primary_entity() else []):
            attr_name = attr_def.get("attr_name", "")
            if attr_name in node_data:
                context[attr_name] = node_data[attr_name]

        # Connected derived entities
        connected = {}
        for neighbor in self.graph.neighbors(transaction_id):
            n_data = self.graph.nodes.get(neighbor, {})
            n_type = n_data.get("entity_type", "unknown")

            if n_type != self._primary_type:
                edge_data = self.graph.edges[transaction_id, neighbor]
                rel = edge_data.get("relationship", "CONNECTED")
                bucket_label = n_data.get("bucket_label", n_data.get("window_index", neighbor))
                connected[rel] = str(bucket_label)

        context["connected_entities"] = connected

        # Count connected high-risk transactions
        high_risk_neighbors = []
        for neighbor in self.graph.neighbors(transaction_id):
            n_data = self.graph.nodes.get(neighbor, {})
            if n_data.get("entity_type") == self._primary_type:
                if n_data.get("risk_tier") == "high":
                    high_risk_neighbors.append({
                        "id": neighbor,
                        "risk_score": n_data.get(self._risk_attr, 0.0),
                    })

        context["connected_high_risk_count"] = len(high_risk_neighbors)

        return context

    def get_similar_flagged_transactions(
        self, transaction_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find transactions connected by similarity edges that are also high-risk.

        Args:
            transaction_id: Source transaction node ID.
            top_k: Maximum number of similar transactions to return.

        Returns:
            List of dicts with id, risk_score, similarity, risk_tier.
        """
        if not self.graph.has_node(transaction_id):
            return []

        similar = []
        for neighbor in self.graph.neighbors(transaction_id):
            n_data = self.graph.nodes.get(neighbor, {})
            if n_data.get("entity_type") != self._primary_type:
                continue

            edge_data = self.graph.edges[transaction_id, neighbor]
            if edge_data.get("relationship") != "SIMILAR_PATTERN":
                continue

            similar.append({
                "id": neighbor,
                "risk_score": n_data.get(self._risk_attr, 0.0),
                "risk_tier": n_data.get("risk_tier", "unknown"),
                "similarity": edge_data.get("similarity", 0.0),
            })

        # Sort by similarity descending
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:top_k]

    def get_high_risk_subgraph(self, threshold: Optional[float] = None) -> nx.Graph:
        """Extract the subgraph of all high-risk transaction nodes and their connections.

        Args:
            threshold: Risk score threshold (defaults to schema config).

        Returns:
            nx.Graph subgraph containing only high-risk nodes and edges between them.
        """
        if threshold is None:
            threshold = self.schema.risk_config.get("high_risk_threshold", 0.7)

        high_risk_nodes = [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get(self._risk_attr, 0.0) >= threshold
               and data.get("entity_type") == self._primary_type
        ]

        return self.graph.subgraph(high_risk_nodes).copy()

    def get_community_summary(self, graph: nx.Graph, transaction_id: str) -> Dict[str, Any]:
        """Get summary of the community that a transaction belongs to.

        Requires that community detection has already been run (community_id attribute).

        Args:
            graph: Graph with community_id attributes.
            transaction_id: Node ID to look up.

        Returns:
            Dict with community stats.
        """
        if not graph.has_node(transaction_id):
            return {"community_id": -1}

        community_id = graph.nodes[transaction_id].get("community_id", -1)

        if community_id == -1:
            return {"community_id": -1}

        # Find all nodes in same community
        community_nodes = [
            nid for nid, data in graph.nodes(data=True)
            if data.get("community_id") == community_id
               and data.get("entity_type") == self._primary_type
        ]

        risk_scores = [
            graph.nodes[nid].get(self._risk_attr, 0.0)
            for nid in community_nodes
        ]

        high_risk_count = sum(
            1 for nid in community_nodes
            if graph.nodes[nid].get("risk_tier") == "high"
        )

        return {
            "community_id": int(community_id),
            "size": len(community_nodes),
            "avg_risk": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "max_risk": float(np.max(risk_scores)) if risk_scores else 0.0,
            "high_risk_count": high_risk_count,
            "high_risk_ratio": high_risk_count / max(len(community_nodes), 1),
        }

    def generate_evidence_bundle(self, transaction_id: str) -> Dict[str, Any]:
        """Generate the complete evidence bundle for the Explainability layer.

        This is the MAIN OUTPUT consumed by Layer 4 (LLM).  It combines:
        - Transaction context (risk score, tier, neighborhood)
        - Similar flagged transactions
        - Community membership
        - A plain-English summary string (template-based, no LLM needed)

        Args:
            transaction_id: Node ID to generate evidence for.

        Returns:
            Complete evidence bundle dict.
        """
        # Get base context
        context = self.get_transaction_context(transaction_id)

        if "error" in context:
            return context

        # Get similar transactions
        similar = self.get_similar_flagged_transactions(transaction_id, top_k=5)
        context["similar_flagged_transactions"] = similar

        # Get community info
        community = self.get_community_summary(self.graph, transaction_id)
        context["community"] = community

        # Determine model prediction label
        risk_score = context.get("risk_score", 0.0)
        risk_tier = context.get("risk_tier", "low")
        context["model_prediction"] = "FRAUD" if risk_tier == "high" else (
            "SUSPICIOUS" if risk_tier == "medium" else "LEGITIMATE"
        )

        # Generate plain-English evidence summary (template-based)
        context["evidence_summary"] = self._generate_summary_text(context, similar, community)

        return context

    def _generate_summary_text(
        self,
        context: Dict[str, Any],
        similar: List[Dict[str, Any]],
        community: Dict[str, Any],
    ) -> str:
        """Generate a template-based plain-English summary.

        No LLM involved — just f-strings.  Layer 4 can later use this as input
        to a more sophisticated explanation generator.
        """
        parts = []

        risk_score = context.get("risk_score", 0.0)
        risk_tier = context.get("risk_tier", "unknown")
        parts.append(
            f"This transaction has a fraud risk score of {risk_score:.3f} "
            f"({risk_tier} risk)."
        )

        n_risk = context.get("neighborhood_risk", 0.0)
        n_count = context.get("neighbor_count", 0)
        if n_count > 0:
            parts.append(
                f"Its {n_count} graph neighbors have an average risk of {n_risk:.3f}."
            )

        connected = context.get("connected_entities", {})
        if connected:
            entity_parts = [f"{rel}: {val}" for rel, val in connected.items()]
            parts.append(f"Connected entities: {', '.join(entity_parts)}.")

        hr_count = context.get("connected_high_risk_count", 0)
        if hr_count > 0:
            parts.append(
                f"It is directly connected to {hr_count} high-risk transaction(s)."
            )

        if similar:
            sim_count = len(similar)
            avg_sim = np.mean([s["similarity"] for s in similar])
            parts.append(
                f"It shares feature patterns with {sim_count} similar transaction(s) "
                f"(avg similarity: {avg_sim:.2f})."
            )

        if community.get("community_id", -1) >= 0:
            parts.append(
                f"It belongs to community {community['community_id']} "
                f"(size: {community['size']}, "
                f"high-risk ratio: {community['high_risk_ratio']:.0%})."
            )

        return " ".join(parts)

    def get_top_risk_transactions(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return the top-K highest risk transaction nodes.

        Args:
            top_k: Number of transactions to return.

        Returns:
            List of (transaction_id, risk_score, risk_tier) dicts, sorted descending.
        """
        txn_nodes = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("entity_type") == self._primary_type:
                txn_nodes.append({
                    "transaction_id": nid,
                    "risk_score": data.get(self._risk_attr, 0.0),
                    "risk_tier": data.get("risk_tier", "unknown"),
                })

        txn_nodes.sort(key=lambda x: x["risk_score"], reverse=True)
        return txn_nodes[:top_k]
