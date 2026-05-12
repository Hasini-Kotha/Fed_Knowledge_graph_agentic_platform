"""KG Analytics — Community detection, centrality, and risk cluster analysis.

Runs graph-level analytics on the enriched knowledge graph.
Supports Louvain community detection (via greedy_modularity_communities as
a networkx built-in fallback) and degree/betweenness centrality.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx

from src.kg.kg_schema import KGSchema

logger = logging.getLogger(__name__)


class KGAnalytics:
    """Graph-level analytics for the knowledge graph.

    Args:
        graph: Enriched nx.Graph with risk attributes.
        schema: KGSchema instance.
    """

    def __init__(self, graph: nx.Graph, schema: KGSchema):
        self.graph = graph
        self.schema = schema

        primary = schema.get_primary_entity()
        self._primary_type = primary.name if primary else "transaction"
        self._risk_attr = primary.risk_score_attr if primary else "risk_score"

    def detect_communities(self, method: Optional[str] = None) -> Dict[str, Any]:
        """Detect communities in the graph and assign community_id to nodes.

        Supports:
          - 'louvain': Uses python-louvain if installed, falls back to networkx greedy.
          - 'greedy': Uses networkx's greedy_modularity_communities.

        Args:
            method: Detection method. Defaults to schema config.

        Returns:
            Dict with community count and size distribution.
        """
        if method is None:
            method = self.schema.analytics_config.get("community_detection", "louvain")

        # Extract transaction-only subgraph for community detection
        txn_nodes = [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get("entity_type") == self._primary_type
        ]

        if len(txn_nodes) < 2:
            logger.warning("Too few transaction nodes (%d) for community detection", len(txn_nodes))
            return {"num_communities": 0, "sizes": []}

        txn_subgraph = self.graph.subgraph(txn_nodes)

        if method == "louvain":
            partition = self._louvain_communities(txn_subgraph)
        else:
            partition = self._greedy_communities(txn_subgraph)

        # Assign community IDs back to the main graph
        for node_id, comm_id in partition.items():
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id]["community_id"] = int(comm_id)

        # Compute sizes
        community_sizes: Dict[int, int] = {}
        for comm_id in partition.values():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

        num_communities = len(community_sizes)
        sizes = sorted(community_sizes.values(), reverse=True)

        logger.info(
            "Community detection (%s): %d communities found, largest=%d",
            method, num_communities, sizes[0] if sizes else 0,
        )

        return {
            "method": method,
            "num_communities": num_communities,
            "sizes": sizes,
            "avg_size": float(np.mean(sizes)) if sizes else 0.0,
        }

    def _louvain_communities(self, subgraph: nx.Graph) -> Dict[str, int]:
        """Try python-louvain, fall back to networkx greedy."""
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(subgraph, random_state=42)
            logger.info("Using python-louvain for community detection")
            return partition
        except ImportError:
            logger.info("python-louvain not installed, falling back to greedy modularity")
            return self._greedy_communities(subgraph)

    def _greedy_communities(self, subgraph: nx.Graph) -> Dict[str, int]:
        """Use networkx's built-in greedy modularity community detection."""
        communities = nx.community.greedy_modularity_communities(subgraph)
        partition = {}
        for comm_id, members in enumerate(communities):
            for member in members:
                partition[member] = comm_id
        return partition

    def compute_centrality(self, metric: Optional[str] = None) -> Dict[str, Any]:
        """Compute centrality measures for transaction nodes.

        Args:
            metric: 'degree', 'betweenness', or 'closeness'. Defaults to config.

        Returns:
            Dict with top-K central nodes and statistics.
        """
        if metric is None:
            metric = self.schema.analytics_config.get("centrality_metric", "degree")

        top_k = self.schema.analytics_config.get("top_k_suspicious", 50)

        txn_nodes = [
            nid for nid, data in self.graph.nodes(data=True)
            if data.get("entity_type") == self._primary_type
        ]

        if len(txn_nodes) < 2:
            return {"metric": metric, "top_nodes": []}

        txn_subgraph = self.graph.subgraph(txn_nodes)

        if metric == "degree":
            centrality = nx.degree_centrality(txn_subgraph)
        elif metric == "betweenness":
            centrality = nx.betweenness_centrality(txn_subgraph, k=min(100, len(txn_nodes)))
        elif metric == "closeness":
            centrality = nx.closeness_centrality(txn_subgraph)
        else:
            centrality = nx.degree_centrality(txn_subgraph)

        # Store on nodes
        for node_id, score in centrality.items():
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id]["centrality"] = float(score)

        # Top-K
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [
            {
                "id": nid,
                "centrality": float(score),
                "risk_score": self.graph.nodes[nid].get(self._risk_attr, 0.0),
                "risk_tier": self.graph.nodes[nid].get("risk_tier", "unknown"),
            }
            for nid, score in sorted_nodes[:top_k]
        ]

        logger.info(
            "Centrality (%s): computed for %d nodes, top centrality=%.4f",
            metric, len(centrality), sorted_nodes[0][1] if sorted_nodes else 0.0,
        )

        return {
            "metric": metric,
            "total_nodes": len(centrality),
            "top_nodes": top_nodes,
        }

    def find_risk_clusters(self, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find communities where average risk exceeds the high-risk threshold.

        Args:
            min_cluster_size: Minimum number of members to count as a cluster.

        Returns:
            List of cluster summaries with id, size, avg_risk, high_risk_ratio.
        """
        high_threshold = self.schema.risk_config.get("high_risk_threshold", 0.7)

        # Group by community
        communities: Dict[int, List[str]] = {}
        for nid, data in self.graph.nodes(data=True):
            if data.get("entity_type") != self._primary_type:
                continue
            comm_id = data.get("community_id", -1)
            if comm_id < 0:
                continue
            communities.setdefault(comm_id, []).append(nid)

        risk_clusters = []
        for comm_id, members in communities.items():
            if len(members) < min_cluster_size:
                continue

            risks = [self.graph.nodes[m].get(self._risk_attr, 0.0) for m in members]
            avg_risk = float(np.mean(risks))
            high_count = sum(1 for r in risks if r >= high_threshold)

            if avg_risk >= self.schema.risk_config.get("medium_risk_threshold", 0.4):
                risk_clusters.append({
                    "community_id": int(comm_id),
                    "size": len(members),
                    "avg_risk": avg_risk,
                    "max_risk": float(np.max(risks)),
                    "high_risk_count": high_count,
                    "high_risk_ratio": high_count / len(members),
                })

        risk_clusters.sort(key=lambda x: x["avg_risk"], reverse=True)
        logger.info("Found %d risk clusters (min_size=%d)", len(risk_clusters), min_cluster_size)
        return risk_clusters

    def get_graph_summary(self) -> Dict[str, Any]:
        """Overall graph statistics."""
        txn_count = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("entity_type") == self._primary_type
        )
        derived_count = self.graph.number_of_nodes() - txn_count

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "transaction_nodes": txn_count,
            "derived_nodes": derived_count,
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
            "connected_components": nx.number_connected_components(self.graph),
        }
