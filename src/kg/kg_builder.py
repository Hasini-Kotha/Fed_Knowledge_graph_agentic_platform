"""KG Builder — Constructs a NetworkX graph from transaction data based on the schema.

The builder reads entity and relationship definitions from the KGSchema and
creates the graph structure without any hardcoded column names.

For the MLG-ULB dataset (anonymized V1–V28), it uses:
  - Transaction nodes (one per row)
  - Amount-bucket nodes (derived from Amount column)
  - Time-window nodes (derived from Time column)
  - Similarity edges (cosine similarity on selected V-features, batched)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

from src.kg.kg_schema import KGSchema, EntityType, RelationshipType

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    """Builds a NetworkX graph from a DataFrame and a KGSchema.

    The graph is constructed in steps:
      1. Add primary entity nodes (one per row)
      2. Add derived entity nodes (buckets, time windows)
      3. Add structural edges (transaction → bucket, transaction → time window)
      4. Add similarity edges (transaction ↔ transaction, batched for memory safety)

    Args:
        schema: KGSchema instance parsed from kg_config.yaml.
    """

    def __init__(self, schema: KGSchema):
        self.schema = schema
        self.graph: nx.Graph = nx.Graph()
        self._build_stats: Dict[str, Any] = {}

    def build(self, df: pd.DataFrame) -> nx.Graph:
        """Build the full knowledge graph from a transaction DataFrame.

        Args:
            df: DataFrame with columns matching the schema.

        Returns:
            Populated nx.Graph.
        """
        self.graph = nx.Graph()

        logger.info("Building KG: %d rows, schema=%s", len(df), self.schema.name)

        # Validate
        is_valid, issues = self.schema.validate(df)
        if not is_valid:
            logger.warning("Schema validation issues: %s", issues)

        # Step 1: Primary entity nodes
        primary = self.schema.get_primary_entity()
        if primary is not None:
            self._add_primary_nodes(df, primary)

        # Step 2: Derived entity nodes
        for et in self.schema.get_derived_entities():
            self._add_derived_nodes(df, et)

        # Step 3: Structural edges
        for rt in self.schema.relationship_types:
            if not rt.is_similarity_edge():
                self._add_structural_edges(df, rt)

        # Step 4: Similarity edges (batched)
        for rt in self.schema.get_similarity_relationships():
            self._add_similarity_edges(df, rt)

        self._build_stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
        }

        logger.info(
            "KG built: %d nodes, %d edges",
            self._build_stats["total_nodes"],
            self._build_stats["total_edges"],
        )

        return self.graph

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def _add_primary_nodes(self, df: pd.DataFrame, entity_type: EntityType) -> None:
        """Add one node per row in the DataFrame."""
        prefix = entity_type.id_prefix or f"{entity_type.name}_"
        label_col = entity_type.label_column

        for idx, row in df.iterrows():
            node_id = f"{prefix}{idx}"
            attrs: Dict[str, Any] = {
                "entity_type": entity_type.name,
                "row_index": int(idx),
            }

            # Add configured attributes
            for attr_def in entity_type.attributes:
                col = attr_def.get("column", "")
                attr_name = attr_def.get("attr_name", col)
                if col in df.columns:
                    val = row[col]
                    attrs[attr_name] = float(val) if isinstance(val, (int, float, np.number)) else str(val)

            # Add ground truth label if available
            if label_col and label_col in df.columns:
                attrs["ground_truth"] = int(row[label_col])

            self.graph.add_node(node_id, **attrs)

        logger.info("Added %d primary nodes (type=%s)", len(df), entity_type.name)

    def _add_derived_nodes(self, df: pd.DataFrame, entity_type: EntityType) -> None:
        """Add derived entity nodes (buckets, time windows)."""
        if entity_type.bucket_column not in df.columns:
            logger.warning(
                "Skipping derived entity %s: column %s not found",
                entity_type.name, entity_type.bucket_column,
            )
            return

        col_data = df[entity_type.bucket_column].values

        if entity_type.bucket_edges and entity_type.bucket_labels:
            # Edge-based bucketing (e.g., amount ranges)
            for label in entity_type.bucket_labels:
                node_id = f"{entity_type.name}_{label}"
                self.graph.add_node(
                    node_id,
                    entity_type=entity_type.name,
                    bucket_label=label,
                )
            logger.info(
                "Added %d derived nodes (type=%s, edge-based)",
                len(entity_type.bucket_labels), entity_type.name,
            )

        elif entity_type.bucket_count > 0:
            # Equal-width bucketing (e.g., time windows)
            for i in range(entity_type.bucket_count):
                node_id = f"{entity_type.name}_{i}"
                self.graph.add_node(
                    node_id,
                    entity_type=entity_type.name,
                    window_index=i,
                )
            logger.info(
                "Added %d derived nodes (type=%s, count-based)",
                entity_type.bucket_count, entity_type.name,
            )

    # ------------------------------------------------------------------
    # Edge builders
    # ------------------------------------------------------------------

    def _add_structural_edges(self, df: pd.DataFrame, rel_type: RelationshipType) -> None:
        """Add edges between primary nodes and derived bucket/window nodes."""
        source_et = self.schema.get_entity_type(rel_type.source_entity)
        target_et = self.schema.get_entity_type(rel_type.target_entity)

        if source_et is None or target_et is None:
            logger.warning("Skipping edge type %s: entity type not found", rel_type.name)
            return

        prefix = source_et.id_prefix or f"{source_et.name}_"
        col = target_et.bucket_column

        if col not in df.columns:
            logger.warning("Skipping edge %s: column %s not found", rel_type.name, col)
            return

        col_data = df[col].values
        edge_count = 0

        if target_et.bucket_edges and target_et.bucket_labels:
            # Edge-based bucketing
            edges = target_et.bucket_edges
            labels = target_et.bucket_labels

            for idx in range(len(df)):
                val = col_data[idx]
                bucket_label = labels[-1]  # default to last bucket
                for i in range(len(edges) - 1):
                    if edges[i] <= val < edges[i + 1]:
                        bucket_label = labels[i]
                        break

                source_id = f"{prefix}{df.index[idx]}"
                target_id = f"{target_et.name}_{bucket_label}"
                self.graph.add_edge(
                    source_id, target_id,
                    relationship=rel_type.name,
                )
                edge_count += 1

        elif target_et.bucket_count > 0:
            # Equal-width windowing
            min_val = float(np.min(col_data))
            max_val = float(np.max(col_data))
            window_width = (max_val - min_val + 1e-9) / target_et.bucket_count

            for idx in range(len(df)):
                val = col_data[idx]
                window_idx = int((val - min_val) / window_width)
                window_idx = min(window_idx, target_et.bucket_count - 1)

                source_id = f"{prefix}{df.index[idx]}"
                target_id = f"{target_et.name}_{window_idx}"
                self.graph.add_edge(
                    source_id, target_id,
                    relationship=rel_type.name,
                )
                edge_count += 1

        logger.info("Added %d structural edges (type=%s)", edge_count, rel_type.name)

    def _add_similarity_edges(self, df: pd.DataFrame, rel_type: RelationshipType) -> None:
        """Add similarity edges between transactions (batched for memory safety).

        Uses cosine similarity on selected features.  For each transaction,
        only the top-K most similar neighbors are connected.
        """
        features = [f for f in rel_type.similarity_features if f in df.columns]
        if not features:
            logger.warning("No similarity features found for %s", rel_type.name)
            return

        primary = self.schema.get_primary_entity()
        prefix = primary.id_prefix or f"{primary.name}_" if primary else "txn_"
        threshold = rel_type.similarity_threshold
        top_k = rel_type.top_k_neighbors
        batch_sz = rel_type.batch_size

        # Extract feature matrix and normalize rows for cosine similarity
        feat_matrix = df[features].values.astype(np.float32)
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        feat_normed = feat_matrix / norms

        n = len(df)
        indices = df.index.values
        total_edges = 0

        logger.info(
            "Computing similarity edges: %d rows, %d features, batch=%d, top_k=%d",
            n, len(features), batch_sz, top_k,
        )

        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            batch = feat_normed[start:end]  # (batch_sz, D)

            # Cosine similarity: batch × all rows
            sim_matrix = batch @ feat_normed.T  # (batch_sz, n)

            for local_i in range(end - start):
                global_i = start + local_i
                sims = sim_matrix[local_i]

                # Zero out self-similarity
                sims[global_i] = -1.0

                # Find top-K above threshold
                above_threshold = np.where(sims >= threshold)[0]

                if len(above_threshold) == 0:
                    continue

                # Sort and take top-K
                if len(above_threshold) > top_k:
                    top_indices = above_threshold[
                        np.argsort(sims[above_threshold])[-top_k:]
                    ]
                else:
                    top_indices = above_threshold

                src_id = f"{prefix}{indices[global_i]}"
                for j in top_indices:
                    tgt_id = f"{prefix}{indices[j]}"
                    if not self.graph.has_edge(src_id, tgt_id):
                        self.graph.add_edge(
                            src_id, tgt_id,
                            relationship=rel_type.name,
                            similarity=float(sims[j]),
                        )
                        total_edges += 1

        logger.info("Added %d similarity edges (type=%s)", total_edges, rel_type.name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _count_node_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("entity_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_edge_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            t = data.get("relationship", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def get_build_stats(self) -> Dict[str, Any]:
        """Return statistics from the last build."""
        return self._build_stats

    def save(self, path: str) -> None:
        """Save graph to disk in GraphML format.

        GraphML does not support list/dict node attributes, so we convert
        any complex attributes to JSON strings before saving.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Deep-copy and sanitize attributes for GraphML
        g_copy = self.graph.copy()

        for node_id in g_copy.nodes():
            for key, val in list(g_copy.nodes[node_id].items()):
                if isinstance(val, (list, dict)):
                    g_copy.nodes[node_id][key] = json.dumps(val)
                elif isinstance(val, np.number):
                    g_copy.nodes[node_id][key] = float(val)

        for u, v in g_copy.edges():
            for key, val in list(g_copy.edges[u, v].items()):
                if isinstance(val, (list, dict)):
                    g_copy.edges[u, v][key] = json.dumps(val)
                elif isinstance(val, np.number):
                    g_copy.edges[u, v][key] = float(val)

        nx.write_graphml(g_copy, str(save_path))
        logger.info("Graph saved: %s (%d nodes, %d edges)", save_path, g_copy.number_of_nodes(), g_copy.number_of_edges())

    @classmethod
    def load(cls, path: str, schema: KGSchema) -> "KnowledgeGraphBuilder":
        """Load a previously saved graph."""
        builder = cls(schema)
        builder.graph = nx.read_graphml(str(path))
        logger.info("Graph loaded: %s", path)
        return builder
