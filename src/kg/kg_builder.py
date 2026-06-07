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
import time
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

    def build(self, df: pd.DataFrame, embedding_matrix: np.ndarray = None) -> nx.Graph:
        """Build the full knowledge graph from a transaction DataFrame.

        Args:
            df: DataFrame with columns matching the schema.
            embedding_matrix: Optional (N, D) float32 array of learned MLP embeddings.
                When provided, SIMILAR_PATTERN edges are built in this semantically
                rich space instead of raw config features.  This produces behaviorally
                meaningful clusters aligned with the FL model's fraud representations.

        Returns:
            Populated nx.Graph.
        """
        self.graph = nx.Graph()

        logger.info("Building KG: %d rows, schema=%s", len(df), self.schema.name)
        if embedding_matrix is not None:
            logger.info("Using learned MLP embeddings (%d-dim) for similarity edges", embedding_matrix.shape[1])
        else:
            logger.info("Using raw config features for similarity edges (no embeddings provided)")

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

        # Step 4: Similarity edges — use embeddings if provided
        for rt in self.schema.get_similarity_relationships():
            self._add_similarity_edges(df, rt, embedding_matrix=embedding_matrix)

        self._build_stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "embeddings_used": embedding_matrix is not None,
        }

        logger.info(
            "KG built: %d nodes, %d edges (embeddings=%s)",
            self._build_stats["total_nodes"],
            self._build_stats["total_edges"],
            self._build_stats["embeddings_used"],
        )

        return self.graph

    # ------------------------------------------------------------------
    # Incremental rolling-window update
    # ------------------------------------------------------------------

    def add_transactions(
        self,
        new_df: pd.DataFrame,
        new_embedding_matrix: np.ndarray = None,
        cutoff_timestamp: float = None,
        max_age_seconds: float = None,
        timestamp_attr: str = "timestamp",
        risk_tier_attr: str = "risk_tier",
        sim_edge_type: str = "SIMILAR_PATTERN",
    ) -> Dict[str, Any]:
        """Add new transactions to the existing graph (incremental rolling-window update).

        Eviction constraint (BOTH conditions must be true to evict a node):
          1. Node's timestamp < cutoff_timestamp  (old enough to evict)
          2. ALL of its SIMILAR_PATTERN neighbors have risk_tier == "low"
             (not part of any active fraud ring)

        If a node is old but has a high/medium-risk neighbor, it is RETAINED.
        This preserves cross-day fraud ring context even past the window expiry.

        After eviction, new transaction nodes are added and similarity edges are
        computed between new nodes and ALL remaining nodes in the graph.

        Args:
            new_df: DataFrame of new transactions (same schema as original build).
            new_embedding_matrix: Optional (N, D) MLP embedding matrix for new rows.
                If None, falls back to raw config features for similarity computation.
            cutoff_timestamp: Absolute timestamp cutoff (epoch seconds). Nodes with
                timestamp < this value are eviction candidates. Used directly when
                provided. Pass None to use max_age_seconds instead.
            max_age_seconds: Duration-based eviction (e.g., 172800 for 2 days).
                Computes cutoff as time.time() - max_age_seconds. Only used when
                cutoff_timestamp is None. Pass None to skip eviction entirely.
            timestamp_attr: Node attribute name storing the timestamp value.
                First tries "timestamp" (from data's Time column), then falls back
                to "_ingested_at" (ingestion time, always available).
            risk_tier_attr: Node attribute name storing the risk tier string.
                Defaults to "risk_tier" (set by KGEnricher.label_risk_tiers).
            sim_edge_type: Relationship type name for similarity edges.
                Defaults to "SIMILAR_PATTERN".

        Returns:
            Dict with eviction_stats and add_stats.
        """
        primary = self.schema.get_primary_entity()
        if primary is None:
            logger.error("No primary entity type found in schema — cannot add transactions.")
            return {}

        prefix = primary.id_prefix or f"{primary.name}_"

        # ── Step 0: Resolve cutoff timestamp ─────────────────────────────
        # Priority: explicit cutoff_timestamp > max_age_seconds > no eviction
        resolved_cutoff = cutoff_timestamp
        if resolved_cutoff is None and max_age_seconds is not None:
            resolved_cutoff = time.time() - max_age_seconds
            logger.info(
                "Using max_age_seconds=%.0f (cutoff=%.1f, ~%.1f days window)",
                max_age_seconds, resolved_cutoff, max_age_seconds / 86400,
            )

        # ── Step 1: Collect eviction candidates ───────────────────────────
        evicted_count = 0
        retained_due_to_risk = 0

        if resolved_cutoff is not None:
            txn_nodes = [
                nid for nid, d in self.graph.nodes(data=True)
                if d.get("entity_type") == primary.name
            ]

            candidates = []
            for nid in txn_nodes:
                # Try real timestamp first (from data's Time column), fall back to ingestion time
                ts = self.graph.nodes[nid].get(timestamp_attr,
                       self.graph.nodes[nid].get("_ingested_at", None))
                if ts is not None and float(ts) < resolved_cutoff:
                    candidates.append(nid)

            logger.info(
                "Eviction candidates (timestamp < %.1f): %d nodes",
                resolved_cutoff, len(candidates),
            )

            # Apply dual constraint: evict only if node itself AND ALL similar neighbors are low-risk
            safe_to_evict = []
            for nid in candidates:
                # Check 1: Is the node itself high or medium risk?
                node_risk = self.graph.nodes[nid].get(risk_tier_attr, "low")
                if node_risk in ("high", "medium"):
                    retained_due_to_risk += 1
                    continue

                # Get all SIMILAR_PATTERN neighbors
                sim_neighbors = [
                    nbr for nbr in self.graph.neighbors(nid)
                    if self.graph.edges[nid, nbr].get("relationship") == sim_edge_type
                    and self.graph.nodes[nbr].get("entity_type") == primary.name
                ]

                if not sim_neighbors:
                    # Isolated node and itself is low-risk — safe to evict
                    safe_to_evict.append(nid)
                    continue

                # Check 2: Is ANY neighbor high or medium risk?
                has_active_risk_neighbor = any(
                    self.graph.nodes[nbr].get(risk_tier_attr, "low") in ("high", "medium")
                    for nbr in sim_neighbors
                )

                if has_active_risk_neighbor:
                    # RETAIN — this old node is connected to an active fraud ring
                    retained_due_to_risk += 1
                else:
                    # Node is low-risk AND ALL neighbors are low-risk → safe to evict
                    safe_to_evict.append(nid)

            # Remove safe-to-evict nodes (edges are removed automatically by NetworkX)
            self.graph.remove_nodes_from(safe_to_evict)
            evicted_count = len(safe_to_evict)

            logger.info(
                "Evicted %d nodes | Retained %d old nodes (active fraud ring members)",
                evicted_count, retained_due_to_risk,
            )
        else:
            logger.info("No cutoff_timestamp provided — skipping eviction (add-only mode).")

        # ── Step 2: Add new primary nodes ─────────────────────────────────
        before_node_count = self.graph.number_of_nodes()

        # BUG FIX 1: Re-index new_df to prevent node ID collisions with existing nodes
        max_existing_idx = -1
        for nid, d in self.graph.nodes(data=True):
            if d.get("entity_type") == primary.name:
                try:
                    idx_val = int(nid.replace(prefix, ""))
                    max_existing_idx = max(max_existing_idx, idx_val)
                except ValueError:
                    pass

        if max_existing_idx >= 0:
            new_df = new_df.copy()
            new_df.index = range(max_existing_idx + 1, max_existing_idx + 1 + len(new_df))

        self._add_primary_nodes(new_df, primary)

        # ── Step 3: Ensure derived nodes exist (idempotent) ───────────────
        # Derived nodes (amount_bucket, time_window) persist across updates.
        # Only add them if they don't already exist in the graph.
        for et in self.schema.get_derived_entities():
            if et.bucket_column in new_df.columns:
                self._add_derived_nodes_if_missing(new_df, et)

        # ── Step 4: Add structural edges for new nodes only ───────────────
        new_node_ids = set(
            f"{prefix}{idx}" for idx in new_df.index
        )
        for rt in self.schema.relationship_types:
            if not rt.is_similarity_edge():
                self._add_structural_edges_for_subset(new_df, rt, new_node_ids)

        # ── Step 5: Add similarity edges ──────────────────────────────────
        # New nodes are compared against ALL remaining nodes in the graph.
        # This enables cross-batch fraud ring detection.
        existing_txn_nodes = [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get("entity_type") == primary.name
            and nid not in new_node_ids
        ]

        for rt in self.schema.get_similarity_relationships():
            self._add_incremental_similarity_edges(
                new_df=new_df,
                new_node_ids=new_node_ids,
                existing_txn_nodes=existing_txn_nodes,
                rel_type=rt,
                new_embedding_matrix=new_embedding_matrix,
                prefix=prefix,
            )

        after_node_count = self.graph.number_of_nodes()
        added_nodes = after_node_count - before_node_count + evicted_count

        update_stats = {
            "evicted_nodes": evicted_count,
            "retained_due_to_risk": retained_due_to_risk,
            "new_transactions_added": len(new_df),
            "actual_nodes_added": added_nodes,
            "total_nodes_after": self.graph.number_of_nodes(),
            "total_edges_after": self.graph.number_of_edges(),
        }

        logger.info("Incremental update complete: %s", update_stats)
        return update_stats

    def _add_derived_nodes_if_missing(
        self, df: pd.DataFrame, entity_type: EntityType
    ) -> None:
        """Add derived nodes only if they don't already exist in the graph.

        Safe to call multiple times — already-existing nodes are skipped.
        """
        if entity_type.bucket_column not in df.columns:
            return

        col_vals = df[entity_type.bucket_column]
        min_val = float(col_vals.min())
        max_val = float(col_vals.max())

        if entity_type.bucket_edges:
            for i, label in enumerate(entity_type.bucket_labels):
                node_id = f"{entity_type.name}_{i}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        entity_type=entity_type.name,
                        label=label,
                        bucket_index=i,
                    )
        elif getattr(entity_type, 'bucket_width', 0) > 0:
            # BUG FIX 2: Dynamic fixed-width bucketing (scales infinitely without min/max)
            for val in col_vals:
                bucket_idx = int(val / entity_type.bucket_width)
                node_id = f"{entity_type.name}_{bucket_idx}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        entity_type=entity_type.name,
                        bucket_index=bucket_idx,
                        range_start=bucket_idx * entity_type.bucket_width,
                    )
        elif getattr(entity_type, 'bucket_count', 0) > 0:
            for i in range(entity_type.bucket_count):
                node_id = f"{entity_type.name}_{i}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        entity_type=entity_type.name,
                        bucket_index=i,
                        min_val=min_val,
                        max_val=max_val,
                    )

    def _add_structural_edges_for_subset(
        self,
        df: pd.DataFrame,
        rel_type: RelationshipType,
        node_id_subset: set,
    ) -> None:
        """Add structural edges for a subset of transaction nodes only.

        Used during incremental updates to avoid re-adding edges for existing nodes.
        """
        primary = self.schema.get_primary_entity()
        target_et = self.schema.get_entity_type(rel_type.target_entity)
        if primary is None or target_et is None:
            return

        prefix = primary.id_prefix or f"{primary.name}_"
        col = target_et.bucket_column
        if col not in df.columns:
            return

        col_vals = df[col]
        min_val = float(col_vals.min())
        max_val = float(col_vals.max())
        edge_count = 0

        for idx, row in df.iterrows():
            source_id = f"{prefix}{idx}"
            if source_id not in node_id_subset:
                continue

            val = float(row[col])

            if target_et.bucket_edges:
                edges = target_et.bucket_edges
                bucket_idx = len(edges) - 2  # default to last bucket
                for i in range(len(edges) - 1):
                    if edges[i] <= val < edges[i + 1]:
                        bucket_idx = i
                        break
            elif getattr(target_et, 'bucket_width', 0) > 0:
                bucket_idx = int(val / target_et.bucket_width)
            else:
                span = max_val - min_val if max_val != min_val else 1.0
                bucket_idx = int((val - min_val) / span * target_et.bucket_count)
                bucket_idx = min(bucket_idx, target_et.bucket_count - 1)

            target_id = f"{target_et.name}_{bucket_idx}"
            if self.graph.has_node(target_id) and not self.graph.has_edge(source_id, target_id):
                self.graph.add_edge(
                    source_id, target_id,
                    relationship=rel_type.name,
                )
                edge_count += 1

        logger.info(
            "Added %d structural edges for new nodes (type=%s)",
            edge_count, rel_type.name,
        )

    def _add_incremental_similarity_edges(
        self,
        new_df: pd.DataFrame,
        new_node_ids: set,
        existing_txn_nodes: List[str],
        rel_type: RelationshipType,
        new_embedding_matrix: np.ndarray = None,
        prefix: str = "txn_",
    ) -> None:
        """Compute similarity edges between new nodes and ALL nodes in the graph.

        New nodes are compared against:
          - Other new nodes in this batch (intra-batch similarity)
          - Existing nodes still in the rolling window (cross-batch fraud ring detection)

        Args:
            new_df: DataFrame for the new batch.
            new_node_ids: Set of node IDs for the new batch.
            existing_txn_nodes: List of existing transaction node IDs in graph.
            rel_type: Similarity relationship type definition.
            new_embedding_matrix: Optional MLP embeddings for new_df rows.
            prefix: Transaction node ID prefix.
        """
        threshold = rel_type.similarity_threshold
        top_k = rel_type.top_k_neighbors
        batch_sz = rel_type.batch_size

        # ── Prepare new batch feature matrix ──────────────────────────────
        if new_embedding_matrix is not None:
            new_feat = new_embedding_matrix.astype(np.float32)
            source_label = f"MLP embeddings ({new_feat.shape[1]}-dim)"
        else:
            features = [f for f in rel_type.similarity_features if f in new_df.columns]
            if not features:
                logger.warning("No similarity features found — skipping incremental edges")
                return
            new_feat = new_df[features].values.astype(np.float32)
            source_label = f"raw features {features}"

        # ── Collect existing node embeddings from graph attributes ─────────
        # Existing nodes store their embedding in "embedding" attribute (if set).
        # If not available, we can only compare new→new similarity.
        existing_feat_list = []
        existing_ids = []
        for nid in existing_txn_nodes:
            emb = self.graph.nodes[nid].get("embedding", None)
            if emb is not None:
                existing_feat_list.append(emb)
                existing_ids.append(nid)

        # Build combined feature matrix: [new_batch | existing_with_embeddings]
        new_indices = list(new_df.index)

        if existing_feat_list:
            existing_feat = np.array(existing_feat_list, dtype=np.float32)
            all_feat = np.vstack([new_feat, existing_feat])
            all_ids = [f"{prefix}{idx}" for idx in new_indices] + existing_ids
        else:
            all_feat = new_feat
            all_ids = [f"{prefix}{idx}" for idx in new_indices]
            if existing_txn_nodes:
                logger.info(
                    "Existing nodes have no stored embeddings — computing intra-batch similarity only."
                )

        # Store embeddings on new nodes for future incremental calls
        for i, idx in enumerate(new_indices):
            node_id = f"{prefix}{idx}"
            if self.graph.has_node(node_id):
                self.graph.nodes[node_id]["embedding"] = new_feat[i].tolist()

        # ── Normalize for cosine similarity ───────────────────────────────
        norms = np.linalg.norm(all_feat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        all_feat_normed = all_feat / norms

        n_new = len(new_indices)
        n_all = len(all_ids)
        total_edges = 0

        logger.info(
            "Incremental similarity: %d new rows vs %d total (source=%s, threshold=%.2f)",
            n_new, n_all, source_label, threshold,
        )

        # ── Only compute similarity for new rows (rows 0..n_new-1) ────────
        for start in range(0, n_new, batch_sz):
            end = min(start + batch_sz, n_new)
            batch = all_feat_normed[start:end]   # (batch_sz, D)

            # Compare new batch rows against ALL nodes (new + existing)
            sim_matrix = batch @ all_feat_normed.T  # (batch_sz, n_all)

            for local_i in range(end - start):
                global_i = start + local_i
                sims = sim_matrix[local_i].copy()
                sims[global_i] = -1.0  # zero out self

                above_threshold = np.where(sims >= threshold)[0]
                if len(above_threshold) == 0:
                    continue

                if len(above_threshold) > top_k:
                    top_indices = above_threshold[
                        np.argsort(sims[above_threshold])[-top_k:]
                    ]
                else:
                    top_indices = above_threshold

                src_id = all_ids[global_i]
                for j in top_indices:
                    tgt_id = all_ids[j]
                    if src_id == tgt_id:
                        continue
                    if not self.graph.has_edge(src_id, tgt_id):
                        self.graph.add_edge(
                            src_id, tgt_id,
                            relationship=rel_type.name,
                            similarity=float(sims[j]),
                        )
                        total_edges += 1

        logger.info(
            "Added %d incremental similarity edges (type=%s, source=%s)",
            total_edges, rel_type.name, source_label,
        )



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

            # Add ingestion timestamp for lifecycle management
            attrs["_ingested_at"] = time.time()

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

        elif getattr(entity_type, 'bucket_width', 0) > 0:
            # Dynamic fixed-width bucketing
            for val in col_data:
                bucket_idx = int(val / entity_type.bucket_width)
                node_id = f"{entity_type.name}_{bucket_idx}"
                if not self.graph.has_node(node_id):
                    self.graph.add_node(
                        node_id,
                        entity_type=entity_type.name,
                        bucket_index=bucket_idx,
                        range_start=bucket_idx * entity_type.bucket_width,
                    )
            logger.info(
                "Added derived nodes (type=%s, dynamic-width)",
                entity_type.name,
            )

        elif getattr(entity_type, 'bucket_count', 0) > 0:
            # Equal-width bucketing (e.g., time windows)
            for i in range(entity_type.bucket_count):
                node_id = f"{entity_type.name}_{i}"
                if not self.graph.has_node(node_id):
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

        elif getattr(target_et, 'bucket_width', 0) > 0:
            for idx in range(len(df)):
                val = col_data[idx]
                bucket_idx = int(val / target_et.bucket_width)
                source_id = f"{prefix}{df.index[idx]}"
                target_id = f"{target_et.name}_{bucket_idx}"
                self.graph.add_edge(
                    source_id, target_id,
                    relationship=rel_type.name,
                )
                edge_count += 1

        elif getattr(target_et, 'bucket_count', 0) > 0:
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

    def _add_similarity_edges(
        self,
        df: pd.DataFrame,
        rel_type: RelationshipType,
        embedding_matrix: np.ndarray = None,
    ) -> None:
        """Add similarity edges between transactions (batched for memory safety).

        When embedding_matrix is provided (MLP hidden-layer activations), similarity
        is computed in the learned fraud-discriminative space.  Otherwise falls back
        to the raw config features from kg_config.yaml.

        Args:
            df: Transaction DataFrame.
            rel_type: Relationship type definition from schema.
            embedding_matrix: Optional (N, D) MLP embedding matrix.
        """
        primary = self.schema.get_primary_entity()
        prefix = primary.id_prefix or f"{primary.name}_" if primary else "txn_"
        threshold = rel_type.similarity_threshold
        top_k = rel_type.top_k_neighbors
        batch_sz = rel_type.batch_size

        # Decide feature source
        if embedding_matrix is not None:
            # Use learned MLP embeddings — semantically meaningful for fraud detection
            feat_matrix = embedding_matrix.astype(np.float32)
            source_label = f"MLP embeddings ({feat_matrix.shape[1]}-dim)"
        else:
            # Fallback: raw config features
            features = [f for f in rel_type.similarity_features if f in df.columns]
            if not features:
                logger.warning("No similarity features found for %s", rel_type.name)
                return
            feat_matrix = df[features].values.astype(np.float32)
            source_label = f"raw features {features}"

        # L2-normalise rows for cosine similarity via dot product
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        feat_normed = feat_matrix / norms

        n = len(df)
        indices = df.index.values
        total_edges = 0

        logger.info(
            "Computing similarity edges: %d rows, source=%s, batch=%d, top_k=%d, threshold=%.2f",
            n, source_label, batch_sz, top_k, threshold,
        )

        for start in range(0, n, batch_sz):
            end = min(start + batch_sz, n)
            batch = feat_normed[start:end]  # (batch_sz, D)

            # Cosine similarity: batch × all rows
            sim_matrix = batch @ feat_normed.T  # (batch_sz, n)

            for local_i in range(end - start):
                global_i = start + local_i
                sims = sim_matrix[local_i].copy()

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

        logger.info("Added %d similarity edges (type=%s, source=%s)", total_edges, rel_type.name, source_label)

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
