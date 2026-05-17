"""KG Schema — Reads kg_config.yaml and creates typed definitions for entities and relationships.

This is the KG equivalent of src/data/schema.py.  All entity types, relationship
types, and their attributes are parsed from the config file — the Python code
never hardcodes domain-specific names like "merchant" or "account".
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EntityType:
    """Definition of a single entity type in the knowledge graph."""

    name: str
    id_prefix: str = ""
    id_column: str = ""
    attributes: List[Dict[str, str]] = field(default_factory=list)
    label_column: str = ""
    risk_score_attr: str = "risk_score"
    derived: bool = False

    # Bucket-derived entity fields
    bucket_column: str = ""
    bucket_edges: List[float] = field(default_factory=list)
    bucket_labels: List[str] = field(default_factory=list)
    bucket_count: int = 0
    bucket_width: float = 0.0

    def __repr__(self) -> str:
        kind = "derived" if self.derived else "primary"
        return f"EntityType(name={self.name!r}, kind={kind})"


@dataclass
class RelationshipType:
    """Definition of a single relationship type in the knowledge graph."""

    name: str
    source_entity: str
    target_entity: str
    weight_column: str = ""
    similarity_features: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.85
    top_k_neighbors: int = 10
    batch_size: int = 500

    def is_similarity_edge(self) -> bool:
        return len(self.similarity_features) > 0

    def __repr__(self) -> str:
        return f"RelationshipType({self.name!r}: {self.source_entity} → {self.target_entity})"


@dataclass
class KGSchema:
    """Complete schema for a knowledge graph, parsed from kg_config.yaml.

    Attributes:
        name: Human-readable name of the KG.
        version: Schema version string.
        domain: Domain identifier (e.g. 'fraud_detection').
        entity_types: List of entity type definitions.
        relationship_types: List of relationship type definitions.
        risk_config: Risk thresholds and propagation settings.
        analytics_config: Community detection and centrality settings.
        output_config: Output format and artifacts directory.
    """

    name: str
    version: str
    domain: str
    entity_types: List[EntityType]
    relationship_types: List[RelationshipType]
    risk_config: Dict[str, Any]
    analytics_config: Dict[str, Any]
    output_config: Dict[str, Any]

    @classmethod
    def from_config(cls, config_path: str) -> "KGSchema":
        """Load and parse a kg_config.yaml file into a KGSchema.

        Args:
            config_path: Path to kg_config.yaml.

        Returns:
            Fully populated KGSchema instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            KeyError: If required keys are missing.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"KG config not found: {config_path}")

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        kg = raw.get("kg", raw)

        # Parse entity types
        entity_types = []
        for e in kg.get("entities", []):
            entity_types.append(
                EntityType(
                    name=e["type"],
                    id_prefix=e.get("id_prefix", ""),
                    id_column=e.get("id_column", ""),
                    attributes=e.get("attributes", []),
                    label_column=e.get("label_column", ""),
                    risk_score_attr=e.get("risk_score_attr", "risk_score"),
                    derived=e.get("derived", False),
                    bucket_column=e.get("bucket_column", ""),
                    bucket_edges=e.get("bucket_edges", []),
                    bucket_labels=e.get("bucket_labels", []),
                    bucket_count=e.get("bucket_count", 0),
                )
            )

        # Parse relationship types
        relationship_types = []
        for r in kg.get("relationships", []):
            relationship_types.append(
                RelationshipType(
                    name=r["type"],
                    source_entity=r["source_entity"],
                    target_entity=r["target_entity"],
                    weight_column=r.get("weight_column", ""),
                    similarity_features=r.get("similarity_features", []),
                    similarity_threshold=r.get("similarity_threshold", 0.85),
                    top_k_neighbors=r.get("top_k_neighbors", 10),
                    batch_size=r.get("batch_size", 500),
                )
            )

        schema = cls(
            name=kg.get("name", "unnamed_kg"),
            version=kg.get("version", "0.0.0"),
            domain=kg.get("domain", "unknown"),
            entity_types=entity_types,
            relationship_types=relationship_types,
            risk_config=kg.get("risk", {}),
            analytics_config=kg.get("analytics", {}),
            output_config=kg.get("output", {}),
        )

        logger.info(
            "KGSchema loaded: name=%s, entities=%d, relationships=%d",
            schema.name,
            len(schema.entity_types),
            len(schema.relationship_types),
        )
        return schema

    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Look up an entity type by name."""
        for et in self.entity_types:
            if et.name == name:
                return et
        return None

    def get_relationship_type(self, name: str) -> Optional[RelationshipType]:
        """Look up a relationship type by name."""
        for rt in self.relationship_types:
            if rt.name == name:
                return rt
        return None

    def get_primary_entity(self) -> Optional[EntityType]:
        """Return the first non-derived entity (usually 'transaction')."""
        for et in self.entity_types:
            if not et.derived:
                return et
        return None

    def get_derived_entities(self) -> List[EntityType]:
        """Return all derived entity types."""
        return [et for et in self.entity_types if et.derived]

    def get_similarity_relationships(self) -> List[RelationshipType]:
        """Return relationship types that use feature similarity."""
        return [rt for rt in self.relationship_types if rt.is_similarity_edge()]

    def validate(self, df) -> Tuple[bool, List[str]]:
        """Validate that a DataFrame has the columns needed by this schema.

        Args:
            df: pandas DataFrame to validate.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        primary = self.get_primary_entity()

        if primary is None:
            issues.append("No primary (non-derived) entity type defined in schema.")
            return False, issues

        # Check attribute columns
        for attr in primary.attributes:
            col = attr.get("column", "")
            if col and col not in df.columns:
                issues.append(f"Missing attribute column '{col}' for entity '{primary.name}'")

        # Check label column
        if primary.label_column and primary.label_column not in df.columns:
            issues.append(f"Missing label column '{primary.label_column}'")

        # Check bucket columns for derived entities
        for et in self.get_derived_entities():
            if et.bucket_column and et.bucket_column not in df.columns:
                issues.append(
                    f"Missing bucket column '{et.bucket_column}' for derived entity '{et.name}'"
                )

        # Check similarity features
        for rt in self.get_similarity_relationships():
            for feat in rt.similarity_features:
                if feat not in df.columns:
                    issues.append(
                        f"Missing similarity feature '{feat}' for relationship '{rt.name}'"
                    )

        is_valid = len(issues) == 0
        if is_valid:
            logger.info("KG schema validation PASSED for %d columns", len(df.columns))
        else:
            logger.warning("KG schema validation FAILED: %d issues", len(issues))

        return is_valid, issues

    def summary(self) -> Dict[str, Any]:
        """Return a human-readable summary dict."""
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "entity_types": [et.name for et in self.entity_types],
            "relationship_types": [rt.name for rt in self.relationship_types],
            "risk_thresholds": self.risk_config,
        }

    def __repr__(self) -> str:
        return (
            f"KGSchema(name={self.name!r}, domain={self.domain!r}, "
            f"entities={len(self.entity_types)}, relationships={len(self.relationship_types)})"
        )
