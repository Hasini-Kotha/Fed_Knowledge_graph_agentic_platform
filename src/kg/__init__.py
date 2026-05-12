"""Knowledge Graph Layer (Layer 3) for the Federated Learning Platform."""
from src.kg.kg_schema import KGSchema, EntityType, RelationshipType
from src.kg.kg_builder import KnowledgeGraphBuilder
from src.kg.kg_enricher import KGEnricher
from src.kg.kg_query import KGQueryEngine
from src.kg.kg_analytics import KGAnalytics

__all__ = [
    "KGSchema",
    "EntityType",
    "RelationshipType",
    "KnowledgeGraphBuilder",
    "KGEnricher",
    "KGQueryEngine",
    "KGAnalytics",
]
