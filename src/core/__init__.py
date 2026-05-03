"""Core module — The Domain-Agnostic Engine.

This layer sits above data and models to ensure they stay generic.
"""

from src.core.metadata_engine import MetadataMapper, create_default_mapping
from src.core.vectorizer import DynamicVectorizer
from src.core.contract import VectorContract, PrivacyConfig, FedProxConfig