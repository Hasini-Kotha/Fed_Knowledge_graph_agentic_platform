from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


@dataclass
class SchemaContract:
    """Platform-agnostic schema contract for tabular binary classification datasets.

    This contract defines the required structure for any tabular binary classification
    dataset to be used in the federated learning pipeline. It is domain-agnostic
    and reusable for fraud detection, healthcare, cybersecurity, etc.
    """

    name: str
    version: str = "1.0.0"

    label_column: str = "Class"
    label_positive: int = 1
    label_negative: int = 0

    required_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)
    timestamp_columns: List[str] = field(default_factory=list)

    feature_columns: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.feature_columns and not self.numeric_columns:
            self.numeric_columns = self.feature_columns.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "label_column": self.label_column,
            "label_positive": self.label_positive,
            "label_negative": self.label_negative,
            "required_columns": self.required_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "identifier_columns": self.identifier_columns,
            "timestamp_columns": self.timestamp_columns,
            "feature_columns": self.feature_columns,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaContract":
        return cls(
            name=data.get("name", "unknown"),
            version=data.get("version", "1.0.0"),
            label_column=data.get("label_column", "Class"),
            label_positive=data.get("label_positive", 1),
            label_negative=data.get("label_negative", 0),
            required_columns=data.get("required_columns", []),
            numeric_columns=data.get("numeric_columns", []),
            categorical_columns=data.get("categorical_columns", []),
            identifier_columns=data.get("identifier_columns", []),
            timestamp_columns=data.get("timestamp_columns", []),
            feature_columns=data.get("feature_columns", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SchemaContract":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self, df) -> tuple[bool, List[str]]:
        """Validate DataFrame against this schema contract.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not self.required_columns:
            errors.append("No required_columns defined in schema")
            return False, errors

        for col in self.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if self.label_column not in df.columns:
            errors.append(f"Label column '{self.label_column}' not found")

        if not self.feature_columns:
            if self.numeric_columns:
                pass
            elif self.categorical_columns:
                pass
            else:
                errors.append("No feature columns defined")

        return len(errors) == 0, errors

    def get_feature_columns(self, df) -> List[str]:
        """Get list of feature columns present in DataFrame."""
        features = []
        for col in self.numeric_columns or self.feature_columns:
            if col in df.columns:
                features.append(col)
        for col in self.categorical_columns:
            if col in df.columns:
                features.append(col)
        return features


def create_fraud_schema() -> SchemaContract:
    """Create schema contract for Kaggle MLG-ULB credit card fraud dataset.

    Dataset: IEEE-CIS Fraud Detection (Kaggle MLG-ULB)
    Source: Sample_datasets/credit-card-1/creditcard.csv

    Features:
    - Time: Seconds elapsed from first transaction
    - V1-V28: PCA-transformed features (anonymized)
    - Amount: Transaction amount
    - Class: 1=fraud, 0=legitimate
    """
    schema = SchemaContract(
        name="kaggle_mlg_ulb_fraud",
        version="1.0.0",
        label_column="Class",
        label_positive=1,
        label_negative=0,
        required_columns=["Class", "Amount"],
        numeric_columns=[
            "Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
        ],
        feature_columns=[
            "Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
        ],
        metadata={
            "description": "Kaggle MLG-ULB Credit Card Fraud Detection Dataset",
            "source": "IEEE-CIS Fraud Detection",
            "domain": "fraud_detection",
            "task": "binary_classification",
        },
    )
    return schema


def create_sim_fraud_schema() -> SchemaContract:
    """Create schema contract for simulated bank fraud dataset.

    This schema is based on the fraudTrain/fraudTest datasets from Sample_datasets.
    These have more features that require preprocessing.
    """
    schema = SchemaContract(
        name="simulated_bank_fraud",
        version="1.0.0",
        label_column="is_fraud",
        label_positive=1,
        label_negative=0,
        required_columns=["is_fraud", "amt"],
        numeric_columns=[
            "amt",
            "city_pop",
            "unix_time",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
        ],
        categorical_columns=["category", "gender", "state", "merchant"],
        identifier_columns=["cc_num", "trans_num"],
        timestamp_columns=["trans_date_trans_time", "dob"],
        metadata={
            "description": "Simulated Bank Fraud Detection Dataset",
            "source": "Simulated transactions",
            "domain": "fraud_detection",
            "task": "binary_classification",
            "notes": "Requires preprocessing for categorical encoding",
        },
    )
    return schema


DEFAULT_SCHEMAS = {
    "kaggle_mlg_ulb_fraud": create_fraud_schema,
    "simulated_bank_fraud": create_sim_fraud_schema,
}


def get_schema(name: str) -> SchemaContract:
    """Get schema by name or create default."""
    if name in DEFAULT_SCHEMAS:
        return DEFAULT_SCHEMAS[name]()
    raise ValueError(f"Unknown schema: {name}")
