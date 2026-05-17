import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML or JSON."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path:
            self.load(config_path)

    def load(self, path: str):
        """Load config from file."""
        path = Path(path)

        if path.suffix in [".yaml", ".yml"]:
            with open(path, "r") as f:
                self.config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-notation key."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        return self.config.get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        return self.config.copy()


def load_config(config_path: str) -> Dict[str, Any]:
    """Convenience function to load config."""
    loader = ConfigLoader(config_path)
    return loader.to_dict()


def create_default_config(output_path: str = "src/config/data_config.yaml"):
    """Create default data configuration."""
    default_config = {
        "data": {
            "source": "kaggle",
            "data_dir": "Sample_datasets",
            "dataset_file": "credit-card-1/creditcard.csv",
            "schema": "kaggle_mlg_ulb_fraud",
        },
        "split": {
            "num_clients": 3,
            "test_ratio": 0.15,
            "val_ratio": 0.20,
            "random_seed": 42,
            "non_iid": False,
            "sort_by": None,
            "output_dir": "data/splits",
            "prefix": "",
        },
        "columns": {"label_column": "Class", "label_positive": 1, "label_negative": 0},
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    return default_config
 
