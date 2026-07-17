import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    """
    Loads and parses YAML configuration files for the reproduction pipeline.
    Provides standard retrieval and environment checking.
    """
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir

    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Loads a single YAML configuration file.
        """
        if not filename.endswith(".yaml"):
            filename = f"{filename}.yaml"
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads all core configuration files into a dictionary.
        """
        configs = {}
        for name in ["models", "datasets", "training", "evaluation", "apis"]:
            try:
                configs[name] = self.load_config(name)
            except FileNotFoundError:
                configs[name] = {}
        return configs
