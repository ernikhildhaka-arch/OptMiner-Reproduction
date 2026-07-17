import os
import yaml
import pytest
from src.config_loader import ConfigLoader

def test_config_loader(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    dummy_config = {"backbone_model": "Qwen3-8B", "seed": 42}
    config_file = config_dir / "models.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(dummy_config, f)
        
    loader = ConfigLoader(config_dir=str(config_dir))
    data = loader.load_config("models")
    assert data["backbone_model"] == "Qwen3-8B"
    assert data["seed"] == 42
    
    # Assert missing files fail with exception
    with pytest.raises(FileNotFoundError):
        loader.load_config("nonexistent_yaml_file")
