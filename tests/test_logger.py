import os
import json
import pytest
import logging
from src.logger import setup_logger, ExperimentLogger

def test_logger(tmp_path):
    log_dir = tmp_path / "logs"
    logger = setup_logger(str(log_dir))
    exp_logger = ExperimentLogger(logger, log_dir=str(log_dir))
    
    entry = exp_logger.log_run(
        experiment_id="test-exp-01",
        model="Qwen3-8B",
        dataset="MAMO",
        prompt_version="v1.0",
        seed=42,
        runtime=12.5,
        api_usage={"tokens": 120},
        solver_status="Optimal"
    )
    
    assert entry["experiment_id"] == "test-exp-01"
    assert entry["solver_status"] == "Optimal"
    
    # Check output exists in structured jsonl file
    jsonl_file = log_dir / "experiments_structured.jsonl"
    assert os.path.exists(jsonl_file)
    with open(jsonl_file, "r", encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["experiment_id"] == "test-exp-01"
        assert data["runtime"] == 12.5
