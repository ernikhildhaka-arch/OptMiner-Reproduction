import os
import time
import argparse
import yaml
from typing import Dict, Any, Optional
from experiment_registry import ExperimentRegistry

class ExperimentRunner:
    """
    Runs registered experiments. Handles loading configs, setting up registry,
    executing tasks, and logging results dynamically.
    """
    def __init__(self, registry: ExperimentRegistry, logger: Optional[Any] = None):
        self.registry = registry
        self.logger = logger

    def run(self, name: str) -> Any:
        """
        Executes a registered experiment name, timing it, and logging metadata.
        """
        experiment = self.registry.get(name)
        config = experiment["config"]
        entrypoint = experiment["entrypoint"]
        
        print(f"=== Starting Experiment: {name} ===")
        start_time = time.time()
        
        try:
            result = entrypoint(config)
            runtime = time.time() - start_time
            print(f"[✓] Experiment {name} completed in {runtime:.2f}s")
            
            if self.logger:
                self.logger.log_run(
                    experiment_id=name,
                    model=config.get("model", "Unknown"),
                    dataset=config.get("dataset", "Unknown"),
                    prompt_version=config.get("prompt_version", "1.0"),
                    seed=config.get("seed", 42),
                    runtime=runtime,
                    api_usage={"tokens": 0},
                    solver_status="Success",
                    errors=None
                )
            return result
        except Exception as e:
            runtime = time.time() - start_time
            print(f"[✗] Experiment {name} failed: {e}")
            if self.logger:
                self.logger.log_run(
                    experiment_id=name,
                    model=config.get("model", "Unknown"),
                    dataset=config.get("dataset", "Unknown"),
                    prompt_version=config.get("prompt_version", "1.0"),
                    seed=config.get("seed", 42),
                    runtime=runtime,
                    api_usage={},
                    solver_status="Failure",
                    errors=str(e)
                )
            raise e
