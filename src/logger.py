import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Sets up text file and stream logging.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("opt_miner_reproduction")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        log_file = os.path.join(log_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(levelname)s: %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logger

class ExperimentLogger:
    """
    Logs structured details of each experiment run in a JSONL file.
    """
    def __init__(self, logger: logging.Logger, log_dir: str = "logs"):
        self.logger = logger
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_run(self, 
                experiment_id: str, 
                model: str, 
                dataset: str, 
                prompt_version: str, 
                seed: int, 
                runtime: float, 
                api_usage: Dict[str, Any], 
                solver_status: str, 
                errors: Optional[str] = None) -> Dict[str, Any]:
        """
        Appends run details to structured logs and records to logger info.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "experiment_id": experiment_id,
            "model": model,
            "dataset": dataset,
            "prompt_version": prompt_version,
            "seed": seed,
            "runtime": runtime,
            "api_usage": api_usage,
            "solver_status": solver_status,
            "errors": errors or ""
        }
        
        json_log_path = os.path.join(self.log_dir, "experiments_structured.jsonl")
        with open(json_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        self.logger.info(
            f"Logged Run [{experiment_id}] | Model: {model} | Dataset: {dataset} | "
            f"Seed: {seed} | Runtime: {runtime:.2f}s | Solver Status: {solver_status} | Errors: {bool(errors)}"
        )
        return log_entry
