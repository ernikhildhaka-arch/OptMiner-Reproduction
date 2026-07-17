import os
import json
import subprocess
from typing import Dict, Any, Optional

class SolverMetricsCollector:
    """
    Executes generated Gurobi optimization models in a separate python execution
    sandbox, collects solver run status, objective values, variable counts, and runtimes.
    """
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute_and_collect(self, code: str) -> Dict[str, Any]:
        """
        Runs code, captures return logs, parses optimization parameters.
        """
        metrics = {
            "status": "Error",
            "runtime": 0.0,
            "objective_value": None,
            "vars": 0,
            "bins": 0,
            "constraints": 0,
            "errors": ""
        }
        
        # Instrumentation suffix to query runtime Gurobi attributes directly
        suffix = """
# EXTRACT METRICS FOR EVALUATION
try:
    import json
    import gurobipy as gp
    found = False
    for name, obj in list(globals().items()):
        if isinstance(obj, gp.Model):
            found = True
            # Call optimize if Gurobi hasn't run yet
            if obj.Status in [gp.GRB.LOADED, 0]:
                obj.optimize()
                
            stats = {
                "status": "Optimal" if obj.Status == gp.GRB.OPTIMAL else str(obj.Status),
                "runtime": obj.Runtime,
                "objective_value": obj.ObjVal if obj.Status == gp.GRB.OPTIMAL else None,
                "vars": obj.NumVars,
                "bins": obj.NumBinVars + obj.NumIntVars,
                "constraints": obj.NumConstrs
            }
            with open("gurobi_metrics.json", "w") as f:
                json.dump(stats, f)
            break
    if not found:
        with open("gurobi_metrics.json", "w") as f:
            json.dump({"status": "NoModelDetected"}, f)
except Exception as e:
    with open("gurobi_metrics_error.json", "w") as f:
        json.dump({"error": str(e)}, f)
"""
        
        filepath = "temp_gurobi_eval.py"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code + "\n" + suffix)
            
        metrics_file = "gurobi_metrics.json"
        error_file = "gurobi_metrics_error.json"
        
        # Clear legacy files
        for f in [metrics_file, error_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
                
        try:
            result = subprocess.run(
                ["python", filepath],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                metrics["status"] = "Syntax/Runtime Error"
                metrics["errors"] = (result.stderr or result.stdout or "").strip()
                return metrics
                
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    stats = json.load(f)
                metrics.update(stats)
                if stats.get("status") == "NoModelDetected":
                    metrics["status"] = "No Model Detected"
                    metrics["errors"] = "Execution succeeded but no gp.Model instance was detected in globals."
            elif os.path.exists(error_file):
                with open(error_file, "r") as f:
                    err_stats = json.load(f)
                metrics["status"] = "Evaluation Instrument Error"
                metrics["errors"] = err_stats.get("error", "")
            else:
                metrics["status"] = "Unknown Failure"
                metrics["errors"] = "Execution finished without generating metric indicators."
                
        except subprocess.TimeoutExpired:
            metrics["status"] = "Timeout"
            metrics["errors"] = f"Execution exceeded limit of {self.timeout}s."
        except Exception as e:
            metrics["status"] = "Exception"
            metrics["errors"] = str(e)
        finally:
            # Clean up sandbox files
            for f in [filepath, metrics_file, error_file]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        return metrics
