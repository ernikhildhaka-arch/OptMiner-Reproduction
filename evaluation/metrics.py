from typing import Dict, Any, List

class EvaluationMetrics:
    """
    Main aggregator for accumulating and computing evaluation statistics 
    (runtimes, correct rates, structural matches) across benchmark runs.
    """
    def __init__(self):
        self.results = []

    def add_result(self, problem_id: str, dataset: str, model: str, metrics: Dict[str, Any]):
        """
        Records a single evaluation run mapping.
        """
        self.results.append({
            "problem_id": problem_id,
            "dataset": dataset,
            "model": model,
            **metrics
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns average runtime, correct rates, and structural matches.
        """
        total = len(self.results)
        if total == 0:
            return {}
            
        successes = sum(1 for r in self.results if r.get("correct", False))
        accuracy = (successes / total) * 100
        avg_runtime = sum(r.get("runtime", 0.0) for r in self.results) / total
        
        var_matches = sum(1 for r in self.results if r.get("var_match", False)) / total * 100
        bin_matches = sum(1 for r in self.results if r.get("bin_match", False)) / total * 100
        cons_matches = sum(1 for r in self.results if r.get("cons_match", False)) / total * 100
        
        return {
            "total_count": total,
            "success_count": successes,
            "pass_at_1_accuracy": accuracy,
            "average_solver_runtime": avg_runtime,
            "var_match_accuracy": var_matches,
            "bin_match_accuracy": bin_matches,
            "cons_match_accuracy": cons_matches
        }
