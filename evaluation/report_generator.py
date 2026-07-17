import os
import json
import csv
from typing import List, Dict, Any

class ReportGenerator:
    """
    Generates summary report tables, dataset-wise summaries, CSV/JSON outputs,
    and formats failure mode analysis reports in results/.
    """
    def __init__(self, log_dir: str = "results"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def save_reports(self, results: List[Dict[str, Any]], summary: Dict[str, Any], experiment_id: str):
        """
        Saves run stats to JSON, CSV, and Markdown files.
        """
        # 1. Save JSON Report
        json_path = os.path.join(self.log_dir, f"{experiment_id}_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "runs": results}, f, indent=4)
            
        # 2. Save CSV Report
        csv_path = os.path.join(self.log_dir, f"{experiment_id}_report.csv")
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
                
        # 3. Generate Markdown Report
        md_path = os.path.join(self.log_dir, f"{experiment_id}_summary.md")
        md_content = self.generate_markdown_content(results, summary, experiment_id)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
            
    def generate_markdown_content(self, results: List[Dict[str, Any]], summary: Dict[str, Any], experiment_id: str) -> str:
        """
        Formats experiment logs into a Markdown table with failure modes.
        """
        total = len(results)
        if total == 0:
            return f"# Summary Report: {experiment_id}\nNo runs processed."
            
        successes = sum(1 for r in results if r.get("correct", False))
        failures = total - successes
        
        # Failure modes categories
        syntax_errors = sum(1 for r in results if r.get("status") == "Syntax/Runtime Error")
        timeouts = sum(1 for r in results if r.get("status") == "Timeout")
        mismatched_objectives = sum(1 for r in results if r.get("status") == "Optimal" and not r.get("correct"))
        nomodel_errors = sum(1 for r in results if r.get("status") == "No Model Detected")
        
        md = [
            f"# Opt-Miner Claim 1 Evaluation Summary: {experiment_id}",
            "",
            "## Overall Performance",
            f"- **Total Problems Tested:** {total}",
            f"- **Pass@1 Success Rate:** {summary.get('pass_at_1_accuracy', 0.0):.2f}%",
            f"- **Successful Solves:** {successes}",
            f"- **Solver Run Failure Count:** {failures}",
            f"- **Average Solver Runtime:** {summary.get('average_solver_runtime', 0.0):.2f}s",
            "",
            "## Structural Correspondence Rates",
            f"- **Var Match Rate:** {summary.get('var_match_accuracy', 0.0):.2f}%",
            f"- **Bin Match Rate:** {summary.get('bin_match_accuracy', 0.0):.2f}%",
            f"- **Cons Match Rate:** {summary.get('cons_match_accuracy', 0.0):.2f}%",
            "",
            "## Failure Mode Analysis",
            "| Failure Mode | Count | Percentage |",
            "| :--- | :--- | :--- |",
            f"| Syntax / Runtime Errors | {syntax_errors} | {(syntax_errors/total)*100 if total > 0 else 0.0:.2f}% |",
            f"| Gurobi Timeouts | {timeouts} | {(timeouts/total)*100 if total > 0 else 0.0:.2f}% |",
            f"| Mismatched Objective Values | {mismatched_objectives} | {(mismatched_objectives/total)*100 if total > 0 else 0.0:.2f}% |",
            f"| No gp.Model Instance Found | {nomodel_errors} | {(nomodel_errors/total)*100 if total > 0 else 0.0:.2f}% |",
            "",
            "## Individual Problem Results Table",
            "| Problem ID | Dataset | Model | Status | Runtime | Pass@1 Correct |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ]
        
        for r in results:
            md.append(
                f"| {r.get('problem_id')} | {r.get('dataset')} | {r.get('model')} | "
                f"{r.get('status')} | {r.get('runtime', 0.0):.2f}s | {r.get('correct')} |"
            )
            
        return "\n".join(md)
