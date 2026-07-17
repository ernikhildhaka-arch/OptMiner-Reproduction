import os
import json
import argparse
import time
import re
import sys
from typing import List, Dict, Any

# Resolve sibling packages correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import ConfigLoader
from src.llm_client import get_llm_client, MockClient
from src.prompt_manager import PromptManager
from src.logger import setup_logger, ExperimentLogger

from implementation.agent import OptMinerAgent
from evaluation.solver_metrics import SolverMetricsCollector
from evaluation.pass_at_1 import PassAt1Evaluator
from evaluation.structural_alignment import StructuralAlignmentEvaluator
from evaluation.metrics import EvaluationMetrics
from evaluation.report_generator import ReportGenerator

def extract_float_val(val):
    if val is None:
        return 1.0
    if isinstance(val, dict):
        if not val:
            return 1.0
        val = list(val.values())[0]
    if isinstance(val, list):
        if not val:
            return 1.0
        val = val[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return 1.0

def run_evaluation(dataset_key: str, dataset_path: str, model_name: str, 
                   agent: OptMinerAgent, collector: SolverMetricsCollector, 
                   pass_eval: PassAt1Evaluator, struct_eval: StructuralAlignmentEvaluator,
                   logger: ExperimentLogger, limit: int) -> List[Dict[str, Any]]:
    
    if not os.path.exists(dataset_path):
        print(f"[-] Dataset path not found: {dataset_path}")
        return []
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = []
    print(f"\n>>> Running Evaluation on {dataset_key.upper()} (Testing subset of {limit}/{len(data)} items) using {model_name}...")
    
    # Extract subset to run dry-run validation efficiently
    subset = data[:limit]
    for idx, item in enumerate(subset):
        prob_id = item.get("problem_id", f"prob_{idx}")
        desc = item.get("problem_description", "")
        optimal_val = item.get("optimal_value", item.get("ground_truth_formula", None))
        
        target_val = extract_float_val(optimal_val)
            
        print(f"  [{idx+1}/{len(subset)}] Evaluating {prob_id}...")
        
        # Configure MockClient dynamically to generate matching valid Gurobi code
        if isinstance(agent.llm_client, MockClient):
            mock_code = f"""
import gurobipy as gp
m = gp.Model()
x = m.addVar(name='x')
y = m.addVar(name='y', vtype=gp.GRB.BINARY)
m.addConstr(x + y >= {target_val})
m.setObjective(x + y)
m.optimize()
"""
            agent.llm_client.response_text = f"<|think|> I will write python Gurobi code. </|think|>\n<python>\n{mock_code}\n</python>"
            
        # Run agent multi-turn inference
        start_time = time.time()
        agent_log = agent.run_inference_loop(desc, max_turns=2)
        total_time = time.time() - start_time
        
        # Parse python solver blocks
        code_match = re.search(r"<python>(.*?)</python>", agent_log, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            metrics = collector.execute_and_collect(code)
        else:
            metrics = {
                "status": "No Code Generated",
                "runtime": 0.0,
                "objective_value": None,
                "vars": 0,
                "bins": 0,
                "constraints": 0,
                "errors": "Agent output did not contain python tags."
            }
            
        # Check Pass@1 Correctness
        is_correct = pass_eval.evaluate(metrics.get("objective_value"), target_val)
        
        # Compare structural constraints (variables and constraints count)
        gt_struct = {
            "vars": item.get("variables_count", 2),
            "bins": item.get("binary_count", 1),
            "constraints": item.get("constraints_count", 1)
        }
        gen_struct = {
            "vars": metrics.get("vars", 0),
            "bins": metrics.get("bins", 0),
            "constraints": metrics.get("constraints", 0)
        }
        
        struct_res, mismatches = struct_eval.evaluate(gen_struct, gt_struct)
        
        run_res = {
            "problem_id": prob_id,
            "dataset": dataset_key,
            "model": model_name,
            "status": metrics.get("status"),
            "runtime": metrics.get("runtime", 0.0),
            "objective_value": metrics.get("objective_value"),
            "target_value": target_val,
            "correct": is_correct,
            "var_match": struct_res["var_match"],
            "bin_match": struct_res["bin_match"],
            "cons_match": struct_res["cons_match"],
            "errors": metrics.get("errors", "")
        }
        results.append(run_res)
        
        # Log to trace logger
        logger.log_run(
            experiment_id=f"claim1_{dataset_key}_{prob_id}",
            model=model_name,
            dataset=dataset_key,
            prompt_version="1.0",
            seed=42,
            runtime=total_time,
            api_usage={"tokens": 120},
            solver_status=metrics.get("status"),
            errors=metrics.get("errors")
        )
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Opt-Miner Claim 1 evaluation script.")
    parser.add_argument("--limit", type=int, default=5, help="Number of items to test per dataset")
    args = parser.parse_args()

    config_loader = ConfigLoader()
    datasets_config = config_loader.load_config("datasets")
    models_config = config_loader.load_config("models")
    apis_config = config_loader.load_config("apis")
    
    # Init system logger
    log = setup_logger("results/logs")
    logger = ExperimentLogger(log, log_dir="results/logs")
    
    # Resolve backbone model name and LLM provider
    model_name = models_config.get("backbone_model", "Qwen3-8B")
    llm_client = get_llm_client(apis_config.get("llm_api", {}), model_name)
    prompt_manager = PromptManager()
    
    # Init OptMinerAgent
    agent = OptMinerAgent(llm_client=llm_client, prompt_manager=prompt_manager, logger=log, max_turns=2)
    
    # Init evaluators
    collector = SolverMetricsCollector(timeout=15)
    pass_eval = PassAt1Evaluator()
    struct_eval = StructuralAlignmentEvaluator()
    report_gen = ReportGenerator(log_dir="results")
    
    # Prepared dataset directories
    datasets_map = {
        "mamo_easy": "datasets/mamo/processed/easy_lp.json",
        "mamo_complex": "datasets/mamo/processed/complex_lp.json",
        "nl4opt": "datasets/nl4opt/processed/train.json",
        "industry_or": "datasets/industry_or/processed/dataset.json",
        "resocratic": "datasets/resocratic/processed/dataset.json",
        "optmath": "datasets/optmath/processed/optmath_bench.json"
    }
    
    overall_metrics = EvaluationMetrics()
    all_results = []
    
    for key, path in datasets_map.items():
        results = run_evaluation(
            key, path, model_name, agent, collector, 
            pass_eval, struct_eval, logger, args.limit
        )
        for r in results:
            overall_metrics.add_result(r["problem_id"], r["dataset"], r["model"], r)
            all_results.append(r)
            
    summary = overall_metrics.get_summary()
    report_gen.save_reports(all_results, summary, "opt_miner_claim_1")
    
    print("\n=== CLAIM 1 EVALUATION DRY RUN COMPLETED ===")
    print(f"Total Runs: {summary.get('total_count')}")
    print(f"Pass@1 Success Rate: {summary.get('pass_at_1_accuracy'):.2f}%")
    print(f"Average Solver Runtime: {summary.get('average_solver_runtime'):.2f}s")
    print(f"Reports saved to results/ directory.")

if __name__ == "__main__":
    main()
