import os
import json
import argparse
from typing import Dict, Any, List

def preprocess_mamo(input_path: str, output_path: str):
    """
    Parses MAMO raw JSON Lines dataset files and reformats them.
    """
    if not os.path.exists(input_path):
        print(f"[-] Input file not found for MAMO: {input_path}")
        return
        
    try:
        processed = []
        with open(input_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                processed.append({
                    "problem_id": f"mamo_{item.get('id', idx)}",
                    "problem_description": item.get("Question", ""),
                    "optimal_value": item.get("Answer", None)
                })
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=4)
        print(f"[OK] Preprocessed MAMO into {output_path}")
    except Exception as e:
        print(f"[-] MAMO preprocessing failed: {e}")

def preprocess_nl4opt(input_path: str, output_path: str):
    """
    Parses NL4Opt raw JSON Lines task formulation files and reformats them.
    """
    if not os.path.exists(input_path):
        print(f"[-] Input file not found for NL4Opt: {input_path}")
        return
        
    try:
        processed = []
        with open(input_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                # Unpack numeric hashed wrapper
                if len(item) == 1:
                    item = list(item.values())[0]
                processed.append({
                    "problem_id": f"nl4opt_{idx}",
                    "problem_description": item.get("document", ""),
                    "variables": item.get("variables", []),
                    "constraints": item.get("constraints", [])
                })
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=4)
        print(f"[OK] Preprocessed NL4Opt into {output_path}")
    except Exception as e:
        print(f"[-] NL4Opt preprocessing failed: {e}")

def preprocess_other(input_path: str, output_path: str, name: str):
    """
    Standardizes IndustryOR, Resocratic, and OptMATH JSON files.
    """
    if not os.path.exists(input_path):
        print(f"[-] Input file not found for {name}: {input_path}")
        return
        
    try:
        processed = []
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        items = data if isinstance(data, list) else list(data.values())
        for idx, item in enumerate(items):
            desc = item.get("en_question", item.get("question", item.get("problem_description", "")))
            ans = item.get("en_answer", item.get("results", item.get("optimal_value", None)))
            processed.append({
                "problem_id": f"{name.lower()}_{item.get('id', idx)}",
                "problem_description": desc,
                "ground_truth_formula": ans
            })
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=4)
        print(f"[OK] Preprocessed {name} into {output_path}")
    except Exception as e:
        print(f"[-] {name} preprocessing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset preprocessing suite.")
    parser.add_argument("--mamo_easy_input", default="datasets/mamo/mamo_easy_lp.jsonl")
    parser.add_argument("--mamo_easy_output", default="datasets/mamo/processed/easy_lp.json")
    parser.add_argument("--mamo_complex_input", default="datasets/mamo/mamo_complex_lp.jsonl")
    parser.add_argument("--mamo_complex_output", default="datasets/mamo/processed/complex_lp.json")
    parser.add_argument("--nl4opt_train_input", default="datasets/nl4opt/train.jsonl")
    parser.add_argument("--nl4opt_train_output", default="datasets/nl4opt/processed/train.json")
    
    parser.add_argument("--industry_or_input", default="datasets/industry_or/dataset.json")
    parser.add_argument("--industry_or_output", default="datasets/industry_or/processed/dataset.json")
    parser.add_argument("--resocratic_input", default="datasets/resocratic/dataset.json")
    parser.add_argument("--resocratic_output", default="datasets/resocratic/processed/dataset.json")
    parser.add_argument("--optmath_input", default="datasets/optmath/optmath_bench.json")
    parser.add_argument("--optmath_output", default="datasets/optmath/processed/optmath_bench.json")
    args = parser.parse_args()
    
    preprocess_mamo(args.mamo_easy_input, args.mamo_easy_output)
    preprocess_mamo(args.mamo_complex_input, args.mamo_complex_output)
    preprocess_nl4opt(args.nl4opt_train_input, args.nl4opt_train_output)
    preprocess_other(args.industry_or_input, args.industry_or_output, "IndustryOR")
    preprocess_other(args.resocratic_input, args.resocratic_output, "Resocratic")
    preprocess_other(args.optmath_input, args.optmath_output, "OptMATH")

if __name__ == "__main__":
    main()
