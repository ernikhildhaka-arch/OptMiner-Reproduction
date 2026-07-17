import os
import json
import yaml
from typing import Dict, Any, List, Tuple, Optional

class DatasetValidator:
    """
    Validates dataset files, checks for structural formats, missing fields,
    duplicate entries, and aggregates statistics for Claim 1 reproduction.
    Supports both standard .json and .jsonl (JSON Lines) files.
    """
    def __init__(self, config_paths: Optional[Dict[str, str]] = None):
        self.config_paths = config_paths or {}

    def check_missing_files(self, key: str, expected_files: List[str]) -> List[str]:
        dir_path = self.config_paths.get(key)
        if not dir_path or not os.path.exists(dir_path):
            return expected_files
        return [f for f in expected_files if not os.path.exists(os.path.join(dir_path, f))]

    def validate_schema_and_duplicates(self, filepath: str) -> Tuple[bool, int, List[str]]:
        if not os.path.exists(filepath):
            return False, 0, ["File not found"]
            
        errors = []
        duplicates = 0
        seen_values = set()
        
        try:
            items = []
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = list(data.values())
                    else:
                        return False, 0, ["JSON root must be a list or dict of objects."]

            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(f"Item {idx} is not a dictionary.")
                    continue
                
                # Unpack single-key nested dicts if it represents a hashed wrapper (NL4Opt format)
                if len(item) == 1 and not any(k in item for k in ["problem_description", "description", "Question", "document", "en_question", "question"]):
                    inner_item = list(item.values())[0]
                    if isinstance(inner_item, dict):
                        item = inner_item
                    
                # Schema check: look for at least one description field
                desc_fields = ["problem_description", "description", "Question", "document", "en_question", "question"]
                has_desc = any(f in item for f in desc_fields)
                if not has_desc:
                    errors.append(f"Item {idx} missing description fields {desc_fields}")
                        
                # Duplicate check based on text description
                val = None
                for f in desc_fields:
                    if f in item:
                        val = item[f]
                        break
                        
                if val:
                    if val in seen_values:
                        duplicates += 1
                    else:
                        seen_values.add(val)
                        
            return len(errors) == 0, duplicates, errors[:20]
        except Exception as e:
            return False, 0, [f"JSON parsing error: {e}"]

    def gather_stats(self, filepath: str) -> Dict[str, Any]:
        stats = {"count": 0, "obj_types": {}}
        if not os.path.exists(filepath):
            return stats
            
        try:
            items = []
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            items.append(json.loads(line))
                else:
                    data = json.load(f)
                    items = data if isinstance(data, list) else list(data.values())
            
            stats["count"] = len(items)
            
            for item in items:
                if len(item) == 1 and not any(k in item for k in ["problem_description", "description", "Question", "document", "en_question", "question"]):
                    inner_item = list(item.values())[0]
                    if isinstance(inner_item, dict):
                        item = inner_item
                obj = item.get("objective_type", "Unknown")
                stats["obj_types"][obj] = stats["obj_types"].get(obj, 0) + 1
        except Exception:
            pass
        return stats

    def generate_report(self, report_path: str = "results/dataset_validation_report.md") -> str:
        from download_datasets import DATASET_INFO
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report = [
            "# Dataset Validation Report",
            "This report documents the structural integrity, schema adherence, and duplicate states of the Claim 1 benchmarks.",
            ""
        ]
        
        for key, info in DATASET_INFO.items():
            dir_path = self.config_paths.get(key, "N/A")
            report.append(f"## {info['name']}")
            report.append(f"- **Configured Directory:** `{dir_path}`")
            
            missing = self.check_missing_files(key, list(info["url_map"].keys()))
            if missing:
                report.append(f"- **Status:** Missing files: {missing}")
                report.append("")
                continue
                
            report.append("- **Files Status:** All expected files present.")
            for filename in info["url_map"].keys():
                filepath = os.path.join(dir_path, filename)
                valid, dups, errors = self.validate_schema_and_duplicates(filepath)
                stats = self.gather_stats(filepath)
                
                report.append(f"  * **File:** `{filename}`")
                report.append(f"    - Valid Schema: {valid}")
                report.append(f"    - Total Entries: {stats['count']}")
                report.append(f"    - Duplicate Entries: {dups}")
                if errors:
                    report.append(f"    - Schema Errors: {errors}")
            report.append("")
            
        report_content = "\n".join(report)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_content

def main():
    with open("config/datasets.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = config.get("paths", {})
    validator = DatasetValidator(paths)
    report = validator.generate_report()
    print("=== Validation Report Generated ===")
    print(report)

if __name__ == "__main__":
    main()
