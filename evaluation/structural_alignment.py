from typing import Dict, Any, Tuple

class StructuralAlignmentEvaluator:
    """
    Compares the structural parameters (Var Match, Bin Match, Cons Match)
    as defined in Table 8. Generates detailed mismatch reports.
    """
    def evaluate(self, generated: Dict[str, int], ground_truth: Dict[str, int]) -> Tuple[Dict[str, bool], Dict[str, str]]:
        """
        Evaluates variable and constraint counts, reporting boolean matches and mismatch logs.
        """
        results = {
            "var_match": False,
            "bin_match": False,
            "cons_match": False
        }
        
        mismatches = {}
        
        # Total Variables Match
        gen_vars = generated.get("vars", 0)
        gt_vars = ground_truth.get("vars", 0)
        results["var_match"] = gen_vars == gt_vars
        if gen_vars != gt_vars:
            mismatches["vars"] = f"Generated {gen_vars} variables vs Ground Truth {gt_vars} variables"
            
        # Binary/Integer Variables Match
        gen_bins = generated.get("bins", 0)
        gt_bins = ground_truth.get("bins", 0)
        results["bin_match"] = gen_bins == gt_bins
        if gen_bins != gt_bins:
            mismatches["bins"] = f"Generated {gen_bins} binary/integer vars vs Ground Truth {gt_bins} binary/integer vars"
            
        # Constraints Match
        gen_cons = generated.get("constraints", 0)
        gt_cons = ground_truth.get("constraints", 0)
        results["cons_match"] = gen_cons == gt_cons
        if gen_cons != gt_cons:
            mismatches["constraints"] = f"Generated {gen_cons} constraints vs Ground Truth {gt_cons} constraints"
            
        return results, mismatches
