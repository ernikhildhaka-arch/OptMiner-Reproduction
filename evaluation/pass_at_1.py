from typing import Optional

class PassAt1Evaluator:
    """
    Computes Pass@1 modeling accuracy exactly as defined in the paper.
    A generated model is correct if it achieves optimal status and its
    objective value matches the ground truth objective value.
    """
    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def evaluate(self, generated_val: Optional[float], ground_truth_val: Optional[float]) -> bool:
        """
        Compares numerical outputs within absolute and relative limits.
        """
        if generated_val is None or ground_truth_val is None:
            return False
            
        try:
            val_g = float(generated_val)
            val_t = float(ground_truth_val)
            
            abs_diff = abs(val_g - val_t)
            if val_t == 0.0:
                return abs_diff < self.tolerance
                
            rel_diff = abs_diff / abs(val_t)
            return rel_diff < self.tolerance or abs_diff < self.tolerance
        except (ValueError, TypeError):
            return False
