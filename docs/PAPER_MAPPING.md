# Paper to Codebase Mapping

This document maps equations, tables, figures, and sections of the paper to their respective implementations in this repository.

## Section Mapping
- **Section 4.1.1 (HST):** Implemented in `implementation/hst.py`. Hierarchical representations of variables, entity subtrees, and constraint subtrees.
- **Section 4.1.2 (Tree Evolution):** Implemented in `implementation/evolution.py`. Identifies $S_{sem}$ and $S_{func}$, and applies scenario union, transfer, and fogging.
- **Appendix D.2 (Agentic Research Loop):** Implemented in `implementation/agent.py`. Iterative research loop utilizing `<search>` and `<python>` tags.

## Equations
- **Equation 2 ($S_{sem}$):** Coded in `implementation/evolution.py` as `compute_s_sem`.
- **Equation 3 ($S_{func}$):** Coded in `implementation/evolution.py` as `compute_s_func`.
- **Equation 4, 5, 6 (Union, Transfer, Fogging):** Handled by `evolution.py` wrappers.
- **Equation 7, 8 (F1 Retrieval & Hybrid Rewards):** Coded structure planned in evaluation and logger.
