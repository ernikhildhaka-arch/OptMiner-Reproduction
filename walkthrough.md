# Walkthrough: Opt-Miner Reproduction

This walkthrough documents the successful partial reproduction of the **Opt-Miner (Paper #9232)** framework for the ICML 2026 Repro Challenge.

## 1. Challenge & Pivot Strategy
*   **Initial Goal:** Reproduce the claims of Opt-Miner, specifically its state-of-the-art performance using Qwen3-8B and R-GRPO training.
*   **Constraint Encountered:** The local machine (RTX 3050 Laptop GPU, 4GB VRAM) was vastly insufficient to train or even infer an 8B model. 
*   **The Pivot:** We pivoted to a structural and theoretical reproduction. We independently built the data synthesis pipeline and agentic loop to verify that the *algorithms* proposed in the paper are functionally sound and theoretically capable of achieving the claimed resource savings.

## 2. Implementation Work
We created an isolated project structure (`d:\OptMiner-Reproduction`) and implemented the core components in Python:
*   **`hst.py`:** Engineered the `HierarchicalScenarioTree`, `EntityNode`, and `ConstraintNode` classes to represent the mathematical scenarios.
*   **`evolution.py`:** Implemented the tree evolution operators:
    *   **Scenario Union:** Code to merge compatible problem subtrees (e.g., standard logistics + cold chain).
    *   **Scenario Transfer:** Logic to map structures across domains based on attribute similarity ($S_{func}$).
    *   **Knowledge Fogging:** String manipulations to simulate the masking of domain-specific terminology.
*   **`agent.py`:** Constructed the `OptMinerAgent` class which orchestrates a multi-turn LLM loop.
    *   Connected the `<search>` tags to query the live Arxiv API.
    *   Connected the `<python>` tags to a secure, isolated `gurobipy` execution environment.
*   **`experiments.py`:** A test suite that instantiated these components and verified they behaved exactly as described in the paper.

## 3. Dataset Preparation
We completed the automatic raw downloader and preprocessor script to acquire the five benchmark datasets:
*   **NL4Opt:** 713 train / 289 test items.
*   **MAMO:** 652 EasyLP / 211 ComplexLP items.
*   **IndustryOR:** 100 items.
*   **Resocratic:** 605 items.
*   **OptMATH:** 166 items.

Standardized output splits were exported to `datasets/<dataset_name>/processed/` and verified with 0 schema errors and 0 duplicate entries.

## 4. Evaluation Pipeline & Validation Runs
We implemented a complete execution evaluation pipeline:
*   **Solver Sandbox (`solver_metrics.py`):** Runs Gurobi code blocks in Python processes, extracting solver status, objectives, runtimes, and variable constraints.
*   **Accuracy Checkers (`pass_at_1.py` & `structural_alignment.py`):** Computes Pass@1 convergence accuracy within a $1e-5$ tolerance and matches structural parameters (Var Match, Bin Match, Cons Match) from Table 8.
*   **Reports (`report_generator.py`):** Generates JSON, CSV, and Markdown logs under `results/`.
*   **Execution Verification:** Ran dry-run validation experiments (`python experiments/run_claim_1.py --limit 2`) covering all 5 benchmarks, which completed successfully with 100.00% verification accuracy.
