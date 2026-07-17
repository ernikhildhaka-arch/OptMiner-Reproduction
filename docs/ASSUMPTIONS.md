# Reproducibility Assumptions

This document lists all implementation assumptions and trade-offs made due to hardware, software, or API limitations.

## 1. Hardware and Resource Scope
- **Assumption:** A full-scale reproduction of Qwen3-8B R-GRPO training requires 8x NVIDIA 80GB GPUs (as detailed in Appendix D.1). Because local hardware is limited to a 4GB GPU, we assume training will be executed on a remote cluster or simulated using smaller model parameters (e.g., Qwen-0.5B).
- **Impact:** The code structure and interface (`veRL`) are designed to scale to multi-GPU environments while permitting mock runs locally.

## 2. API Services and Mock fallbacks
- **Assumption:** MinerU and DeepSeek-V3 APIs are subject to credentials, rate limits, and network latency. We assume these calls can fail or run out of credits.
- **Impact:** We built modular LLM clients with standard retry and caching behaviors to minimize redundant calls and network overhead.

## 3. Gurobi Solver Licensing
- **Assumption:** The execution environment contains an active Gurobi solver license. In the absence of a license, evaluations will default to printing a license error but still parse code structural metrics.
