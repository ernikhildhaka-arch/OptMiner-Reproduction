# Opt-Miner: Empowering Information-Seeking Agent with Tree-Guided Data Synthesis

This repository is a faithful reproduction framework for the ICML 2026 paper:
> **Opt-Miner: Empowering Information-Seeking Agent with Tree-Guided Data Synthesis for Optimization Modeling**

## Project Structure
```
├── config/             # YAML configurations (Models, datasets, evaluation, APIs)
├── datasets/           # Dataset loaders, download status trackers, and validators
├── docs/               # Assumptions, mapping lists, and compliance checklists
├── evaluation/         # Skeletons for Pass@1, structural alignment, and report generation
├── experiments/        # Registries and runners for reproducible experiments
├── implementation/     # Core algorithms (HST representation, tree evolution operators, agent)
├── prompts/            # Raw text prompt templates (Prompt_cmp, Prompt_syn, etc.)
├── src/                # Shared utilities (LLM client wrappers, loggers, config loaders)
└── tests/              # Unit tests verifying infrastructure offline
```

## Setup and Installation

### 1. Requirements
* Python 3.10+
* Gurobi Solver (with active academic/commercial license)
* Dependencies:
  ```bash
  pip install pyyaml requests pytest numpy pypdf
  ```

### 2. Configuration Setup
Configure parameters inside `config/` files:
* `models.yaml`: Set target backbone models (e.g. Qwen3-8B) and decoding settings.
* `datasets.yaml`: Paths mapping NL4Opt, MAMO, and other datasets.
* `apis.yaml`: Setup endpoints, credentials, and API parameters for OpenAI, DeepSeek, and MinerU.

## Dynamic Prompt Management
Prompt templates are isolated in `prompts/` to ensure no hardcoding:
* `Prompt_cmp.txt`: Used for domain equivalence.
* `Prompt_syn.txt`: Merges scenarios in the same domain.
* `Prompt_trans.txt`: Adapts logic across domains.
* `Prompt_abs.txt`: Mask details (fogging).
* `Prompt_research.txt`: Agentic research loop.

Loaded dynamically using:
```python
from src.prompt_manager import PromptManager
manager = PromptManager()
prompt = manager.get_prompt("Prompt_cmp", scenario_1="...", scenario_2="...")
```

## Running Unit Tests
All infrastructure components can be tested completely offline (without requesting actual APIs) by running:
```bash
pytest tests/
```
