# Reproducibility Checklists

## 1. Pre-requisites Checklist
- [x] Python 3.10+ installation
- [x] Gurobi solver installed with active academic/commercial license
- [x] PyYAML, requests, and pytest packages installed
- [ ] API keys for OpenAI / DeepSeek and MinerU added to env variables

## 2. Directory Structure Verification
- [x] `config/` holds model/dataset/solver configurations
- [x] `prompts/` holds raw text prompt templates
- [x] `datasets/` structured for caching
- [x] `tests/` contains isolated offline unit tests
- [x] `src/` contains llm interface, logger, and config loaders
