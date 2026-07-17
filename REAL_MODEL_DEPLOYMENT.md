# Opt-Miner Deployment & Inference Setup: Real LLM Execution Guide

This document describes how to configure, deploy, and verify the real model inference pipeline for reproducing **Claim 1**.

---

## 1. Execution Modes

The repository supports two separate execution modes configured entirely via `config/models.yaml` and `config/apis.yaml`:

### Mode A: Local Pipeline Verification
*   **Purpose:** Verifies that the tokenizer, device loading, prompt templates, agentic loops, Gurobi sandbox, and report generators execute end-to-end without crashing.
*   **Model:** Lightweight `Qwen/Qwen2.5-0.5B-Instruct` (or another small compatible model).
*   **Hardware:** Laptop or small workstation (runs easily on CPU or 4GB VRAM GPU).
*   **Note:** This is purely for codebase verification and is **not** a scientific claim validation.

### Mode B: Scientific Evaluation
*   **Purpose:** Runs the actual reproduction benchmarks for Claim 1.
*   **Model:** `Qwen/Qwen3-8B-Instruct` (the closest officially available model to the paper's fine-tuned model).
*   **Hardware:** Remote GPU compute node (requires >16GB VRAM for unquantized loading, or 8GB-12GB VRAM for 8-bit/4-bit quantization).

---

## 2. Dependency Installation

Log into your remote GPU compute node and install the required packages:

```bash
# 1. Install PyTorch with CUDA support matching your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install Accelerate and Transformers
pip install transformers accelerate

# 3. Install bitsandbytes (required for 4-bit and 8-bit local model quantization)
pip install bitsandbytes

# 4. (Optional) Install vLLM if hosting model as a backend service
pip install vllm
```

---

## 3. GPU VRAM Memory Audit & Recommendations (Qwen3-8B-Instruct)

Estimates of VRAM usage for the 8B model based on token precision:

| Precision | Est. VRAM Weight Memory | Est. VRAM Total Memory (with KV Cache) | RTX A4000 (16 GB) | Dual RTX A4000 | RTX 5080 (16 GB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FP16** | ~16.0 GB | ~18.5 - 20.0 GB | Out-of-Memory | **Recommended (TP=2)** | Out-of-Memory |
| **8-bit** | ~8.0 GB | ~10.5 - 11.5 GB | **Recommended** | Supported | **Recommended** |
| **4-bit** | ~4.0 GB | ~5.5 - 6.5 GB | Supported | Supported | Supported |

### Recommendations
*   **Single GPU with 16GB VRAM (RTX A4000 / RTX 5080):** Run in **8-bit precision** to fit comfortably without running out of memory.
*   **Dual GPUs (Dual RTX A4000):** Run in **FP16 precision** using vLLM tensor parallelism (`tensor-parallel-size=2`) to split the weights across both cards.

---

## 4. Configuration Instructions

### Local Pipeline Verification (Mode A)
In `config/models.yaml`:
```yaml
backbone_model: "Qwen/Qwen2.5-0.5B-Instruct"
```
In `config/apis.yaml`:
```yaml
llm_api:
  provider: "huggingface"
  load_in_4bit: false
  load_in_8bit: false
```

### Scientific Evaluation (Mode B - Local Quantized Transformers)
In `config/models.yaml`:
```yaml
backbone_model: "Qwen/Qwen3-8B-Instruct"
```
In `config/apis.yaml`:
```yaml
llm_api:
  provider: "huggingface"
  load_in_8bit: true   # Enable 8-bit precision to fit 16GB VRAM limit
  load_in_4bit: false
```

### Scientific Evaluation (Mode B - vLLM Server Adapter)
1. Host the model on the compute node:
   ```bash
   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-Instruct --port 8000 --tensor-parallel-size 2
   ```
2. Configure `config/apis.yaml`:
   ```yaml
   llm_api:
     provider: "vllm"
     endpoint: "http://localhost:8000/v1/chat/completions"
   ```

---

## 5. Running Verification & Benchmarks

To execute the pipelines:

```bash
# Preprocess raw downloads first
python datasets/preprocess.py

# Run verification (1 sample)
python experiments/run_claim_1.py --limit 1

# Run full evaluation benchmark
python experiments/run_claim_1.py --limit 1000
```
