# Remote GPU Compute Node Deployment Checklist

This guide outlines the commands and verification steps required to deploy the **Opt-Miner** reproduction pipeline onto a fresh Ubuntu remote compute node.

---

## 1. Environment Setup

### Clone Repository
```bash
git clone <repository_url> opt-miner-reproduction
cd opt-miner-reproduction
```

### Create and Activate Conda Environment
Ensure Conda is installed on the remote machine, then run:
```bash
conda create -n optminer python=3.10 -y
conda activate optminer
```

---

## 2. Dependency Installation

### Install PyTorch with CUDA Support
Select the appropriate CUDA version matching the remote GPU node (e.g., CUDA 12.1):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Core Pip Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Tool Verifications

### CUDA Verification
Verify that PyTorch correctly recognizes the GPU devices:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Gurobi License Verification
If utilizing Gurobi for model testing:
1. Obtain an Academic or Commercial license key from the [Gurobi Portal](https://portal.gurobi.com/).
2. Retrieve the license using the `grbgetkey` utility:
   ```bash
   grbgetkey <your-license-key-uuid>
   ```
3. Test that the Gurobi engine executes successfully:
   ```bash
   python -c "import gurobipy as gp; m = gp.Model(); m.addVar(); m.optimize(); print('[OK] Gurobi License Valid!')"
   ```

---

## 4. Running Experiments

### Step 1: Preprocess Datasets
Ensure all prepared datasets are reformatted into standard processed objects:
```bash
python datasets/preprocess.py
```

### Step 2: Dry Run Test
Run a quick validation (2 samples per dataset) to ensure there are no file path, loader, or solver errors:
```bash
python experiments/run_claim_1.py --limit 2
```

### Step 3: Run Full Benchmarks
To run the full evaluation pipeline (e.g., limit of 1000 samples per dataset) using the configured LLM client:
```bash
python experiments/run_claim_1.py --limit 1000
```

### Expected Outputs
*   **Logs:** Standard output logs written to `results/logs/`.
*   **Reports:**
    *   `results/opt_miner_claim_1_report.json`
    *   `results/opt_miner_claim_1_report.csv`
    *   `results/opt_miner_claim_1_summary.md` (Contains overall accuracy summaries and failure analysis tables).

---

## 5. Troubleshooting Guide

### Issue: `ModuleNotFoundError: No module named 'src'`
*   **Cause:** Sibling directories are not added to `sys.path` by default when running scripts.
*   **Resolution:** Our executable scripts auto-inject the project root to `sys.path` at runtime. However, if problems occur, set the environment variable:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

### Issue: `gurobipy.GurobiError: No license found`
*   **Cause:** Gurobi requires a valid license to run optimization checks.
*   **Resolution:** Set the `GRB_LICENSE_FILE` environment variable to point to your `gurobi.lic` path:
    ```bash
    export GRB_LICENSE_FILE=/path/to/gurobi.lic
    ```

### Issue: `gurobipy.GurobiError: Size limit exceeded`
*   **Cause:** Gurobi's free trial/restricted license restricts models to 2000 variables and 2000 constraints.
*   **Resolution:** Obtain and install an academic/commercial license using `grbgetkey`.
