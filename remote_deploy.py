import re, subprocess, sys

REPO = "/workspace/OptMiner-Reproduction"

# --- models.yaml ---
with open(f"{REPO}/config/models.yaml", "r") as f:
    content = f.read()
content = re.sub(r"backbone_model:.*", 'backbone_model: "Qwen/Qwen3-8B-Instruct"', content)
with open(f"{REPO}/config/models.yaml", "w") as f:
    f.write(content)
print("[OK] models.yaml updated:")
print(open(f"{REPO}/config/models.yaml").read())

# --- apis.yaml: use 4-bit quantization to fit 12GB VRAM on RTX 3060 ---
with open(f"{REPO}/config/apis.yaml", "r") as f:
    content = f.read()
content = re.sub(r"load_in_4bit:.*", "load_in_4bit: true", content)
content = re.sub(r"load_in_8bit:.*", "load_in_8bit: false", content)
with open(f"{REPO}/config/apis.yaml", "w") as f:
    f.write(content)
print("[OK] apis.yaml updated (4-bit quantization enabled for RTX 3060 12GB):")
print(open(f"{REPO}/config/apis.yaml").read())

# --- Install dependencies ---
print("\n[INFO] Installing Python dependencies...")
pkgs = [
    "pyyaml requests pytest",
    "gurobipy",
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "transformers accelerate",
    "bitsandbytes",
]
for pkg in pkgs:
    print(f"  pip install {pkg}")
    ret = subprocess.run(f"pip install {pkg}", shell=True, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  [WARN] {ret.stderr[-300:]}")
    else:
        print(f"  [OK]")

# --- Preprocess datasets ---
print("\n[INFO] Preprocessing datasets...")
ret = subprocess.run(f"cd {REPO} && python3 datasets/preprocess.py", shell=True, capture_output=True, text=True)
print(ret.stdout[-500:] if ret.stdout else "")
if ret.returncode != 0:
    print("[WARN] preprocess.py:", ret.stderr[-300:])
else:
    print("[OK] Datasets preprocessed.")

# --- Dry run: 1 sample ---
print("\n[INFO] Running 1-sample dry run to verify model loads...")
ret = subprocess.run(f"cd {REPO} && python3 experiments/run_claim_1.py --limit 1", shell=True, text=True)
if ret.returncode != 0:
    print("[FAIL] Dry run failed. Check error above.")
    sys.exit(1)
print("[OK] Dry run passed. Launching full evaluation in background...")

# --- Full evaluation ---
import os
os.system(f"cd {REPO} && nohup python3 experiments/run_claim_1.py --limit 1000 > {REPO}/claim1_run.log 2>&1 &")
print(f"\n[STARTED] Full Claim 1 evaluation is running in background.")
print(f"[LOG] Monitor progress with: tail -f {REPO}/claim1_run.log")
