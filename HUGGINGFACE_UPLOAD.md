# Uploading Reproduction Artifacts to HuggingFace

This guide walks you through publishing the Opt‑Miner reproduction results on the HuggingFace Hub.
It creates a **Dataset** (for the raw result files) and a **Space** (for an interactive demo of the evaluation pipeline).

---

## 1️⃣ Prerequisites

- A HuggingFace account (free). Sign‑up at https://huggingface.co/join
- Install the `huggingface_hub` Python package:
  ```bash
  pip install huggingface_hub
  ```
- Authenticate locally (once):
  ```bash
  huggingface-cli login
  # Paste your HF token from https://huggingface.co/settings/tokens
  ```
- Ensure you have **git‑lfs** installed (required for large files):
  ```bash
  sudo apt-get install git-lfs   # Ubuntu
  # or `choco install git-lfs` on Windows
  git lfs install
  ```

---

## 2️⃣ Create a Repository for the Dataset

```bash
# From the project root
cd d:/OptMiner-Reproduction

# Create a new repo on the Hub (replace <YOUR_NAME> with your HF username)
huggingface-cli repo create OptMiner-Reproduction-dataset --type dataset

# Clone the newly created repo
git clone https://huggingface.co/<YOUR_NAME>/OptMiner-Reproduction-dataset
cd OptMiner-Reproduction-dataset
```

### Add the artifacts

Copy the `results/` folder (all markdown, CSV, JSON files) and the `TRACKIO_LOGBOOK.md` into the cloned repo:

```bash
cp -r ../results ./
cp ../TRACKIO_LOGBOOK.md ./
```

### Commit & Push

```bash
git lfs track "*.csv" "*.json" "*.md"
git add .
git commit -m "Add reproduction results, logbook, and claim reports"
git push origin main
```

Your dataset repository will now be publicly downloadable.

---

## 3️⃣ Create a Space for an Interactive Demo (optional)

A **Space** lets you host a small web UI that runs the evaluation pipeline on the fly.
We will build a minimal Streamlit app.

### 3.1 Scaffold the Space

```bash
# From the project root (still in OptMiner-Reproduction)
mkdir hf-space && cd hf-space

# Minimal Streamlit app
cat > app.py <<'PY'
import streamlit as st
import json, pandas as pd, os

st.title("Opt‑Miner Reproduction Demo")

# Load results summary if present
summary_path = os.path.abspath("../results/opt_miner_claim_1_summary.md")
if os.path.exists(summary_path):
    with open(summary_path) as f:
        st.markdown(f.read())
else:
    st.warning("Summary file not found.")

st.subheader("Download raw artifacts")
for fname in os.listdir("../results"):
    if fname.endswith(('.json', '.csv', '.md')):
        with open(os.path.join('..', 'results', fname), 'rb') as f:
            st.download_button(label=f"Download {fname}", data=f.read(), file_name=fname)
PY
```

Create a `requirements.txt`:
```bash
cat > requirements.txt <<'REQ'
streamlit
pandas
REQ
```

### 3.2 Push the Space

```bash
# Initialise a new repo on the Hub (type `space`)
huggingface-cli repo create OptMiner-Reproduction-demo --type space

# Clone it
git clone https://huggingface.co/spaces/<YOUR_NAME>/OptMiner-Reproduction-demo
cd OptMiner-Reproduction-demo

# Copy the app files
cp ../hf-space/app.py .
cp ../hf-space/requirements.txt .

git add .
git commit -m "Initial Streamlit demo for Opt‑Miner reproduction"
git push origin main
```

The Space will spin up (few minutes). Once ready, you can share the URL:
`https://huggingface.co/spaces/<YOUR_NAME>/OptMiner-Reproduction-demo`

---

## 4️⃣ Add Links to the OpenReview Submission

The ICML OpenReview page expects a **GitHub URL** and optionally a **HuggingFace dataset URL**.
Add the following markdown to the *Supplementary Material* section of your submission:

```markdown
**Reproduction Code & Data**
- GitHub: https://github.com/ernikhildhaka-arch/OptMiner-Reproduction
- HuggingFace Dataset: https://huggingface.co/<YOUR_NAME>/OptMiner-Reproduction-dataset
- Interactive Demo (Space): https://huggingface.co/spaces/<YOUR_NAME>/OptMiner-Reproduction-demo
```

---

## 5️⃣ Checklist Before Publishing

- [ ] All claim reports (`opt_miner_claim_*.md`) are present in the dataset repo
- [ ] `TRACKIO_LOGBOOK.md` is included for full reproducibility trace
- [ ] `README.md` mentions the HuggingFace links (already updated)
- [ ] Repository is **public** (check settings on the Hub)
- [ ] Space runs without errors (check the logs on the Space page)
- [ ] OpenReview submission updated with the new URLs

---

## 6️⃣ Troubleshooting

| Problem | Fix |
|---|---|
| `git lfs` uploads hang | Ensure you have enough bandwidth; try `git lfs push --all origin` |
| Space fails to start | Check `requirements.txt` for missing packages, view **Logs** tab on the Space page |
| Large files > 5 GB | Split into multiple files or host on an external storage (e.g., AWS S3) and provide a download link |

---

## 7️⃣ Final Note

With the dataset and interactive demo publicly available, reviewers and the community can fully verify every claim of the paper. This completes the reproducibility package for **Opt‑Miner**.

---

*Prepared by the Antigravity coding assistant on 2026‑07‑21.*
