import re

REPO = "/workspace/OptMiner-Reproduction"

# Fix models.yaml: correct model ID to Qwen/Qwen3-8B
with open(f"{REPO}/config/models.yaml", "r") as f:
    content = f.read()
content = re.sub(r"backbone_model:.*", 'backbone_model: "Qwen/Qwen3-8B"', content)
with open(f"{REPO}/config/models.yaml", "w") as f:
    f.write(content)

print("[OK] models.yaml updated:")
print(open(f"{REPO}/config/models.yaml").read())
