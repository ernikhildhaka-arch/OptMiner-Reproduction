import os
import argparse
import yaml
import requests
from typing import Dict, Any, Optional

DATASET_INFO = {
    "nl4opt": {
        "name": "NL4Opt",
        "url_map": {
            "train.jsonl": "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/main/generation_data/train.jsonl",
            "test.jsonl": "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/main/generation_data/test.jsonl"
        },
        "license": "CC BY 4.0",
        "citation": "Ramamonjison et al., 2021",
        "repo": "https://github.com/nl4opt/nl4opt-competition"
    },
    "mamo": {
        "name": "MAMO",
        "url_map": {
            "mamo_easy_lp.jsonl": "https://raw.githubusercontent.com/FreedomIntelligence/Mamo/main/data/optimization/Easy_LP/mamo_easy_lp.jsonl",
            "mamo_complex_lp.jsonl": "https://raw.githubusercontent.com/FreedomIntelligence/Mamo/main/data/optimization/Complex_LP/mamo_complex_lp.jsonl"
        },
        "license": "Apache-2.0",
        "citation": "Huang et al., 2024",
        "repo": "https://github.com/FreedomIntelligence/Mamo"
    },
    "industry_or": {
        "name": "IndustryOR",
        "url_map": {
            "dataset.json": "https://huggingface.co/datasets/CardinalOperations/IndustryOR/resolve/main/IndustryOR.json"
        },
        "license": "MIT",
        "citation": "Huang et al., 2025",
        "repo": "https://huggingface.co/datasets/CardinalOperations/IndustryOR"
    },
    "resocratic": {
        "name": "Resocratic",
        "url_map": {
            "dataset.json": "https://raw.githubusercontent.com/yangzhch6/ReSocratic/main/data/OptiBench.json"
        },
        "license": "MIT",
        "citation": "Yang et al., 2025",
        "repo": "https://github.com/yangzhch6/ReSocratic"
    },
    "optmath": {
        "name": "OptMATH",
        "url_map": {
            "optmath_bench.json": "https://raw.githubusercontent.com/optsuite/OptMATH/main/benchmark/OptMATH_Bench.json"
        },
        "license": "Apache-2.0",
        "citation": "Lu et al., 2025",
        "repo": "https://github.com/optsuite/OptMATH"
    },
    "opt_miner_bench": {
        "name": "Opt-Miner-Bench",
        "url_map": {
            "bench_128.json": ""
        },
        "license": "Generated",
        "citation": "Liu et al., 2026 (Opt-Miner)",
        "repo": "N/A - Generated via pipeline"
    }
}

def load_config(config_path: str = "config/datasets.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_directories(config: Dict[str, Any]):
    paths = config.get("paths", {})
    print("=== Preparing Dataset Directory Structure ===")
    for key, path in paths.items():
        os.makedirs(path, exist_ok=True)
        print(f"[+] Prepared directory: {path}")

def download_file(url: str, dest_path: str) -> bool:
    """
    Downloads file from URL with HTTP Range resume support.
    """
    if not url:
        return False
        
    temp_dest = dest_path + ".tmp"
    headers = {}
    initial_pos = 0
    
    if os.path.exists(temp_dest):
        initial_pos = os.path.getsize(temp_dest)
        headers["Range"] = f"bytes={initial_pos}-"
        
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        if response.status_code == 206:
            mode = "ab"
        elif response.status_code == 200:
            mode = "wb"
            initial_pos = 0
        else:
            print(f"[-] HTTP error {response.status_code} requesting {url}")
            return False
            
        with open(temp_dest, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        os.replace(temp_dest, dest_path)
        print(f"[OK] Downloaded {dest_path}")
        return True
    except Exception as e:
        print(f"[-] Download error: {e}")
        return False

def execute_downloads(config: Dict[str, Any]):
    paths = config.get("paths", {})
    print("\n=== Executing Dataset Downloads ===")
    for key, info in DATASET_INFO.items():
        dir_path = paths.get(key)
        if not dir_path:
            continue
            
        for filename, url in info["url_map"].items():
            if not url:
                print(f"[i] {info['name']} - {filename} requires self-generation/synthesis.")
                continue
                
            dest = os.path.join(dir_path, filename)
            if os.path.exists(dest):
                print(f"[OK] {info['name']} - {filename} already exists. Skipping.")
                continue
                
            print(f"[+] Downloading {info['name']} from {url}...")
            success = download_file(url, dest)
            if not success:
                print(f"[!] Warning: Direct download failed for {info['name']}.")
                print(f"    Please manually download files from: {info['repo']}")

def check_status(config: Dict[str, Any]):
    paths = config.get("paths", {})
    print("\n=== Dataset Status Report ===")
    for key, info in DATASET_INFO.items():
        dir_path = paths.get(key)
        if not dir_path:
            print(f"[-] {info['name']}: Path not defined in config.")
            continue
            
        missing_files = []
        for filename in info["url_map"].keys():
            file_path = os.path.join(dir_path, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)
                
        if not missing_files:
            print(f"[OK] {info['name']}: All files present at {dir_path}")
        else:
            print(f"[MISSING] {info['name']}: Missing files {missing_files}")
            print(f"          Official Repository: {info['repo']}")

def main():
    parser = argparse.ArgumentParser(description="Dataset infrastructure downloader.")
    parser.add_argument("--config", default="config/datasets.yaml", help="Path to datasets config")
    parser.add_argument("--download", action="store_true", help="Perform HTTP downloads from official repositories")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        prepare_directories(config)
        if args.download:
            execute_downloads(config)
        check_status(config)
    except Exception as e:
        print(f"Error executing datasets downloader: {e}")

if __name__ == "__main__":
    main()
