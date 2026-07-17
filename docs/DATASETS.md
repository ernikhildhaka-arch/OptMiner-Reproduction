# Benchmark Datasets for Claim 1

This document specifies the benchmarks required to replicate the main experimental results of the Opt-Miner paper (Claim 1: Pass@1 modeling accuracy comparison).

## Directory Layout
All datasets are stored in `datasets/<dataset_name>` and processed files are output to `datasets/<dataset_name>/processed/`:
```
datasets/
├── mamo/
│   ├── mamo_easy_lp.jsonl
│   ├── mamo_complex_lp.jsonl
│   └── processed/
│       ├── easy_lp.json
│       └── complex_lp.json
├── nl4opt/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── processed/
│       ├── train.json
│       └── test.json
├── industry_or/
│   ├── dataset.json
│   └── processed/
│       └── dataset.json
├── resocratic/
│   ├── dataset.json
│   └── processed/
│       └── dataset.json
├── optmath/
│   ├── optmath_bench.json
│   └── processed/
│       └── optmath_bench.json
└── opt_miner_bench/
    └── bench_128.json
```

## Datasets Catalog

| Dataset Name | Source Repository | License | Citation | Paper Section | Expected Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NL4Opt** | [nl4opt-competition](https://github.com/nl4opt/nl4opt-competition) | CC BY 4.0 | Ramamonjison et al., 2021 | Section 5.1 | ~2.5 MB |
| **MAMO** | [mamo-benchmark](https://github.com/FreedomIntelligence/Mamo) | Apache-2.0 | Huang et al., 2024 | Section 5.1 | ~4.2 MB |
| **IndustryOR** | [IndustryOR](https://huggingface.co/datasets/CardinalOperations/IndustryOR) | MIT | Huang et al., 2025 | Section 5.1 | ~0.8 MB |
| **Resocratic** | [ReSocratic](https://github.com/yangzhch6/ReSocratic) | MIT | Yang et al., 2025 | Section 5.1 | ~1.5 MB |
| **OptMATH** | [optmath](https://github.com/optsuite/OptMATH) | Apache-2.0 | Lu et al., 2025 | Section 5.1 | ~2.2 MB |
| **Opt-Miner-Bench** | Generated | Generated | Liu et al., 2026 | Section 5.1 | ~1 MB |

## Preprocessing Details
1. **Raw Storage:** Original downloaded json/jsonl files are preserved in the root subdirectory to maintain data integrity.
2. **Standardization:** `datasets/preprocess.py` parses each dataset format and outputs standardized objects containing:
   * `problem_id`
   * `problem_description`
   * `optimal_value` / `ground_truth_formula`
3. **Location:** Standardized splits are written separately into the `processed/` folders to avoid modification of raw source datasets.
