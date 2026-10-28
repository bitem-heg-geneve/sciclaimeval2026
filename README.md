# SIB Text Mining — NTCIR-19 SciClaimEval submission runs

Code and submitted artifacts for the **SIB Text Mining** runs in the
[NTCIR-19 SciClaimEval](https://sciclaimeval.github.io/) shared task (PNG evidence track).

Runs published here have one directory per official run ID under `runs/`. Each directory is
self-contained: the single-file runner that produced the run, the submitted JSON, and a README
with the method, the exact command and the official score.

## Runs

Eight official runs: three for Subtask 1, five for Subtask 2. The Code column links each run
to its code, either in this repository or in
[mottinl/ntcir2026](https://github.com/mottinl/ntcir2026).

### Task 1 - Claim Label Prediction

| Run | Method | Official test | Code |
|---|---|--:|---|
| `SIBTM_T1_R01` | Qwen3-VL-32B-Instruct, custom `rec4` prompt, greedy | Acc **81.1** / Pair **64.1** | [`runs/SIBTM_T1_R01`](runs/SIBTM_T1_R01) |
| `SIBTM_T1_R02` | Baseline Qwen3-VL-32B, no adaptation | Acc 77.9 / Pair 58.3 | [ntcir2026 · `pipeline_baseline`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) |
| `SIBTM_T1_R03` | Qwen3-VL-32B + QLoRA fine-tuning | Acc 79.7 / Pair 62.0 | [ntcir2026 · `pipeline_sibils`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) |

### Task 2 - Claim Evidence Prediction

| Run | Method | Official test | Code |
|---|---|--:|---|
| `SIBTM_T2_R01` | Per-image verifier → joint tie-break cascade | Acc **90.5** | [`runs/SIBTM_T2_R01`](runs/SIBTM_T2_R01) |
| `SIBTM_T2_R02` | Baseline Qwen3-VL-32B, no adaptation | Acc 65.3 | [ntcir2026 · `pipeline_baseline`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) |
| `SIBTM_T2_R03` | Qwen3-VL-32B + QLoRA fine-tuning | Acc 72.2 | [ntcir2026 · `pipeline_sibils`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) |
| `SIBTM_T2_R04` | Qwen3-VL-32B decider + chart/table advisors | Acc 80.1 | [ntcir2026 · `pipeline_advisors`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_advisors) |
| `SIBTM_T2_R05` | `SIBTM_T2_R03` weights + step-by-step reasoning prompt | Acc 80.8 | [ntcir2026 · `pipeline_sibils`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) |

## Running a run

The runners use [PEP 723](https://peps.python.org/pep-0723/) inline dependency metadata, so
each file is enough on its own — no `pyproject.toml`, no lockfile, no environment setup:

```bash
uv run runs/SIBTM_T1_R01/task1_standalone.py --split test
uv run runs/SIBTM_T2_R01/task2_standalone.py --split test
```

Requirements:

- **GPU**: one NVIDIA A100 80 GB (the model is Qwen3-VL-32B-Instruct AWQ W8A16, served by vLLM).
- **HuggingFace auth**: both SciClaimEval dataset repos are gated — run `hf auth login` with an
  account that has been granted access. Model and dataset revisions are pinned in the scripts.
- `uv`, Python 3.12.

The full environment and artifact manifest (OS, driver, CUDA, resolved package versions, model
revision, submission digests) is `REPRODUCIBILITY_MANIFEST` at the top of
`runs/SIBTM_T1_R01/task1_standalone.py`; both runs share it.

Decoding is greedy, but a fresh GPU serving session is not guaranteed to reproduce every
submitted label bit-for-bit. The preserved submission JSON in each run directory, with the
SHA-256 recorded in the manifest, is the exact result artifact.
