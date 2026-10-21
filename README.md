# SIB Text Mining — NTCIR-19 SciClaimEval submission runs

Code and submitted artifacts for the **SIB Text Mining** runs in the
[NTCIR-19 SciClaimEval](https://sciclaimeval.github.io/) shared task (PNG evidence track).

One directory per official run ID under `runs/`. Each directory is self-contained: the
single-file runner that produced the run, the submitted JSON, and a README with the method,
the exact command and the official score.

## Runs

| Run | Subtask | Method | Official test | Published here |
|---|---|---|--:|---|
| `SIBTM_T1_R01` | 1 - Claim Label Prediction | Qwen3-VL-32B-Instruct, custom `rec4` prompt, greedy | Acc **81.3** / Pair **64.1** | [`runs/SIBTM_T1_R01`](runs/SIBTM_T1_R01) |
| `SIBTM_T1_R02` | 1 - Claim Label Prediction | Baseline Qwen3-VL-32B, no adaptation | Acc 78.3 / Pair 58.3 | [`runs/SIBTM_T1_R02`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) |
| `SIBTM_T2_R03` | 1 - Claim Label Prediction | Qwen3-VL-32B + QLoRA fine-tuning | Acc 80.0 / Pair 62.0 | [`runs/SIBTM_T1_R03`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) |
| `SIBTM_T2_R01` | 2 - Claim Evidence Prediction | Per-image verifier → joint tie-break cascade | Acc **90.5** | [`runs/SIBTM_T2_R01`](runs/SIBTM_T2_R01) |
| `SIBTM_T2_R02` | 2 - Claim Evidence Prediction | Baseline Qwen3-VL-32B, no adaptation | Acc 65.3 | [`runs/SIBTM_T2_R02`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) |
| `SIBTM_T2_R03` | 2 - Claim Evidence Prediction | Qwen3-VL-32B + QLoRA fine-tuning | Acc 72.2 | [`runs/SIBTM_T2_R03`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) |
| `SIBTM_T2_R04` | 2 - Claim Evidence Prediction | Qwen3-VL-32B decider + chart/table advisors | Acc 80.1 | [`runs/SIBTM_T2_R04`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_advisors) |
| `SIBTM_T2_R05` | 2 - Claim Evidence Prediction | SIBTM_T2_R03 weights + step-by-step reasoning prompt | Acc 80.8 | [`runs/SIBTM_T2_R05`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) |

`SIBTM_T1_R01` and `SIBTM_T2_R01` are the team's best run in each subtask.

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
