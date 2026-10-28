# SIB Text Mining — NTCIR-19 SciClaimEval submission runs

The **SIB Text Mining** runs in the
[NTCIR-19 SciClaimEval](https://sciclaimeval.github.io/) shared task (PNG evidence track):
official results and links to the code of each run.

## Runs

Eight official runs: three for Subtask 1, five for Subtask 2. Each run ID links to its code,
either in this repository or in [mottinl/ntcir2026](https://github.com/mottinl/ntcir2026).

### Subtask 1 - Claim Label Prediction

| Run | Method | Official results |
|---|---|--:|
| [`SIBTM_T1_R01`](runs/SIBTM_T1_R01) | Qwen3-VL-32B-Instruct, custom `rec4` prompt, greedy | Acc **81.1** / Pair **64.1** |
| [`SIBTM_T1_R02`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) | Baseline Qwen3-VL-32B, no adaptation | Acc 77.9 / Pair 58.3 |
| [`SIBTM_T1_R03`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) | Qwen3-VL-32B + QLoRA fine-tuning | Acc 79.7 / Pair 62.0 |

### Subtask 2 - Claim Evidence Prediction

| Run | Method | Official results |
|---|---|--:|
| [`SIBTM_T2_R01`](runs/SIBTM_T2_R01) | Per-image verifier → joint tie-break cascade | Acc **90.5** |
| [`SIBTM_T2_R02`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_baseline) | Baseline Qwen3-VL-32B, no adaptation | Acc 65.3 |
| [`SIBTM_T2_R03`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) | Qwen3-VL-32B + QLoRA fine-tuning | Acc 72.2 |
| [`SIBTM_T2_R04`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_advisors) | Qwen3-VL-32B decider + chart/table advisors | Acc 80.1 |
| [`SIBTM_T2_R05`](https://github.com/mottinl/ntcir2026/tree/main/pipeline_sibils) | `SIBTM_T2_R03` weights + step-by-step reasoning prompt | Acc 80.8 |

Runs whose ID links into `runs/` have a self-contained directory: the runner, the submitted
JSON, and a README with the method, the exact command and the official score.
