# SIBTM_T1_R01 — Subtask 1, Claim Label Prediction

Team **SIB Text Mining**, PNG evidence track. Best of the team's Subtask-1 runs.

## Official test results

| P | R | Macro-F1 | Acc | Pair Acc |
|--:|--:|--:|--:|--:|
| 81.3 | 81.3 | 81.1 | **81.1** | **64.1** |

Dev self-check with the released dev labels: 83.0 accuracy, 69.0 pair accuracy.

## Method

- **Model** — `Qwen3-VL-32B-Instruct`, AWQ W8A16 quantization, served with vLLM.
- **Prompt** — `rec4`: a contradiction-only decision rule (only an actual conflict refutes;
  failing to confirm a detail does not) plus a plottability test that separates claim parts the
  image can physically show from peripheral ones. The wording is accuracy-critical and is
  reproduced verbatim in the script.
- **Decoding** — greedy (`temperature=0`), single pass (`votes=1`).
- **Input** — the PNG evidence image, the claim, and the caption/context text. No table
  extraction, no external tools, no fine-tuning.

## Run it

```bash
uv run runs/SIBTM_T1_R01/task1_standalone.py --split test          # the submission run
uv run runs/SIBTM_T1_R01/task1_standalone.py --split dev           # labeled self-check
uv run runs/SIBTM_T1_R01/task1_standalone.py --split test --limit 8   # smoke test
```

Writes `submissions/subtask1_<split>.json` (official format) next to the script, plus a
`.raw.jsonl` sidecar with the full model output per claim.

## Artifact

`submissions/subtask1_test.json` — the file submitted to the organizers.

```
sha256  d7cc9971fc1018c8d7e13ea1f853ed1cf5feab92170b178c2ae9f0a4584746f6
```

Environment and pins: see `REPRODUCIBILITY_MANIFEST` in `task1_standalone.py`
(sha256 `b56f372a8a3e47077b5db08c03b57623455baf60dace6247e38a51ca803719cb`).
