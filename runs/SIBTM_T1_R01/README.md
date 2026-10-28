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

## Prompt history

`prompt_history.py` holds, verbatim, every Subtask-1 claim-verification prompt that the
inference-time line behind this run evaluated on the dev split, plus the localisation and
tie-break-score prompts of two side experiments. The adaptation line behind SIBTM_T1_R02
and SIBTM_T1_R03 is outside it. Each prompt records what changed from its parent (where
it has one), the error analysis or goal that motivated it, and what it scored. The main line runs from the first JSON prompt through
SciVer's chain-of-thought prompt (`baseline`) to `rec1` → `rec2` → `rec4`, with `rec3`
and `rec5` as rejected steps. The NEI, table-injection, localisation/crop, highlight and
confidence-score experiments are recorded as side branches. Only the standard library is
needed:

```bash
uv run runs/SIBTM_T1_R01/prompt_history.py                  # lineage table
uv run runs/SIBTM_T1_R01/prompt_history.py show rec4        # why / changes / result / full text
uv run runs/SIBTM_T1_R01/prompt_history.py diff rec2 rec4   # sentence-level diff of two prompts
uv run runs/SIBTM_T1_R01/prompt_history.py check            # rec4 here == the one task1_standalone.py runs
```

## Artifact

`submissions/subtask1_test.json` — the file submitted to the organizers.

```
sha256  d7cc9971fc1018c8d7e13ea1f853ed1cf5feab92170b178c2ae9f0a4584746f6
```

Environment and pins: see `REPRODUCIBILITY_MANIFEST` in `task1_standalone.py`
(sha256 `b56f372a8a3e47077b5db08c03b57623455baf60dace6247e38a51ca803719cb`).
