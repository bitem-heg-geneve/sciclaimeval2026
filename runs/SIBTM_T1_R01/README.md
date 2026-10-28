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

[`prompt_history.py`](prompt_history.py) is meant to be read, top to bottom, on GitHub: it
holds, verbatim, every Subtask-1 prompt evaluated on the dev split on the way to this run,
from the first JSON prompt through SciVer's chain-of-thought prompt (`baseline`) to `rec4`
and the rejected `rec5`, plus the side experiments (abstention, table injection, crop and
zoom, a tie-break score). The team's other Subtask-1 runs (SIBTM_T1_R02, SIBTM_T1_R03) are
outside it. Its opening docstring has the results table, the story in brief and how the
prompt text evolved; then each step gives why it was written, what changed from its parent,
what it scored and what we kept, followed by the prompt itself. Search `# STEP rec4 ` for the
submitted prompt.

Running it is optional (standard library only):

```bash
uv run runs/SIBTM_T1_R01/prompt_history.py                  # lineage table
uv run runs/SIBTM_T1_R01/prompt_history.py show rec4        # one step, prompt included
uv run runs/SIBTM_T1_R01/prompt_history.py diff rec2 rec4   # sentence-level diff of two prompts
uv run runs/SIBTM_T1_R01/prompt_history.py check            # prompt hashes; rec4 == the runner's
```

## Artifact

`submissions/subtask1_test.json` — the file submitted to the organizers.

```
sha256  d7cc9971fc1018c8d7e13ea1f853ed1cf5feab92170b178c2ae9f0a4584746f6
```

Environment and pins: see `REPRODUCIBILITY_MANIFEST` in `task1_standalone.py`
(sha256 `b56f372a8a3e47077b5db08c03b57623455baf60dace6247e38a51ca803719cb`).
