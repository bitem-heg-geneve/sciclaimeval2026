# SIBTM_T2_R01 — Subtask 2, Claim Evidence Prediction

Team **SIB Text Mining**, PNG evidence track. Best of the team's Subtask-2 runs.

## Official test results

| Acc |
|--:|
| **90.5** |

Dev self-check with the released dev labels: 92.9 accuracy.

## Method — greedy tie-break cascade

No separate model: the Subtask-1 `rec4` verifier (`SIBTM_T1_R01`) is reused as the scoring
primitive. One vLLM session, three phases:

1. **decompose** — run the Subtask-1 verifier on *each* candidate image separately (one greedy
   decode per image); take the argmax of the support signal.
2. **escalate** — a hard tie (both images judged the same) or an unparsable prediction goes to
   **one** joint two-image call, routed by modality: figures use the `compare` prompt
   (read the claimed quantity out of each image, then decide), tables use the `pick` prompt.
3. **default** — anything still unresolved falls back to `evidence_id_1`, so every sample gets
   a label.

The prompts are blind: they never mention how the twin-image pairs were constructed, and the
dataset's `operation` field is never fetched.

## Run it

```bash
uv run runs/SIBTM_T2_R01/task2_standalone.py --split test          # the submission run
uv run runs/SIBTM_T2_R01/task2_standalone.py --split dev           # labeled self-check (92.9%)
uv run runs/SIBTM_T2_R01/task2_standalone.py --split test --limit 8   # smoke test
```

Writes `submissions/subtask2_<split>.json` (official format) next to the script, plus per-phase
`.decompose.raw.jsonl`, `.compare.raw.jsonl` and `.pick.raw.jsonl` provenance sidecars.

`task2_standalone.py` imports the prompt, verdict parser, model constants and dataset pins from
`task1_standalone.py` in the same directory rather than duplicating them. The copy here is
byte-identical to `runs/SIBTM_T1_R01/task1_standalone.py`
(sha256 `b56f372a8a3e47077b5db08c03b57623455baf60dace6247e38a51ca803719cb`) — each run directory
is kept independently runnable.

## Artifact

`submissions/subtask2_test.json` — the file submitted to the organizers.

```
sha256  54209a1112f86af0193891152e40103647af5d66f0d50e53594086508e2ebb5f
```

Environment and pins: see `REPRODUCIBILITY_MANIFEST` in `task1_standalone.py`.
