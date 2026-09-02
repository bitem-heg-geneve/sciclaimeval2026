#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm==0.16.0",            # pinned: greedy determinism holds only on the same stack
#     "huggingface-hub==0.36.0",
# ]
# ///
"""Standalone SciClaimEval Subtask-1 (Claim Label Prediction) runner.

Self-contained reproduction of the shipped submission configuration — no imports
from claim_verifier, no unused prompt variants, no intermediate jsonl files:

  data   : HuggingFace release JSON + PNG evidence images, read straight from the
           hub cache (downloaded on first run; both repos are gated -> needs HF auth)
  model  : Qwen3-VL-32B-Instruct AWQ W8A16 via vLLM
  prompt : rec4 (contradiction-only decision rule + plottability test)
  decode : greedy (temperature=0), single pass (votes=1), PNG image + claim +
           caption/context text only

Usage (PEP 723 inline metadata: uv builds an isolated env from the header above —
no pyproject.toml / uv.lock / checkout needed, the file alone is enough):
    uv run runs/SIBTM_T1_R01/task1_standalone.py --split test          # the submission run
    uv run runs/SIBTM_T1_R01/task1_standalone.py --split dev           # labeled self-check
    uv run runs/SIBTM_T1_R01/task1_standalone.py --split test --limit 8   # smoke test

Outputs:
    submissions/subtask1_<split>.json    official format [{claim_id, pred_label}]
    submissions/subtask1_<split>.raw.jsonl   provenance (full model output per claim)

The canonical environment/artifact manifest shared by both standalone runners is
`REPRODUCIBILITY_MANIFEST` below.  Greedy decoding removes token sampling, but fresh
GPU serving sessions are not guaranteed to reproduce every submitted label bit-for-bit.
This runner reproduces the submitted configuration; the preserved submission JSON and
its SHA-256 digest are the exact result artifact.
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Outputs land beside this script, i.e. runs/<RUN_ID>/submissions/
ROOT = Path(__file__).resolve().parent

MODEL = "cyankiwi/Qwen3-VL-32B-Instruct-AWQ-8bit"
MODEL_REVISION = "62bac5cbb4467dd65f6de2dcb2ccc4d1f4e0d8f9"
BASE_PROCESSOR_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
BASE_PROCESSOR_REVISION = "0cfaf48183f594c314753d30a4c4974bc75f3ccb"

# Canonical manifest for BOTH inference-time submission runners.  Task 2 imports this
# file already, so the historical setup is recorded once rather than duplicated.  These
# values describe the unchanged VM used for the formal runs; code cannot enforce the
# hardware/driver fields.  The package versions are the resolved PEP-723 environments,
# while the two direct requirements remain executable pins at the top of each script.
REPRODUCIBILITY_MANIFEST = {
    "platform": {
        "os": "Ubuntu 22.04.5 LTS",
        "kernel": "6.8.0-124-generic",
        "virtualization": "KVM",
        "glibc": "2.35",
        "gpu": "NVIDIA A100 80 GB (GRID A100D-80C vGPU)",
        "nvidia_grid_driver": "570.158.01",
        "cuda_toolkit": "12.8.93",
    },
    "python": "3.12.12",
    "resolved_packages": {
        "vllm": "0.16.0",
        "torch": "2.9.1 (CUDA 12.8 build; cuDNN 9.10.2)",
        "torchvision": "0.24.1",
        "triton": "3.5.1",
        "transformers": "4.57.6",
        "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.0",
        "compressed-tensors": "0.13.0",
        "pillow": "12.3.0",
        "numpy": "2.2.6",
        "safetensors": "0.8.0",
    },
    "model": {"repo": MODEL, "revision": MODEL_REVISION},
    "base_processor": {
        "repo": BASE_PROCESSOR_MODEL,
        "revision": BASE_PROCESSOR_REVISION,
    },
    "submission_sha256": {
        "subtask1_test.json": "d7cc9971fc1018c8d7e13ea1f853ed1cf5feab92170b178c2ae9f0a4584746f6",
        "subtask2_test.json": "54209a1112f86af0193891152e40103647af5d66f0d50e53594086508e2ebb5f",
    },
}

# The quantized repository revision lacked these two multimodal processor files when the
# runs were made.  The formal-run VM used byte-identical copies from the pinned base-model
# revision.  Bootstrap and verify that historical state so a fresh cache behaves like the
# submission VM instead of depending on an undocumented manual cache edit.
PROCESSOR_FILE_SHA256 = {
    "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model_snapshot() -> Path:
    """Fetch the pinned model and restore its two submission-time processor files."""
    model_snapshot = Path(snapshot_download(MODEL, revision=MODEL_REVISION))
    processor_snapshot = Path(snapshot_download(
        BASE_PROCESSOR_MODEL,
        revision=BASE_PROCESSOR_REVISION,
        allow_patterns=list(PROCESSOR_FILE_SHA256),
    ))
    for filename, expected_digest in PROCESSOR_FILE_SHA256.items():
        source = processor_snapshot / filename
        destination = model_snapshot / filename
        if not destination.exists():
            shutil.copy2(source, destination)
        actual_digest = _sha256(destination)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Unexpected {filename} digest: {actual_digest}; expected {expected_digest}"
            )
    return model_snapshot


MAX_MODEL_LEN = 20480
MAX_TOKENS = 10240
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 2560 * 28 * 28

# Dataset repos. The dev revision is PINNED to the benchmark snapshot (a newer
# revision exists that changes at least one dev image — see docs); the test
# revision is pinned to the formal-run snapshot used for the submission.
SPLITS = {
    "dev": {
        "repo": "alabnii/sciclaimeval-shared-task",
        "revision": "73b2074477204b8a0e3a2002b19f067c5bed1392",
        "release": "dev_task1_release.json",
    },
    "test": {
        "repo": "alabnii/sciclaimeval-shared-task-test",
        "revision": "19c6e84aa1327f722db4d08c19b922257214780f",
        "release": "test_task1_release.json",
    },
}

# ------------------------------------------------------------------ rec4 prompt
# Verbatim copy of REC4_USER_PROMPT_TEMPLATE (the adopted default: contradiction-
# only decision rule + operational plottability test). Accuracy-critical: do not
# reflow or "improve" the wording.
REC4_USER_PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a figure or table.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Evaluate the claim against the evidence in the IMAGE (primary), using the caption \
and context as support. Decide SUPPORTED or REFUTED.

DECISION RULE (contradiction-based)
- SUPPORTED: the visible evidence is consistent with the claim and nothing in the \
image, caption, or context contradicts it.
- REFUTED: a value, ordering, direction, ranking, or relationship in the evidence \
actually CONTRADICTS the claim.
- Being unable to positively confirm a detail is NOT grounds to refute. Only an \
actual conflict refutes. Absence of proof is not proof of the opposite.

PLOTTABILITY TEST (apply to each part of the claim before judging it)
Ask: "Does THIS image physically contain the thing this part of the claim is about \
(an axis, bar, curve, cell, or legend entry)?"
- NO -> the part is PERIPHERAL: e.g. an appendix or a different table/figure, a \
citation, a p-value / confidence interval / significance not printed here, or a \
"since / because / indicating / despite / although" rationale clause. You may NOT \
refute on a peripheral part; refute only if the image shows the OPPOSITE of it.
- YES -> the part is CORE: check whether the plotted value/order actually \
contradicts the claim.

RECALL PRINCIPLES
1. Read hedged / approximate wording charitably: "almost all", "usually", \
"typically", "~", "roughly", "about", "tends to", and approximate rounding of a \
stated value. These permit exceptions and are never exact assertions.
2. A caption/term paraphrase, synonym, or cosmetic label variant does NOT override \
supporting image evidence.
3. Trust the image's explicit values/labels over the caption when they conflict; do \
not reverse a correct reading on wording alone.

CONTRADICTION GUARDRAILS (refute when the plotted evidence genuinely conflicts; a \
soft adverb cannot dissolve an explicit universal/superlative, but do NOT \
manufacture a conflict)
A. Ordering / ranking / comparison. When the claim asserts an order the chart plots \
("X is lowest", "A beats B", "highest on Y"), read the ACTUAL plotted order and \
REFUTE only if it is reversed or contradicted (e.g. a category the claim calls a \
failure is in fact among the highest). Do NOT refute a plotted order you simply \
cannot resolve -- inability to read is not a contradiction.
B. Exact printed values. When a claim ties a PRECISE printed number (specific \
decimals) to a specific row AND column/condition, confirm it appears at that \
intersection, searching ALL matching sub-blocks (e.g. both a Development and a Test \
section) before concluding it is absent; the right number in the wrong column is a \
CONTRADICTION. Do NOT refute based on a value you estimated by eye from a bar height \
or curve position, or on a percentage/delta you computed yourself, when the claim's \
stated value is within reading/rounding tolerance (about one minor gridline; roughly \
+/-25% for a "~" or order-of-magnitude figure). Hedged values are governed by Recall \
Principle 1, not this guardrail.
C. Explicit universals. Fire ONLY on an explicit "all / every / each / on all \
datasets / consistently / always". A general trend without such a marker is \
SUPPORTED if it holds for the clear majority of relevant rows. A counterexample \
counts only if it is measured against the exact baseline the claim names (not a \
sibling / in-group row, and not a cell the caption flags as by-construction or \
footnoted).
D. Superlatives / exclusivity. Fire ONLY when the claim asserts a SINGLE named item \
is uniquely at the extreme ("the highest / the main / the most / the leading X"): a \
tie or a higher rival refutes. Does NOT cover a collective top-N list whose stated \
descending order matches the chart, or a plain comparative whose reference is \
supplied by the caption/context.
E. Directional comparisons. Compare only rows sharing the SAME condition/setting; do \
NOT fabricate cross-condition pairs. A tie is not a counterexample, and two items \
tied and listed adjacently in a "followed by / >" ranking are consistent. Refute \
only when B strictly exceeds A in the majority of same-condition pairs, or in the \
base condition the claim's aggregate reading references.
F. Perception discipline.
   - Bind each curve/series to its legend label before reading it; resolve every \
legend/axis abbreviation via the caption and cross-check against any result the \
context already states; if your mapping is inconsistent with a stated result, \
re-read the labels before comparing.
   - For an explicit "consistently / always / at every point" claim, judge only the \
ticks you can ACTUALLY resolve on the rendered axis. Do NOT narrate ticks that are \
not plotted, and do NOT assert "no crossing" as a default -- claim a crossing or tie \
only when you can point to the specific x-location where it occurs. For a general or \
converged-performance comparison, judge the dominant/aggregate trend and the final \
values, and treat a momentary touch or pre-convergence crossing as non-decisive.

Distinguish carefully: "a detail is merely unshown, paraphrased, or an unplottable \
framing/magnitude word" (do NOT refute) versus "a specific value, ordering, \
direction, or single-item superlative that the image actually plots is contradicted" \
(REFUTE).

Reason step by step, then conclude by stating: 'Therefore, the final answer is: \
Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.\
"""

# ------------------------------------------------------------------ parsing
_ANSWER_RE = re.compile(r"Answer:\s*(Supported|Refuted)", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_verdict(raw: str) -> tuple[str, str]:
    """Parse a completion into (verdict, parse_path); verdict is SUPPORTED/REFUTED.

    1. Strip <think>...</think> blocks (Qwen3 thinking mode)
    2. Last "Answer: Supported/Refuted" occurrence wins
    3. Fallback: last bare occurrence of SUPPORTED/REFUTED; hard default REFUTED
    """
    text = _THINK_RE.sub("", raw).strip()
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1).upper(), "answer_regex"
    text_upper = text.upper()
    last_supported = text_upper.rfind("SUPPORTED")
    last_refuted = text_upper.rfind("REFUTED")
    if last_supported == -1 and last_refuted == -1:
        return "REFUTED", "hard_default_refuted"
    if last_refuted > last_supported:
        return "REFUTED", "fallback_scan"
    return "SUPPORTED", "fallback_scan"


# ------------------------------------------------------------------ data
def fetch_items(split: str, revision: str | None) -> list[dict]:
    """Download (or reuse from cache) the release JSON + images; return runner rows."""
    cfg = SPLITS[split]
    logger.info("Fetching %s split from %s (revision %s) ...",
                split, cfg["repo"], revision or cfg["revision"])
    snap = Path(snapshot_download(
        cfg["repo"], repo_type="dataset",
        revision=revision or cfg["revision"],
        allow_patterns=[f"data/{cfg['release']}",
                        f"data/figures/{split}/*",
                        f"data/tables_png/{split}/*"],
    ))
    data = snap / "data"
    release = json.load(open(data / cfg["release"]))

    items, missing = [], 0
    for x in release:
        img = data / x["evi_path"]
        if not img.exists():
            missing += 1
            logger.warning("MISSING image for %s: %s", x["claim_id"], img)
            continue
        items.append({
            "claim_id": x["claim_id"],
            "evi_type": x["evi_type"],
            "evi_path": x["evi_path"],
            "image_path": str(img),
            "claim": x["claim"],
            "caption": x.get("caption", ""),
            "context": x.get("context", ""),
            "label": x.get("label", ""),  # empty on test (unlabeled)
        })
    logger.info("%d items (%d missing images)", len(items), missing)
    return items


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Standalone Subtask-1 submission runner")
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--revision", default=None, help="Override the pinned dataset revision")
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test on first N items")
    ap.add_argument("--output", type=Path, default=None,
                    help="Submission JSON path (default: submissions/subtask1_<split>.json)")
    ap.add_argument("--gold", type=Path, default=None,
                    help="Optional corrected-gold JSONL ({claim_id, label}) to score dev "
                         "against instead of the release labels")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    args = ap.parse_args()

    items = fetch_items(args.split, args.revision)
    if args.limit:
        items = items[: args.limit]

    out_path = args.output or ROOT / "submissions" / f"subtask1_{args.split}.json"
    if args.limit:
        out_path = out_path.with_suffix(".smoke.json")
    raw_path = out_path.with_suffix(".raw.jsonl")

    prepare_model_snapshot()
    logger.info("Loading %s @ %s ...", MODEL, MODEL_REVISION)
    llm = LLM(
        model=MODEL,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_memory,
        allowed_local_media_path="/",
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )
    sampling = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)

    conversations = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"file://{it['image_path']}"}},
        {"type": "text", "text": REC4_USER_PROMPT_TEMPLATE.format(
            claim=it["claim"],
            caption=it["caption"] or "N/A",
            context=it["context"] or "N/A",
        )},
    ]}] for it in items]

    logger.info("Running %d items | rec4 greedy (temp=0, votes=1)", len(items))
    outputs = llm.chat(conversations, sampling)

    submission = []
    verdicts, parse_paths = Counter(), Counter()
    truncated = 0
    out_path.parent.mkdir(exist_ok=True)
    with open(raw_path, "w") as fo:
        for it, o in zip(items, outputs):
            raw = o.outputs[0].text
            finish = o.outputs[0].finish_reason
            verdict, parse_path = parse_verdict(raw)
            verdicts[verdict] += 1
            parse_paths[parse_path] += 1
            truncated += finish == "length"
            pred = verdict.capitalize()  # SUPPORTED -> Supported
            submission.append({"claim_id": it["claim_id"], "pred_label": pred})
            fo.write(json.dumps({
                "claim_id": it["claim_id"], "evi_type": it["evi_type"],
                "evi_path": it["evi_path"], "image_path": it["image_path"],
                "claim": it["claim"], "caption": it["caption"], "context": it["context"],
                "verdict": verdict, "parse_path": parse_path,
                "finish_reason": finish, "raw_output": raw,
            }, ensure_ascii=False) + "\n")

    with open(out_path, "w") as f:
        json.dump(submission, f, indent=1)

    print(f"\n{args.split}: n={len(submission)}  dist={dict(verdicts)}")
    print(f"parse paths: {dict(parse_paths)} | truncated: {truncated}/{len(submission)}")
    print(f"wrote {out_path}\n      {raw_path}")

    # Self-check on labeled items (dev; test is unlabeled so this is skipped).
    gold = {it["claim_id"]: it["label"] for it in items if it["label"]}
    if args.gold:
        gold = {r["claim_id"]: r["label"]
                for r in (json.loads(l) for l in open(args.gold) if l.strip())}
        print(f"scoring against corrected gold: {args.gold}")
    if gold:
        scored = [r for r in submission if r["claim_id"] in gold]
        correct = sum(1 for r in scored if gold[r["claim_id"]] == r["pred_label"])
        print(f"accuracy: {correct}/{len(scored)} = {correct/len(scored):.4f}")


if __name__ == "__main__":
    main()
