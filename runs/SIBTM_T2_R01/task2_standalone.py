#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm==0.16.0",            # pinned: greedy determinism holds only on the same stack
#     "huggingface-hub==0.36.0",
# ]
# ///
"""Standalone SciClaimEval Subtask-2 (Claim Evidence Prediction) runner.

One-command reproduction of the shipped submission configuration, companion to
task1_standalone.py (which must sit in the same directory: the rec4 prompt, verdict
parser, model constants and dataset pins are imported from it rather than duplicated).
The canonical environment and artifact manifest is `task1_standalone.py`'s
`REPRODUCIBILITY_MANIFEST`; it is intentionally not repeated here.

Method — greedy tie-break cascade, one vLLM session, three phases:
  1. decompose  run the Subtask-1 rec4 verifier on EACH candidate image separately
                (votes=1 greedy = one decode per image); argmax of the support signal.
  2. escalate   hard tie (both images judged the same) or no valid pred -> ONE joint
                two-image call, routed by modality: figure -> compare, table -> pick.
  3. default    anything still unresolved -> evidence_id_1 (every sample gets a label).

Usage (PEP 723 inline metadata — uv builds an isolated env from the header above):
    uv run runs/SIBTM_T2_R01/task2_standalone.py --split test          # the submission run
    uv run runs/SIBTM_T2_R01/task2_standalone.py --split dev           # labeled self-check (92.9%)
    uv run runs/SIBTM_T2_R01/task2_standalone.py --split test --limit 8   # smoke test

Outputs:
    submissions/subtask2_<split>.json              official [{sample_id, pred_label}]
    submissions/subtask2_<split>.decompose.raw.jsonl   per-image provenance
    submissions/subtask2_<split>.compare.raw.jsonl     joint calls on figure ties
    submissions/subtask2_<split>.pick.raw.jsonl        joint calls on table ties

SECURITY: prompts are blind — they never reveal the twin/manipulation construction,
and the dataset's `operation` field is never even fetched.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the run directory is not a package
from task1_standalone import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL,
    MODEL_REVISION,
    SPLITS,
    _THINK_RE,
    parse_verdict,
    prepare_model_snapshot,
    REC4_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Outputs land beside this script, i.e. runs/<RUN_ID>/submissions/
ROOT = Path(__file__).resolve().parent

MAX_MODEL_LEN = 24576   # joint two-image prompts need more room than Task-1's 20480
MAX_TOKENS = 8192
VALID = ("evidence_id_1", "evidence_id_2")
RELEASE = {"dev": "dev_task2_release.json", "test": "test_task2_release.json"}

# ------------------------------------------------------------------ joint prompts
# Verbatim copies of TASKB_PICK_USER_PROMPT_TEMPLATE / TASKB_COMPARE_USER_PROMPT_TEMPLATE
# (the two joint strategies the shipped cascade routes ties to). Accuracy-critical:
# do not reflow or "improve" the wording.
TASKB_PICK_USER_PROMPT_TEMPLATE = """\
You are given TWO images, labelled image 1 and image 2. Each shows a table or figure from a \
scientific paper. Exactly one of them provides evidence that supports the claim below.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Identify the specific value, ordering, or relationship the claim depends on, read it in BOTH \
image 1 and image 2, and decide which image's data is consistent with the claim.
Reason step by step, then conclude by stating exactly: 'Therefore, the final answer is: Answer: $N' \
(without quotes), where $N is either 1 or 2."""

TASKB_COMPARE_USER_PROMPT_TEMPLATE = """\
You are given TWO images, labelled image 1 and image 2. Each shows a table or figure from a \
scientific paper. Exactly one of them provides evidence that supports the claim below.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Decide which image supports the claim using this procedure:
1. Identify the exact quantity the claim asserts — which value(s), which row/column or series, \
which comparison, ordering, or trend.
2. Read that quantity from IMAGE 1 and state it explicitly.
3. Read the same quantity from IMAGE 2 and state it explicitly.
4. Compare each reading against the claim and decide which image's data is consistent with it. \
Absence of a detail is not a conflict; only an actual contradiction of a plotted value, \
ordering, or relationship rules an image out.
Reason step by step, then conclude by stating exactly: 'Therefore, the final answer is: Answer: $N' \
(without quotes), where $N is either 1 or 2."""

# ------------------------------------------------------------------ parsing
_PICK_RE = re.compile(r"Answer:\s*\**\s*([12])")
_DANGLING_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)  # unclosed think (truncated gen)


def parse_pick(raw: str) -> tuple[int | None, str]:
    """Extract the chosen SLOT (1|2) from a joint completion.

    Only the explicit final 'Answer: N' counts. No "last image N" heuristic: models
    routinely name the REJECTED image last, so that fallback is anti-correlated with
    the true pick — a None here is an honest abstention, not a silent wrong pick.
    """
    text = _THINK_RE.sub("", raw)
    text = _DANGLING_THINK_RE.sub("", text).strip()
    ms = list(_PICK_RE.finditer(text))
    if ms:
        return int(ms[-1].group(1)), "answer_regex"
    return None, "parse_fail"


# ------------------------------------------------------------------ data
def fetch_items(split: str, revision: str | None) -> list[dict]:
    """Download (or reuse from cache) the Task-2 release JSON + images; return rows."""
    cfg = SPLITS[split]
    logger.info("Fetching task2 %s split from %s (revision %s) ...",
                split, cfg["repo"], revision or cfg["revision"])
    snap = Path(snapshot_download(
        cfg["repo"], repo_type="dataset",
        revision=revision or cfg["revision"],
        allow_patterns=[f"data/{RELEASE[split]}",
                        f"data/figures/{split}/*",
                        f"data/tables_png/{split}/*"],
    ))
    data = snap / "data"
    release = json.load(open(data / RELEASE[split]))

    items, missing = [], []
    for x in release:
        p1, p2 = data / x["evidence_id_1"], data / x["evidence_id_2"]
        missing += [str(p) for p in (p1, p2) if not p.exists()]
        items.append({
            "sample_id": x["sample_id"],
            "claim": x["claim"],
            "caption": x.get("caption", ""),
            "context": x.get("context", ""),
            "evidence_id_1": x["evidence_id_1"],
            "evidence_id_2": x["evidence_id_2"],
            "path_1": str(p1),
            "path_2": str(p2),
            "evi_type": x.get("evi_type", ""),
            "gold_label": x.get("label", ""),  # empty on test (unlabeled)
        })
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} evidence images missing, e.g.\n  "
                         + "\n  ".join(missing[:10]))
    logger.info("%d samples, all images present", len(items))
    return items


# ------------------------------------------------------------------ phases
def pair_content(path_1: str, path_2: str, it: dict, template: str) -> list[dict]:
    """Two-image user content; slot 1 = path_1, slot 2 = path_2 (caller owns slot->id)."""
    text = template.format(claim=it["claim"], caption=it["caption"] or "N/A",
                           context=it["context"] or "N/A")
    return [
        {"type": "image_url", "image_url": {"url": f"file://{path_1}"}},
        {"type": "image_url", "image_url": {"url": f"file://{path_2}"}},
        {"type": "text", "text": text},
    ]


def run_decompose(llm, items, votes, temp, max_tokens):
    """Per-image binary rec4; rank the two images by SUPPORTED-vote fraction.

    votes=1 + temp=0 (the shipped setting) = one greedy decode per image; the vote
    plumbing is kept so --votes/--temp still allow the self-consistency variant.
    """
    convs, meta = [], []
    for it in items:
        for idkey, p in [("evidence_id_1", it["path_1"]), ("evidence_id_2", it["path_2"])]:
            content = [
                {"type": "image_url", "image_url": {"url": f"file://{p}"}},
                {"type": "text", "text": REC4_USER_PROMPT_TEMPLATE.format(
                    claim=it["claim"], caption=it["caption"] or "N/A",
                    context=it["context"] or "N/A")},
            ]
            convs.append([{"role": "user", "content": content}])
            meta.append((it["sample_id"], idkey))
    # No fixed seed -> genuine per-sequence diversity when votes>1 (n>1 + fixed seed
    # can collapse to identical samples on some vLLM builds).
    sp = SamplingParams(temperature=temp, top_p=0.95, n=votes, max_tokens=max_tokens)
    outs = llm.chat(convs, sp)

    by_sample = {}
    for (sid, idkey), o in zip(meta, outs):
        verds, valid, n_trunc = [], [], 0
        for c in o.outputs:
            verdict, parse_path = parse_verdict(c.text)
            verds.append(verdict)
            if c.finish_reason == "length":
                n_trunc += 1
            # Exclude votes that never emitted a real Supported/Refuted so truncated/
            # garbled votes don't silently push toward REFUTED.
            if parse_path != "hard_default_refuted":
                valid.append(verdict)
        n_valid = len(valid)
        supp = sum(x == "SUPPORTED" for x in valid)
        d = by_sample.setdefault(sid, {})
        d[f"{idkey}_supp_frac"] = (supp / n_valid) if n_valid else None
        d[f"{idkey}_hard"] = ("SUPPORTED" if supp * 2 > n_valid else "REFUTED") if n_valid else "NA"
        d[f"{idkey}_n_valid"] = n_valid
        d[f"{idkey}_n_trunc"] = n_trunc
        d[f"{idkey}_verds"] = verds
        d[f"{idkey}_raw0"] = o.outputs[0].text

    rows = []
    for it in items:
        d = by_sample[it["sample_id"]]
        f1, f2 = d["evidence_id_1_supp_frac"], d["evidence_id_2_supp_frac"]
        h1, h2 = d["evidence_id_1_hard"], d["evidence_id_2_hard"]
        if f1 is None or f2 is None or f1 == f2:
            pred = None                       # abstain: undefined fraction or a genuine tie
        else:
            pred = "evidence_id_1" if f1 > f2 else "evidence_id_2"
        rows.append({
            "sample_id": it["sample_id"], "gold": it["gold_label"],
            "evi_type": it["evi_type"],
            "supp_frac_1": f1, "supp_frac_2": f2,
            "hard_1": h1, "hard_2": h2, "hard_tie": h1 == h2,
            "pred_id": pred,
            "n_valid_1": d["evidence_id_1_n_valid"], "n_valid_2": d["evidence_id_2_n_valid"],
            "n_trunc_1": d["evidence_id_1_n_trunc"], "n_trunc_2": d["evidence_id_2_n_trunc"],
            "verds_1": d["evidence_id_1_verds"], "verds_2": d["evidence_id_2_verds"],
            "raw0_1": d["evidence_id_1_raw0"], "raw0_2": d["evidence_id_2_raw0"],
        })
    return rows


def run_joint(llm, items, template, max_tokens):
    """One joint two-image call per item, greedy, normal image order only.

    Slot->id mapping: slot 1 = evidence_id_1, slot 2 = evidence_id_2.
    """
    convs = [[{"role": "user",
               "content": pair_content(it["path_1"], it["path_2"], it, template)}]
             for it in items]
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = llm.chat(convs, sp)

    rows = []
    for it, o in zip(items, outs):
        raw = o.outputs[0].text
        slot, parse_path = parse_pick(raw)
        rows.append({
            "sample_id": it["sample_id"], "gold": it["gold_label"],
            "evi_type": it["evi_type"],
            "normal_slot": slot,
            "normal_id": VALID[slot - 1] if slot in (1, 2) else None,
            "normal_parse": parse_path,
            "normal_trunc": o.outputs[0].finish_reason == "length",
            "raw_normal": raw,
        })
    return rows


# ------------------------------------------------------------------ cascade
def escalate(dec_row) -> bool:
    """Tie set: hard verdict tie OR no valid decompose pred (e.g. a parse-fail)."""
    return bool(dec_row.get("hard_tie")) or (dec_row.get("pred_id") not in VALID)


def cascade(items, dec, comp, pick):
    """Assemble final labels; returns (submission_rows, forced_default_count)."""
    out, forced = [], 0
    for it in items:
        r = dec[it["sample_id"]]
        if not escalate(r):
            p = r["pred_id"]                                     # decompose is confident
        else:
            fb = comp if it["evi_type"] == "figure" else pick    # tie -> modality fallback
            p = fb.get(it["sample_id"], {}).get("normal_id")
            if p not in VALID:
                p = r.get("pred_id")                             # last resort before default
        if p not in VALID:
            p = "evidence_id_1"                                  # guarantee a valid label
            forced += 1
        out.append({"sample_id": it["sample_id"], "pred_label": p})
    return out, forced


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Standalone Subtask-2 submission runner")
    ap.add_argument("--split", choices=list(SPLITS), default="test")
    ap.add_argument("--revision", default=None, help="Override the pinned dataset revision")
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test on first N items")
    ap.add_argument("--output", type=Path, default=None,
                    help="Submission JSON path (default: submissions/subtask2_<split>.json)")
    ap.add_argument("--votes", type=int, default=1,
                    help="Decompose decodes per image (1 = shipped greedy setting)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    items = fetch_items(args.split, args.revision)
    if args.limit:
        items = items[: args.limit]

    out_path = args.output or ROOT / "submissions" / f"subtask2_{args.split}.json"
    if args.limit:
        out_path = out_path.with_suffix(".smoke.json")

    def sidecar(phase):
        return out_path.with_suffix(f".{phase}.raw.jsonl")

    def write_jsonl(path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    prepare_model_snapshot()
    logger.info("Loading %s @ %s ...", MODEL, MODEL_REVISION)
    llm = LLM(
        model=MODEL,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=args.gpu_memory,
        allowed_local_media_path="/",
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )

    # phase 1: decompose all items
    logger.info("[decompose] %d items x 2 images | votes=%d temp=%g",
                len(items), args.votes, args.temp)
    dec_rows = run_decompose(llm, items, args.votes, args.temp, args.max_tokens)
    out_path.parent.mkdir(exist_ok=True)
    write_jsonl(sidecar("decompose"), dec_rows)
    dec = {r["sample_id"]: r for r in dec_rows}

    # phases 2+3: joint fallback on the tie set only, routed by modality
    fig_ties = [it for it in items if escalate(dec[it["sample_id"]]) and it["evi_type"] == "figure"]
    tab_ties = [it for it in items if escalate(dec[it["sample_id"]]) and it["evi_type"] != "figure"]
    logger.info("[escalate] %d/%d ties: figure->compare=%d table->pick=%d",
                len(fig_ties) + len(tab_ties), len(items), len(fig_ties), len(tab_ties))
    comp_rows = run_joint(llm, fig_ties, TASKB_COMPARE_USER_PROMPT_TEMPLATE, args.max_tokens) if fig_ties else []
    write_jsonl(sidecar("compare"), comp_rows)
    pick_rows = run_joint(llm, tab_ties, TASKB_PICK_USER_PROMPT_TEMPLATE, args.max_tokens) if tab_ties else []
    write_jsonl(sidecar("pick"), pick_rows)

    # assemble + write submission
    submission, forced = cascade(items, dec,
                                 {r["sample_id"]: r for r in comp_rows},
                                 {r["sample_id"]: r for r in pick_rows})
    with open(out_path, "w") as f:
        json.dump(submission, f, indent=1)

    dist = dict(Counter(r["pred_label"] for r in submission))
    print(f"\n{args.split}: n={len(submission)}  dist={dist}  forced-default={forced}")
    print(f"wrote {out_path}\n      {sidecar('decompose')}\n      {sidecar('compare')}\n"
          f"      {sidecar('pick')}")

    # self-check on labeled items (dev; test is unlabeled so this is skipped)
    gold = {it["sample_id"]: it["gold_label"] for it in items if it["gold_label"]}
    if gold:
        correct = sum(1 for r in submission if gold.get(r["sample_id"]) == r["pred_label"])
        print(f"accuracy: {correct}/{len(gold)} = {correct/len(gold):.4f}")


if __name__ == "__main__":
    main()
