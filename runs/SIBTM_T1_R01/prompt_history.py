#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Every Subtask-1 prompt evaluated around the submitted rec4: what changed at each step, and why.

The submission runs a single prompt, rec4 (`REC4_USER_PROMPT_TEMPLATE` in
task1_standalone.py). This file records how we got there and what we tried after:
every claim-verification prompt we evaluated on the SciClaimEval dev split, plus the
localisation and tie-break-score prompts of two side experiments. Each prompt is
reproduced verbatim (the main line oldest first, then the side branches grouped by
experiment) and followed by a `Step` record:

  why      the error analysis (or goal) that motivated it
  changes  what was edited relative to its parent prompt
  result   what it scored
  verdict  what we kept, dropped, or learned

Main line   : v0 -> v1 -> baseline -> rec1 -> rec2 -> rec3 (dead end)
                                               rec2 -> rec4 (submitted) -> rec5 (rejected)
Side branch : rec2_hybrid, rec4_nei, rec4_pos_nei, rec4_nei2, localize, localize_table,
              rec4_crop, rec4_croponly, rec4_highlight, rec4_conf

Of these prompts only rec4 was submitted (SIBTM_T1_R01). The team's other two Subtask-1
runs, SIBTM_T1_R02 and SIBTM_T1_R03, come from a separate model-adaptation line and are
outside this history. Experiments that changed the model, quantization, decoding, image
resolution or serving stack but not the prompt (8B vs 30B-A3B vs 32B, bf16 vs W8A16,
max_pixels, a Thinking model, vLLM 0.24) are not listed. Nearly all reused rec2 or rec4.
The exceptions are a greedy 30B-A3B run of baseline, quoted under baseline, and a "Three
prompts per claim" run of unrecorded configuration, quoted under v1.

Conventions:
  - Dev = the released dev split, 747 claims (395 Supported / 352 Refuted): 352
    Supported/Refuted pairs plus 43 unpaired claims, all Supported. Scored against the
    released labels (revision pinned in task1_standalone.py).
  - acc = accuracy, pair = pair accuracy, P / R = precision / recall of SUPPORTED,
    FP / FN with SUPPORTED as the positive class.
  - 8B = Qwen3-VL-8B-Instruct (bf16); 30B-A3B = Qwen3-VL-30B-A3B-Instruct (W8A16);
    32B = Qwen3-VL-32B-Instruct AWQ W8A16, the submission model. Greedy = temperature 0.
  - McNemar a/b = exact two-sided test on the claims only the newer / only the older
    prompt got right.
  - Every prompt was chosen on the same dev claims it is scored on, so dev numbers are
    optimistic.

Prompt texts are byte-identical to the development repository's prompt module; v0 and
v1 were recovered from its git history. Placeholders are str.format fields ({claim},
{caption}, {context}; rec2_hybrid adds {parsed_table}, and v0 and the localize prompts
take only {claim}). From baseline on, an empty caption or context is sent as "N/A".

Usage (stdlib only; `python3` works too):
    uv run runs/SIBTM_T1_R01/prompt_history.py                  # lineage table
    uv run runs/SIBTM_T1_R01/prompt_history.py show rec4        # why / changes / result / text
    uv run runs/SIBTM_T1_R01/prompt_history.py diff rec2 rec4   # sentence-level diff
    uv run runs/SIBTM_T1_R01/prompt_history.py diff rec4        # ... against its parent
    uv run runs/SIBTM_T1_R01/prompt_history.py check            # rec4 == task1_standalone.py
"""

import argparse
import ast
import difflib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """One prompt in the history. The prose fields are dedented when printed."""

    id: str
    parent: str | None  # the prompt it was derived from (or, for baseline, replaced)
    date: str
    status: str
    headline: str  # one-line result, for the lineage table
    prompt: str
    why: str
    changes: str
    result: str
    verdict: str


HISTORY: list[Step] = []


def _system_user(system: str, user: str) -> str:
    """Render a system + user prompt pair as one text, for show and diff."""
    return f"[system]\n{system}\n\n[user]\n{user}"


# ------------------------------------------------------------------ v0
V0_SYSTEM_PROMPT = """\
You are a rigorous fact-checking assistant. You will be shown an image \
containing either a table or a chart, along with a textual claim. Your task is \
to determine whether the data in the image supports the claim.

Analyze the image carefully. Extract the relevant data points. Then evaluate \
the claim step by step.

Respond ONLY with valid JSON in this exact format:
{
  "verdict": "SUPPORTED" or "REFUTED" or "NOT_ENOUGH_INFO",
  "evidence": "The specific data from the image relevant to the claim",
  "reasoning": "Step-by-step reasoning connecting evidence to verdict"
}\
"""

V0_USER_PROMPT_TEMPLATE = 'Claim: "{claim}"'

HISTORY.append(Step(
    id="v0",
    parent=None,
    date="2026-03-12",
    status="superseded",
    headline="Qwen2.5-VL-7B: acc 59.8",
    prompt=_system_user(V0_SYSTEM_PROMPT, V0_USER_PROMPT_TEMPLATE),
    why="""
        The first working pipeline, written before any error analysis: a system prompt
        asking for a JSON verdict, and the claim alone as the user message.
    """,
    changes="""
        - First prompt. A system message asks for JSON {verdict, evidence, reasoning}
          with a three-way verdict: SUPPORTED, REFUTED or NOT_ENOUGH_INFO.
        - The user message is the image and `Claim: "<claim>"`: no caption, no context.
    """,
    result="""
        Qwen2.5-VL-7B-Instruct, greedy, 512 max tokens: acc 59.8 (447/747). Nine answers
        were NOT_ENOUGH_INFO, all counted wrong.
    """,
    verdict="""
        Replaced the same day. SciClaimEval labels are binary, so a third answer can only
        lose points. Abstention came back in July in three variants of rec4: once as a
        detector test (rec4_nei), twice as a three-way decision rule (rec4_pos_nei,
        rec4_nei2). All three were dropped.
    """,
))

# ------------------------------------------------------------------ v1
V1_SYSTEM_PROMPT = """\
You are a rigorous scientific fact-checker. You will be shown an image of a \
table or chart from a scientific paper, along with a claim about that data.

## Procedure

1. EXTRACT: Transcribe all relevant numbers, labels, and relationships from the image.
2. PARSE CLAIM: Identify the specific assertion (which values, which comparison, which trend).
3. CHECK: Compare extracted data against the claim. Look for ANY discrepancy.
4. DECIDE: If all specific assertions match the data, verdict is SUPPORTED. \
If ANY assertion is contradicted, verdict is REFUTED.

## Reasoning examples

Example A — REFUTED:
Claim: "Method B consistently outperforms Method A across all datasets"
Data shows Method B: [88.1, 92.3, 76.5] vs Method A: [87.9, 93.1, 75.8].
On dataset 2, Method A (93.1) > Method B (92.3). "Consistently" requires ALL \
datasets, so verdict is REFUTED.

Example B — REFUTED:
Claim: "The proposed method reduces error by 15%"
Data shows baseline error 0.42, proposed 0.38. Reduction = (0.42-0.38)/0.42 = 9.5%, \
not 15%. Verdict is REFUTED.

Example C — SUPPORTED:
Claim: "Model X achieves the highest F1 score on SQuAD"
Data shows Model X F1 = 94.6, all others below 94.0. Verdict is SUPPORTED.

## Output format

You MUST choose either SUPPORTED or REFUTED. Respond with JSON:
{
  "extracted_data": "All relevant values transcribed from the image",
  "claim_analysis": "What specifically the claim asserts",
  "contradictions": "Any discrepancies found between data and claim, or 'None'",
  "verdict": "SUPPORTED" or "REFUTED",
  "evidence": "The decisive data points"
}\
"""

# The caption and context lines were left out when the field was empty.
V1_USER_PROMPT_TEMPLATE = """\
Table/Figure caption: {caption}

Claim to verify: "{claim}"

Context from the paper: {context}\
"""

HISTORY.append(Step(
    id="v1",
    parent="v0",
    date="2026-03-12",
    status="superseded",
    headline="8B: acc 69.1 / pair 40.9 / macro-F1 69.0",
    prompt=_system_user(V1_SYSTEM_PROMPT, V1_USER_PROMPT_TEMPLATE),
    why="""
        Get closer to the organisers' published zero-shot result for Qwen3-VL-8B
        (SciClaimEval paper, Table 4, validation average: macro-F1 72.1, pair 46.9) by
        moving to their model and their binary label set.
    """,
    changes="""
        - Model switched to Qwen3-VL-8B-Instruct, the model the organisers report.
        - System prompt rewritten as a four-step procedure (EXTRACT, PARSE CLAIM, CHECK,
          DECIDE: "If ANY assertion is contradicted, verdict is REFUTED") plus three
          few-shot reasoning examples: a "consistently ... across all datasets" claim
          refuted by one dataset, a "reduces error by 15%" claim refuted by computing
          9.5%, a "highest F1" claim supported.
        - NOT_ENOUGH_INFO removed ("You MUST choose either SUPPORTED or REFUTED"). The
          JSON drops reasoning and gains extracted_data, claim_analysis and
          contradictions fields, enforced by guided (xgrammar) decoding.
        - The user message now carries the caption and the paper context.
        - Same step, not prompt: max tokens 512 -> 1024, an explicit image budget
          (200,704-2,007,040 pixels), and an unparsable answer now defaults to REFUTED.
    """,
    result="""
        8B, greedy, guided JSON, 1024 max tokens: acc 69.08, macro-F1 69.03, pair 40.91
        (144/352), P 71.6 / R 68.9 (FP 108, FN 123). Organisers: macro-F1 72.1, pair 46.9.
    """,
    verdict="""
        Still 3.1 F1 and 6.0 pair points short. Dropped 47 minutes later for a
        reconstruction of the organisers' zero-shot chain-of-thought setup (baseline).
        The commit that made the switch, titled "Three prompts per claim", also reports a
        run at acc 71.2 / macro-F1 71.1 / pair 44.0 (532/747, 155/352) in its message,
        but that run's configuration was not kept and it matches no surviving run. This
        system prompt, minus the few-shot block, survived in the development code as an
        unused guided-decoding option. Its DECIDE step ("If ANY assertion is
        contradicted, verdict is REFUTED") is, in hindsight, a first contradiction
        clause, though v1 still required every assertion to match before answering
        SUPPORTED, which rec1 and rec4 dropped.
    """,
))

# ------------------------------------------------------------------ baseline
USER_PROMPT_TEMPLATE = """\
Claim: {claim}
Context: {context}
Caption: {caption}

Your task is to critically evaluate the claim based on the image and the caption. \
Carefully examine whether the information in the caption truly supports the claim. \
Be skeptical and cautious: if there is any inconsistency, missing evidence, or \
ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying \
potential contradictions, lack of support, or misleading interpretations. If the \
claim is unsupported or contradicted by the caption and image, respond with \
'Refuted'. Only respond with 'Supported' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: \
$ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'. \
Think step by step before answering.\
"""

HISTORY.append(Step(
    id="baseline",
    parent="v1",
    date="2026-03-12",
    status="superseded",
    headline="8B (T=1 default, unrecorded): acc 69.7 / pair 45.2 (P 87.6 / R 49.9)",
    prompt=USER_PROMPT_TEMPLATE,
    why="""
        Approximate the organisers' evaluation protocol. The SciClaimEval paper says its
        zero-shot chain-of-thought prompting follows SciVer (Wang et al., 2025) but prints
        no prompt, so we took SciVer's own OPEN_ENDED_COT_PROMPT1
        (github.com/QDRhhhh/SciVer, utils/constant.py) along with SciVer's decoding
        settings.
    """,
    changes="""
        - Replaces v1 wholesale: no system message, no JSON, no procedure, no examples.
          One user message, the image first.
        - The text is SciVer's prompt with three edits: "respond with 'no'" ->
          "respond with 'Refuted'"; "Only respond with 'yes'" -> "Only respond with
          'Supported'"; "where $ANSWER is your final answer." -> "where $ANSWER is either
          'Supported' or 'Refuted'.".
        - The decision rule is SciVer's strict one: "if there is any inconsistency,
          missing evidence, or ambiguity, consider the claim incorrect ... Only respond
          with 'Supported' if the claim is fully and clearly supported."
        - Free-text answer, parsed from the last "Answer: Supported|Refuted" (the parser
          task1_standalone.py still uses).
        - SciVer's decoding: temperature 1.0, seed 215, 10,240 max tokens.
        - The context is always sent ("N/A" when empty). The organisers send it only to
          the 253 dev claims annotated as needing context ("use-context"), give none to
          the other 494 ("no-context"), and report the two groups separately and pooled.
    """,
    result="""
        8B, T=1.0 (the code default then; the run file does not record it): acc 69.75,
        pair 45.17 (159/352), P 87.56 / R 49.87, FP 28 / FN 198, macro-F1 68.85. It gets
        19 of the 43 unpaired claims, all labelled Supported. Against v1 the accuracy is
        statistically indistinguishable (+0.7; McNemar 125/120, p = 0.80, with different
        decoding), but the errors move almost entirely to false REFUTED. 30B-A3B greedy,
        for scale: acc 69.75, P 85.65 / R 51.39.
    """,
    verdict="""
        Development default from March to 2026-07-01 and the root of every rec* prompt.
        Strongly refutation-biased: it withholds Supported until every detail is
        positively confirmed. The schema is binary with no abstention class, so the
        consistent-but-unconfirmed band has to go to one side. Sending it to Refuted
        forfeits claims like the 43 unpaired dev claims, all labelled Supported, of which
        this prompt gets only 19.
    """,
))

# ------------------------------------------------------------------ rec1
REC1_USER_PROMPT_TEMPLATE = """\
Claim: {claim}
Context: {context}
Caption: {caption}

Your task is to evaluate the claim against the evidence in the image (primary), \
using the caption and context as support. Reason carefully and read exact values, \
rows, columns, series, and points where you can.

Decision rule:
- Respond 'Supported' if the visible evidence is consistent with the claim and \
nothing in the image, caption, or context contradicts it.
- Respond 'Refuted' ONLY if the evidence actually contradicts the claim — i.e. a \
value, direction, ranking, or relationship in the image/caption conflicts with it.

Apply these principles:
1. Absence is not contradiction. Do NOT refute merely because a referenced detail \
is not restated in the image or caption (e.g. an appendix, supplemental p-values or \
significance annotations, a citation, a caption term, or a reasonable domain \
descriptor). Missing restatement ≠ conflict.
2. Read hedged or qualitative wording charitably. Approximate or hedged quantifiers \
and qualitative words ("almost all", "usually … with a few exceptions", "roughly", \
"~", "notable", "tended to") permit exceptions; do not hold them to a strict or \
absolute threshold.
3. A cosmetic caption inconsistency (e.g. swapped color labels) does not override \
otherwise-supporting image evidence.
4. Do not reverse yourself: once you conclude the evidence supports the claim, do \
not flip to 'Refuted' on wording or terminology grounds. If the image and the \
caption conflict, trust the image's explicit labels and values.

Start by explaining your reasoning process clearly. Conclude your analysis by \
stating: 'Therefore, the final answer is: Answer: $ANSWER' (without quotes), where \
$ANSWER is either 'Supported' or 'Refuted'. Think step by step before answering.\
"""

HISTORY.append(Step(
    id="rec1",
    parent="baseline",
    date="2026-07-01",
    status="abandoned",
    headline="8B, 150-claim slice only: projected acc ~77.7, P ~80.4",
    prompt=REC1_USER_PROMPT_TEMPLATE,
    why="""
        The first error audit. Agents viewed the image and the model's reasoning for 50 of
        the baseline's 8B errors (10 FP, 40 FN). 34 of the 40 false negatives were classed
        as over-strict verification: in most, the model read the evidence correctly, then
        refused. About 10 of these were table-encoding misreads (a delta read as an
        absolute value, a two-sample cell) that the auditors expected no decision rule to
        repair (rec1 in fact flipped 5 of the 7 they named). The other 6 FN were
        perception errors. Typical triggers: a detail not restated in the image (an
        appendix, a supplemental p-value, a citation), hedged wording ("almost all",
        "notable") held to an absolute standard, a cosmetic caption mismatch, and the
        model reversing itself after conceding support.
    """,
    changes="""
        - Removed the strict rule ("Be skeptical and cautious: if there is any
          inconsistency, missing evidence, or ambiguity, consider the claim incorrect";
          "Only respond with 'Supported' if the claim is fully and clearly supported"),
          together with "If the claim is unsupported or contradicted by the caption and
          image, respond with 'Refuted'", the "focusing on identifying potential
          contradictions, lack of support, or misleading interpretations" clause, and
          "Carefully examine whether the information in the caption truly supports the
          claim".
        - The image becomes the primary evidence, supported by the caption and context;
          "read exact values, rows, columns, series, and points where you can".
        - New contradiction-based decision rule: Supported if the evidence is consistent
          and nothing contradicts it; Refuted ONLY if a value, direction, ranking or
          relationship actually conflicts.
        - Four principles, one per audited failure mode: (1) absence is not contradiction
          (appendix, supplemental p-values, citation, caption term, domain descriptor);
          (2) read hedged or qualitative wording charitably; (3) a cosmetic caption
          inconsistency (e.g. swapped colour labels) does not override the image; (4) do
          not reverse yourself on wording, and trust the image's labels over the caption.
        - Header and answer line unchanged.
    """,
    result="""
        Run only on a 150-claim stratified dev slice (75 Supported / 75 Refuted, including
        the 50 audited errors), 8B, T=1.0, against a fresh baseline run on the same slice:
        acc 74.0 vs 61.3. It fixed 30 of the fresh baseline's 47 slice FN. Of the 40
        audited FN (errors of the original full run), it answered Supported on 25 of the 34
        over-strict ones but only 1 of the 6 perception errors. But it also turned 10 of
        the fresh baseline's 64 correct REFUTED answers (15.6%) into Supported. Projected
        to the full dev split: acc ~77.7, P ~80.4 / R ~76.5, below the ~85%
        SUPPORTED-precision floor we had set. These are upper estimates: the same
        projection put rec2 at ~78.4 / P ~89.7, 2-3 points above its real full run.
    """,
    verdict="""
        Too permissive. Abandoned the same day and never run on the full split. The recall
        clauses work, but "refute only on contradiction" was stretched into "accept unless
        a contradiction is perceived". The contradiction-only rule came back in rec4.
    """,
))

# ------------------------------------------------------------------ rec2
REC2_USER_PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a figure or table.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Evaluate the claim against the evidence in the IMAGE (primary), using the caption \
and context as support. Decide SUPPORTED or REFUTED.

DECISION RULE
- SUPPORTED: the specific thing the claim asserts is positively confirmed by the \
visible evidence (or is a genuinely peripheral detail that nothing contradicts).
- REFUTED: the claim's core assertion is contradicted by the evidence, OR the claim \
asserts a specific value / ordering / comparison / causal link that the image is \
capable of showing but does NOT positively confirm.

RECALL PRINCIPLES (keep these - do not refuse on cosmetics)
1. Absence-of-restatement is not contradiction. Do NOT refute merely because a \
referenced detail is not restated, or lives in an appendix/unplotted axis. A \
peripheral, unplotted detail that nothing contradicts stays SUPPORTED.
2. Read genuinely hedged / qualitative wording charitably: "almost all", "usually", \
"typically", "~", "roughly", "tends to", approximate rounding of a stated value.
3. Cosmetic wording or caption/term paraphrase (e.g. a category renamed, a synonym, \
a figure-vs-text label variant) does NOT override supporting image evidence.
4. Do not reverse a correct reading over wording alone; trust the image over the caption.

GUARDRAILS (these OVERRIDE the recall principles - a soft word cannot dissolve an \
explicit assertion)
A. Positive-support requirement. If the claim asserts a specific ordering, ranking, \
exact number, causal link, or "A vs B" relationship that the chart actually plots, \
you must positively CONFIRM it in the image. "Nothing explicitly contradicts it" is \
NOT enough. If the asserted value/order/relation is absent or unverifiable in what \
is plotted, answer REFUTED.
B. Exact-value binding. When a claim ties a specific number to a specific row AND \
column/condition (a named window size, axis point, category, or dataset), confirm \
the number sits at that exact intersection. The right number in the wrong column (or \
a self-computed value differing by more than rounding from the stated one) is a \
CONTRADICTION, not a match.
C. Universal quantifiers. A soft adverb (broadly, generally, largely, notable) \
softens the DEGREE of an effect but does NOT weaken an explicit universal ("in all \
metrics", "every", "each", "on all datasets", "consistently", "always"). If the \
claim contains an explicit all/every/each/always and the image shows even one \
counterexample (e.g. a rival value is higher or bolded), answer REFUTED.
D. Superlatives / exclusivity. "the main", "the highest", "the most", "primarily", \
"the leading", "particularly / especially / most notably X" are COMPARATIVE, not \
hedges. The named category must be UNIQUELY at the extreme, or exhibit a stronger \
effect than the other subgroups. If any other category equals or exceeds it, answer \
REFUTED - even if the cited value itself is numerically correct.
E. Directional comparisons. For "group A outperforms group B", check the directly \
comparable pairs and the primary/default reported setting. If B >= A in the majority \
of head-to-head pairs, or in the base condition the claim's aggregate reading \
references, that is a CONTRADICTION - do not rescue it by narrowing to a favorable \
sub-setting.
F. Perception discipline before deciding:
   - For "consistently / always / at every point" claims, enumerate EVERY plotted \
tick (including minor/odd ticks) and scan for any crossing or tie between the two \
named series; a single crossing or touch refutes it.
   - Resolve every legend/axis abbreviation to a condition using the caption, and \
cross-check that mapping against any result the context already states; if your \
mapping is inconsistent with a stated result, re-read the labels before comparing.

Distinguish carefully: "a detail is merely not restated / is a paraphrase" (do NOT \
refute) versus "the claim's specific asserted value, ordering, direction, or \
superlative is contradicted OR has no positive support in what the image plots" (REFUTE).

Reason step by step, then conclude by stating: 'Therefore, the final answer is: \
Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.\
"""

HISTORY.append(Step(
    id="rec2",
    parent="rec1",
    date="2026-07-01",
    status="superseded",
    headline="8B: acc 75.2 / pair 52.8;  32B: acc 80.7 / pair 64.5",
    prompt=REC2_USER_PROMPT_TEMPLATE,
    why="""
        Audit of the 10 slice claims rec1 flipped from a correct REFUTED to Supported. Nine
        were genuine leaks. Four were a quantifier read too charitably ("broadly" allowed
        to dissolve "in all metrics"; "particularly large tables" when medium tables gain
        more). Three were a contradiction missed by perception (a curve crossing, a value
        sitting in the wrong column, an inverted legend mapping). Two were a consistency
        rule too weak to catch an inverted ranking, or a claimed average (20.80%) that the
        table does not yield.
    """,
    changes="""
        - New header: "You are verifying a scientific claim against a figure or table.";
          Claim / Caption / Context each on its own line (Caption now before Context);
          "Decide SUPPORTED or REFUTED."
        - Decision rule rewritten, and no longer contradiction-only. SUPPORTED now needs
          the specific assertion to be "positively confirmed by the visible evidence (or
          is a genuinely peripheral detail that nothing contradicts)". REFUTED now also
          covers a value / ordering / comparison / causal link the image could show but
          does "NOT positively confirm".
        - Recall principles kept but reworded. (1) keeps "not restated" and "appendix",
          adds "unplotted axis" and that a peripheral, unplotted detail stays SUPPORTED,
          but the new decision rule and Guardrail A now override it for anything the
          chart plots; rec1's concrete examples (citation, supplemental p-values, caption
          term, domain descriptor) are gone. (2) adds "typically" and rounding, and drops
          "notable" (moved into C) and "permit exceptions". (3) moves from a cosmetic
          caption inconsistency (swapped colour labels, an example now dropped) to
          cosmetic wording or term paraphrase (renamed category, synonym, label variant).
          (4) is shortened.
        - Six GUARDRAILS that OVERRIDE the recall principles, each taken from the
          auditors' fix suggestion for one or two of the nine leaks:
          A positive-support requirement ("Nothing explicitly contradicts it" is NOT
          enough); B exact-value binding at the named row and column; C a soft adverb
          cannot dissolve an explicit universal (all / every / consistently ...);
          D superlatives and "particularly X" are comparative, so the named item must be
          uniquely at the extreme; E directional comparisons decided by the majority of
          head-to-head pairs; F perception discipline (enumerate every tick for
          "consistently" claims, resolve every legend abbreviation).
        - A closing "Distinguish carefully" line. The answer instruction becomes "Reason
          step by step, then conclude by stating: ...".
    """,
    result="""
        Slice (8B, T=1.0): all 9 leaks closed (correct REFUTED turned Supported: 10/64 ->
        2/64), but rec2 fixed only 18 of the 34 audited over-strict FN (rec1: 25); 14 of
        rec1's 25 fixes survived and 4 were new. Full dev, 8B greedy: acc 75.23, pair
        52.84, P 85.47 / R 64.05 (FP 43, FN 142). The T=1.0 run it was adopted on: acc
        76.31, P 86.82 / R 65.06; against the T=1.0 baseline McNemar 100/51,
        p = 8.2e-5. 30B-A3B greedy: acc 76.04, pair 56.82. 32B greedy: acc 80.72, pair
        64.49, P 88.62 / R 72.91 (FP 37, FN 107). Moving from the 8B to the 32B with
        rec2 unchanged: McNemar 81/40, p = 2.4e-4.
    """,
    verdict="""
        Development default from 2026-07-01 to 2026-07-02: the only one of the
        recall-recovering prompts rec1-rec4 that stayed above the precision floor on the
        8B. The guardrails closed the leaks they targeted. But the positive-confirmation
        rule plus Guardrail A brought back the strict-baseline failure rec1 had removed:
        Guardrail A over-fired on uncontradicted external details, so a citation or an
        unplotted p-value became grounds to refute again (seen most clearly in the 32B
        audit). rec3 and rec4 both start from that problem.
    """,
))

# ------------------------------------------------------------------ rec3
REC3_USER_PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a figure or table.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Evaluate the claim against the evidence in the IMAGE (primary), using the caption \
and context as support. Decide SUPPORTED or REFUTED.

DECISION RULE
- SUPPORTED: the specific thing the claim asserts is positively confirmed by the \
visible evidence (or is a genuinely peripheral detail that nothing contradicts).
- REFUTED: the claim's core assertion is contradicted by the evidence, OR the claim \
asserts a specific value / ordering / comparison / causal link that the image is \
capable of showing but does NOT positively confirm.

RECALL PRINCIPLES (keep these - do not refuse on cosmetics)
1. Absence-of-restatement is not contradiction. Do NOT refute merely because a \
referenced detail is not restated, or lives in an appendix/unplotted axis. A \
peripheral, unplotted detail that nothing contradicts stays SUPPORTED.
2. Read genuinely hedged / qualitative wording charitably: "almost all", "usually", \
"typically", "~", "roughly", "tends to", approximate rounding of a stated value.
3. Cosmetic wording or caption/term paraphrase (e.g. a category renamed, a synonym, \
a figure-vs-text label variant) does NOT override supporting image evidence.
4. Do not reverse a correct reading over wording alone; trust the image over the caption.

GUARDRAILS (these OVERRIDE the recall principles, but fire ONLY within the narrowed \
scope described - a soft word cannot dissolve an explicit assertion, and a guardrail \
must not fire on something the chart cannot plot)
A. Positive-support requirement. If the claim asserts a specific ordering, ranking, \
exact number, causal link, or "A vs B" relationship that the chart actually plots, \
you must positively CONFIRM it; "nothing explicitly contradicts it" is NOT enough, \
and if the asserted value/order/relation is absent or unverifiable in what is plotted, \
answer REFUTED. Scope limits:
   - A applies only to the claim's load-bearing PLOTTED quantity. Clauses introduced \
by "since / because / despite / although / even though" and inferential connectives \
("indicating / suggesting / implying") are NOT separately verifiable: if the plotted \
comparison(s) they accompany are confirmed, do not refute on the framing clause. \
Refute such a clause only if a plotted trend actually contradicts it.
   - Legend text, in-plot annotations, and printed value labels ARE first-class \
evidence. A number shown in the legend or as a printed label counts as positive \
confirmation; refute only if the value appears nowhere in the plot or its text.
   - Positive support binds the asserted DIRECTION/relation, not qualitative \
magnitude words. Hedged or evaluative adverbs ("slightly", "little extra", "modestly", \
"extremely weak/strong") and hedged relational wording ("almost overlaps", "roughly \
equal") are non-falsifiable: confirm the direction (for "almost overlaps", overlapping \
or closely-tracking series within their common domain confirm it) and do not refute \
because your own size estimate differs from the adverb. Compare series only over their \
actually-plotted domain. For a claim that NEGATES a causal source ("the gain does NOT \
come from Y"), a within-noise with-Y vs without-Y delta CONFIRMS the negation; only a \
substantial degradation on removing Y refutes it.
B. Exact-value binding. When a claim ties a PRECISE number (specific decimals) to a \
specific row AND column/condition, confirm it sits at that exact intersection; the \
right number in the wrong column is a CONTRADICTION. Before firing, search ALL \
rows/subsets sharing the same labels (e.g. both a Development and a Test block) for \
the claimed value; refute only if it sits at no matching intersection - a caption \
emphasizing one subset does not excuse skipping the others. Round/approximate \
multipliers ("~10x", "an order of magnitude") are NOT exact assertions: compute the \
ratio in the CORRECT direction and CONFIRM when it lands within roughly +/-25% of the \
stated factor; reserve "differs by more than rounding = contradiction" for precise \
decimals at exact intersections.
C. Universal quantifiers. Fire ONLY when the claim contains an explicit universal \
marker ("in all ...", "every", "each", "on all datasets", "consistently", "always"). \
A general trend WITHOUT such a marker is judged SUPPORTED when it holds for the clear \
majority of relevant rows. When a marker is present, a counterexample counts ONLY if \
(i) it is measured against the exact baseline/reference the claim names - not a sibling \
non-reference row - and (ii) it is a member of the set that "all" ranges over (a \
weaker model INSIDE the winning method/group is not a counterexample to "beats all \
baselines"). Exclude cells the caption flags as by-construction or footnoted. A soft \
adverb still cannot dissolve a genuine explicit universal.
D. Superlatives / exclusivity. Fire ONLY when the claim asserts a SINGLE named item \
is uniquely at the extreme ("the highest / the main / the leading / the most X"): if \
any rival ties or exceeds it, REFUTE - even if the cited value is numerically correct. \
This does NOT cover: (i) a collective top-N list with respective values whose \
descending order matches the chart - verify the order, do not require each member to \
be #1; (ii) a plain comparative ("smaller / larger / lower") whose reference is \
supplied by the caption/context - evaluate against that stated reference, not as an \
intra-group ranking; (iii) exclusivity gated by a qualitative bar ("only X and Y are \
high") - a rival refutes only if it actually meets the bar; values near a detection \
limit or roughly an order of magnitude below the named group are "low".
E. Directional comparisons. For "group A outperforms group B", compare only rows \
sharing the SAME condition/setting - do NOT fabricate cross-condition pairs. A TIE in \
a comparable pair is not a counterexample, and two items tied and listed adjacently in \
a "followed by / >" ranking are consistent (adjacency is not strict >). Read each \
cited value from its own cell first. Refute only when B strictly exceeds A in the \
majority of same-condition pairs, or in the base condition the claim's aggregate \
reading references; a roughly constant large margin across all columns is \
confirmation, not contradiction.
F. Perception discipline before deciding:
   - The every-tick / single-crossing refutation applies ONLY to claims with explicit \
universal/consistency language (all/every/each/consistently/always/at every point). \
For a general comparative or converged-performance claim ("demonstrates superior \
performance to ..."), judge the dominant/aggregate trend and the final converged \
values, and treat epoch-0 / pre-convergence points and momentary ties or crossings as \
non-decisive (Guardrail D likewise does not fire on a transient crossing for such \
non-universal trend claims).
   - For claims that DO carry explicit "consistently / always / at every point" \
language, enumerate EVERY plotted tick (including minor/odd ticks) and scan for any \
crossing or tie between the two named series; a single crossing or touch refutes it.
   - Resolve every legend/axis abbreviation to a condition using the caption, and \
cross-check that mapping against any result the context already states; bind each \
curve to its legend label before reading its value; if your mapping is inconsistent \
with a stated result, re-read the labels before comparing.

Distinguish carefully: "a detail is merely not restated / is a paraphrase / is an \
unplottable framing or magnitude word" (do NOT refute) versus "the claim's specific \
asserted value, ordering, direction, or single-item superlative is contradicted OR \
has no positive support in what the image plots" (REFUTE).

Reason step by step, then conclude by stating: 'Therefore, the final answer is: \
Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.\
"""

HISTORY.append(Step(
    id="rec3",
    parent="rec2",
    date="2026-07-01",
    status="dead end",
    headline="8B (T=1): acc 76.3 / pair 55.4 (P 82.6; ties rec2 at T=1)",
    prompt=REC3_USER_PROMPT_TEMPLATE,
    why="""
        Audit of the 27 claims the baseline got right as Supported and rec2 refuted (full
        dev, 8B, T=1.0). All 27 labels were correct. 21 were a guardrail over-firing (A 9,
        C 3, D 3, E 3, B 2, F 1) and 6 a perception miss. A fired on "since / because /
        indicating" rationale clauses, on values printed in the legend, and on magnitude
        words such as "slightly" or "almost overlaps". C fired on trends without an
        explicit universal and on in-group counterexamples, D on top-N lists and plain
        comparatives, and F on an epoch-0 crossing.
    """,
    changes="""
        - Decision rule and recall principles unchanged: rec2's positive-confirmation
          fold stays.
        - Guardrail header: they now "fire ONLY within the narrowed scope described ... a
          guardrail must not fire on something the chart cannot plot".
        - A keeps its core but gains three scope limits. It binds only the load-bearing
          PLOTTED quantity, so rationale clauses are not separately verifiable. Legend
          text and printed labels are first-class evidence. Magnitude adverbs and hedged
          relations are non-falsifiable. It also gains a rule for negated causal sources.
        - B: only PRECISE numbers (specific decimals); search all matching sub-blocks
          (e.g. Development and Test) before refuting; approximate multipliers ("~10x")
          are confirmed within about +/-25%.
        - C: fires only on an explicit universal marker. A general trend that holds for
          the clear majority is SUPPORTED. A counterexample must be against the named
          baseline and inside the set "all" ranges over. Footnoted or by-construction
          cells are excluded.
        - D: only a SINGLE named item at the extreme; exempts collective top-N lists,
          plain comparatives, and exclusivity gated by a qualitative bar. Drops rec2's
          "primarily / particularly / especially / most notably X" clause (the fix for
          the "particularly large tables" leak).
        - E: compare only rows sharing the same condition; ties are not counterexamples;
          read each cited value from its own cell; a roughly constant large margin is
          confirmation; refute only when B strictly exceeds A. Drops "do not rescue it by
          narrowing to a favorable sub-setting".
        - F: the every-tick rule applies only to explicit "consistently / always" claims;
          general claims are judged on the converged trend; bind each curve to its
          legend label.
        - The closing line also exempts "an unplottable framing or magnitude word" and
          narrows the superlative case to a single item.
        - 4,225 -> 7,855 characters.
    """,
    result="""
        Full dev, 8B, T=1.0 (its only run): acc 76.31, pair 55.40, P 82.63 / R 69.87.
        Against rec2 at the same temperature, accuracy is identical (570/747; McNemar
        62/62, p = 1.0), with recall +4.8 and precision -4.2. 18 of the 27 audited
        regressions came back Supported, the count the proposal predicted, but rec2
        itself re-decoded greedily gets 17 of the same 27 right, so this does not show the
        narrowing worked. It flipped 38 of the baseline's 324 correct REFUTED answers to
        Supported (rec2: 24), where we had predicted a precision cost of only 0.5-1.5
        points.
    """,
    verdict="""
        Never adopted. rec3 moved the model toward Supported across the board (Supported
        answers 296 -> 334, FP 39 -> 58), not only on the targeted cases: narrowing
        guardrails in prose loosened behaviour globally. With one T=1.0 sample per prompt,
        which individual claims flipped is mostly sampling noise. rec4 later reused rec3's
        narrowed B-F wording in condensed form, moved its rationale-clause exemption into
        the plottability test, and kept its closing-line exemption for unplottable
        framing / magnitude words. It dropped A's detailed magnitude-adverb rule, the
        positive-confirmation decision rule rec3 had inherited from rec2, and A's
        legend-value and negated-cause rules.
    """,
))

# ------------------------------------------------------------------ rec4 (submitted)
# Identical to REC4_USER_PROMPT_TEMPLATE in task1_standalone.py (`check` verifies it).
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

HISTORY.append(Step(
    id="rec4",
    parent="rec2",
    date="2026-07-01",
    status="submitted",
    headline="32B: acc 83.0 / pair 69.0;  test (SIBTM_T1_R01): acc 81.1 / pair 64.1",
    prompt=REC4_USER_PROMPT_TEMPLATE,
    why="""
        An outside review of rec2 (Claude on the web, written without project context)
        argued that REFUTED was overloaded: rec2 refutes both a contradicted claim and a
        plottable claim the image does not positively confirm. Three agents then tested
        its hypotheses against our audit data. A per-guardrail ledger put Guardrail A at
        1 fix for 9 breaks, the largest single cause of the 27 rec2 regressions. The
        labels are binary with no abstention class, so the consistent-but-unconfirmed
        band has to go to one side, and the dev labels appear to put it on Supported
        (e.g. significance claims labelled Supported with no p-value printed). "Not
        positively confirmed -> REFUTED" therefore manufactures false negatives. Their
        synthesis concluded that rec3 had loosened B-F as well as A and lost precision,
        and proposed removing only A's positive-support fold while keeping B-F firm. As
        written, rec4 also carries rec3's narrowed B-F scopes in condensed form (majority
        trends Supported, single-item superlatives only, "strictly exceeds"), on top of a
        new decision rule and guardrails recast as contradiction detectors.
    """,
    changes="""
        1. Contradiction-based DECISION RULE. SUPPORTED when the evidence is consistent
           and nothing contradicts it. REFUTED when a value, ordering, direction, ranking
           or relationship actually CONTRADICTS the claim. Plus: "Being unable to
           positively confirm a detail is NOT grounds to refute. Only an actual conflict
           refutes. Absence of proof is not proof of the opposite."
        2. New PLOTTABILITY TEST, replacing rec2's undefined "genuinely peripheral detail"
           and its first recall principle: does THIS image physically contain the axis,
           bar, curve, cell or legend entry this part of the claim is about? No ->
           peripheral (another table or figure, an appendix, a citation, an unprinted
           p-value, a "since / because" rationale): it may not refute unless the image
           shows its opposite. Yes -> core: check it.
        3. Recall principles 4 -> 3: hedged wording (adds "about" and "never exact
           assertions", and restores rec1's "These permit exceptions"), paraphrases and
           cosmetic label variants, and trusting the image's values over the caption.
        4. GUARDRAILS -> CONTRADICTION GUARDRAILS ("... but do NOT manufacture a
           conflict"). They no longer override the recall principles.
        5. A is demoted from "positive-support requirement" to "ordering / ranking /
           comparison": refute only if the plotted order is reversed; "inability to read
           is not a contradiction". "Causal link" leaves the prompt.
        6. B, C, D, E and F take rec3's narrowed scopes in condensed form. B: PRECISE
           printed numbers only, searched in all matching sub-blocks, +/-25% for "~".
           C / D / E: explicit universal markers only, majority trends supported,
           single-item superlatives, same-condition comparisons, ties not
           counterexamples, "strictly exceeds". F: legend binding and the
           converged-trend rule. rec2's "particularly / especially / most notably X"
           clause (D) and "do not rescue it by narrowing to a favorable sub-setting" (E)
           stay out, as in rec3.
        7. New in B: do not refute on a value estimated by eye or a delta the model
           computed itself when the claim is within reading tolerance (about one minor
           gridline).
        8. New in F: for an explicit "consistently / always" claim, judge only the ticks
           that can actually be resolved, and never narrate unplotted ticks or assert "no
           crossing" by default.
        9. rec3's rationale-clause exemption for A moves into the plottability test. A's
           detailed magnitude-adverb rule, its legend-value and negated-cause rules, D's
           qualitative-bar exemption and E's constant-margin exemption are not kept.
        10. The closing line drops "OR has no positive support in what the image plots"
            and, as in rec3, puts unplottable framing / magnitude words on the
            do-not-refute side and limits the refute side to a single-item superlative
            or another value the image actually plots.
        4,225 -> 5,426 characters; header, inputs and answer line unchanged.
    """,
    result="""
        8B greedy: acc 76.31, pair 55.11, P 78.84 / R 75.44. A wash against rec2 (McNemar
        79/71, p = 0.57): on the 8B it only traded precision for recall. 32B greedy:
        acc 83.00, pair 69.03, P 85.83 / R 81.27 (FP 53, FN 74). Against rec2 on the 32B:
        acc +2.3, recall +8.4, precision -2.8, pair +4.5. It recovers 51 of rec2's 107 FN.
        McNemar 63/46, p = 0.125 (pairs 45/29, p = 0.081). Official test (SIBTM_T1_R01;
        917 claims submitted, scored population not confirmed by the organisers): acc
        81.1, macro-F1 81.1, pair 64.1.
    """,
    verdict="""
        Development default from 2026-07-02, the submitted Subtask-1 prompt, and the
        per-image verifier inside SIBTM_T2_R01. Before rec4 ran on the 32B, an audit of 40
        of the 32B's 144 rec2 errors (27 FN, 13 FP) blamed 26 on the verification rubric
        (mostly guardrails over-firing), 6 on perception, 8 on other causes and none on
        model capacity, and predicted that recall would move. On the released labels the
        gain over rec2 is not significant. rec4 is kept on pair accuracy and on that
        prediction holding, not on a significance test. Lesson: a recall-side rule pays
        only when the backbone reads the evidence correctly. On the 8B, misreading was
        the binding constraint.
    """,
))

# ------------------------------------------------------------------ rec5
_PRECISION_GUARD_BLOCK = """\
PRECISION GUARD (do not manufacture SUPPORT either)
"Nothing contradicts it" is NOT sufficient on its own. Before answering SUPPORTED, \
confirm the image actually contains evidence for the claim's MAIN assertion and that \
this evidence is consistent with the claim.
- If the claim's CENTRAL quantity, experiment, comparison, or result is absent from \
this image (you cannot point to the axis, bar, curve, cell, or row it would live \
in), the claim is NOT grounded here -> answer REFUTED. The "peripheral, may-not- \
refute" rule protects only a SUBORDINATE detail (a citation, an appendix, an \
un-plotted p-value, a "since / because / despite" clause) while the claim's main \
assertion is itself shown and consistent; it does NOT license supporting a claim \
whose core evidence is missing.
- A visible conflict is a contradiction even when it looks minor: a swapped or \
mislabeled panel/axis/legend, a wrong value, a reversed order, or a factually false \
subordinate clause. Do NOT excuse it as "cosmetic", "a typo", "a paraphrase", or "a \
general statement open to charitable interpretation". If what the image plots is the \
OPPOSITE of what the claim states, REFUTE.
- A hedge ("can", "may", "tends to") loosens the STRENGTH of an assertion whose \
evidence is present; it does not convert an unshown or contradicted assertion into a \
supported one.

"""

REC5_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "RECALL PRINCIPLES\n",
    _PRECISION_GUARD_BLOCK + "RECALL PRINCIPLES\n",
    1,
)

HISTORY.append(Step(
    id="rec5",
    parent="rec4",
    date="2026-07-02",
    status="rejected",
    headline="32B: acc 82.7 / pair 66.8 (P 89.3 / R 76.5)",
    prompt=REC5_USER_PROMPT_TEMPLATE,
    why="""
        rec4's own 32B reasoning showed over-lenience. It supported a claim whose core
        experiment the image does not show ("it simply does not show the experiment. So,
        nothing contradicts the claim. Therefore, the claim is SUPPORTED"). It called
        mislabelled panel titles "a cosmetic error, not a contradiction". It excused a
        false "despite" clause as "a general statement".
    """,
    changes="""
        - One block inserted before RECALL PRINCIPLES; nothing else changed.
        - The PRECISION GUARD says "Nothing contradicts it" is NOT sufficient on its own.
          If the claim's CENTRAL quantity, experiment, comparison or result is absent
          from the image -> REFUTED. The peripheral protection covers only SUBORDINATE
          details, and only while the main assertion is shown. A visible conflict
          (swapped labels, a wrong value, a reversed order, a false subordinate clause)
          may not be excused as cosmetic, a typo, a paraphrase or a general statement. A
          hedge loosens an assertion's strength; it does not make an unshown assertion
          supported.
        - In effect, a scoped return of rec2's Guardrail A, while rec4's "Absence of
          proof is not proof of the opposite" stays in place.
    """,
    result="""
        32B greedy: acc 82.73, pair 66.76, P 89.35 / R 76.46 (FP 36, FN 93). It removed 25
        of rec4's 53 FP and added 8. But it turned 39 of rec4's true positives into false
        negatives, 22 of them among the 51 that rec4 had recovered from rec2. Against
        rec4: McNemar 45/47, p = 0.917.
    """,
    verdict="""
        Rejected: net -2 claims and -2.3 pair points. Pushing back toward positive
        grounding gives back much of the recall rec4 recovered (recall 81.3 -> 76.5). The
        two positive-support NEI variants below also move away from the contradiction-only
        rule and also lose, but through false positives (FP 153 and 94 vs rec4's 53)
        rather than recall.
    """,
))

# ================================================================== side branches

# ------------------------------------------------------------------ rec2_hybrid
_HYBRID_TABLE_BLOCK = """\
PARSED TABLE (auto-extracted from the image; STRUCTURE AID ONLY - may contain OCR \
errors). Use it only to decide which row x column an entry occupies and which cells \
are bold (=best); read the actual digit values from the IMAGE. Notation: \
"value (+d)" = a value with its improvement over a baseline; "a / b" = two \
sub-metrics in one cell. If the parsed table is clearly malformed or disagrees with \
the image, ignore it - a table-vs-image mismatch never by itself refutes the claim.

{parsed_table}

"""

REC2_HYBRID_USER_PROMPT_TEMPLATE = REC2_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _HYBRID_TABLE_BLOCK + "Evaluate the claim against the evidence in the IMAGE (primary),",
    1,
)

HISTORY.append(Step(
    id="rec2_hybrid",
    parent="rec2",
    date="2026-07-02",
    status="side experiment",
    headline="124 curated tables: 8B 36.3 -> 48.4 (p=0.012); 32B 63.7 -> 64.5 (null)",
    prompt=REC2_HYBRID_USER_PROMPT_TEMPLATE,
    why="""
        The audit of 40 of the baseline's false negatives put roughly a quarter (~10/40)
        down to table-encoding misreads: digits read correctly, but notation, row and
        column roles or scope misunderstood (a "(+28.9)" delta read as an absolute value,
        a "37/47" cell holding two samples). The auditors expected a less strict prompt
        not to repair this class, and under rec2 (greedy) the 8B still missed 89 tables
        as false negatives.
    """,
    changes="""
        - A PARSED TABLE block inserted before "Evaluate the claim ...". It holds a
          span-preserving HTML table (rowspan / colspan, bold cells) extracted by
          PaddleOCR 3.x (TableRecognitionPipelineV2), labelled STRUCTURE AID ONLY.
          Digits are still read from the image. A notation key covers "value (+d)" and
          "a / b". A table-vs-image mismatch never by itself refutes.
        - Falls back to plain rec2 when the extraction is empty.
    """,
    result="""
        Curated set of 124 tables (89 the 8B got wrong as false negatives + 35 controls),
        greedy: 8B 36.3% -> 48.4%, p = 0.012; 32B 63.7% -> 64.5%, p = 1.0. As recorded at
        the time: the per-claim predictions were not kept, so unlike the other results here
        these cannot be re-scored.
    """,
    verdict="""
        It helped the 8B on a set built from the 8B's own table errors. On the same set
        the 32B started far higher, partly by that selection, and the injection added
        nothing measurable. Never ported to rec4, never run on the full dev split, not
        submitted.
    """,
))

# ------------------------------------------------------------------ rec4_nei
_NEI_BLOCK = """\
NOT_ENOUGH_INFO (NEI) -- a narrow LAST RESORT, applied only after every rule above.
Apply the Decision Rule, Plottability Test, Recall Principles, and Guardrails FIRST.
Almost every claim resolves to Supported or Refuted, and you should let it:
- a PERIPHERAL or unshown part resolves to SUPPORTED -- it is NEVER NEI;
- hedged or approximate wording resolves per Recall Principle 1 -- it is NEVER NEI;
- a genuine plotted conflict resolves to REFUTED.
Only if, after all of that, you STILL cannot reach a verdict may you answer NEI, and
ONLY for one of exactly two reasons:
- perception: a CORE part of the claim (one that PASSED the Plottability Test -- the
  axis, bar, curve, cell, or legend entry physically exists in THIS image) is present
  but you genuinely cannot resolve the value you need (overlapping or occluded curves
  you cannot separate, stacked-bar segments too small to read, digits too low
  resolution to disambiguate). You MUST name the specific element you cannot read.
- ambiguous: the claim genuinely admits TWO OPPOSITE verdicts because it is under
  specified against what the image shows (a referent that maps to two different
  rows/series, or a comparison with no single defensible reading), such that a careful
  reader could equally justify Supported or Refuted. You MUST state the two readings.
Do NOT use NEI because: a detail is merely unshown, peripheral, external, or in an
appendix (that is SUPPORTED); the claim is hedged or approximate (that is Recall
Principle 1); you would need outside knowledge; or the claim is simply long, hard, or
you are unsure. Plain uncertainty is NOT NEI -- NEI requires a NAMED unreadable CORE
element or a NAMED two-way ambiguity. When in doubt between NEI and a verdict, give the
verdict.
Even when you answer NEI you MUST still commit a single best guess (Supported or
Refuted) using the same contradiction-based rule: lean SUPPORTED unless you can point
to an actual conflict.

"""

_REC4_NEI_TAIL = """\
Reason step by step, then conclude with EXACTLY these three lines, each on its own \
line and nothing after them:
Verdict: $VERDICT
Reason code: $CODE
Answer: $ANSWER
where $VERDICT is 'Supported', 'Refuted', or 'NEI'; $CODE is 'perception' or \
'ambiguous' when $VERDICT is 'NEI', otherwise 'none'; and $ANSWER is your committed \
best guess and is ALWAYS either 'Supported' or 'Refuted' (when $VERDICT is 'Supported' \
or 'Refuted', $ANSWER must equal it)."""

REC4_NEI_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Reason step by step, then conclude by stating: 'Therefore, the final answer is: "
    "Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.",
    _NEI_BLOCK + _REC4_NEI_TAIL,
    1,
)

HISTORY.append(Step(
    id="rec4_nei",
    parent="rec4",
    date="2026-07-03",
    status="side experiment",
    headline="32B, 5 samples: NEI on 0.7% of claims; error AUC 0.504 vs 0.687",
    prompt=REC4_NEI_USER_PROMPT_TEMPLATE,
    why="""
        Would an explicit NOT_ENOUGH_INFO answer flag the claims the model is likely to
        get wrong better than the free signal of disagreement between samples?
    """,
    changes="""
        - rec4's answer line is replaced by an NEI block and a three-line answer
          (Verdict / Reason code / Answer). The rest of rec4 is byte-identical.
        - NEI is a last resort with exactly two reasons, each of which must be named:
          perception (a core element is present but unreadable) or ambiguous (two
          opposite readings). Peripheral, unshown and hedged parts are never NEI. Plain
          uncertainty is not NEI: "When in doubt between NEI and a verdict, give the
          verdict."
        - A forced best guess (leaning Supported unless there is a conflict) keeps every
          answer scoreable with the same parser.
    """,
    result="""
        32B, 5 samples per claim at T=0.7: NEI on 5 of 747 claims (0.7%), never a
        majority. As an error detector: AUC 0.504, against 0.687 for the vote
        disagreement of the same 5 samples.
    """,
    verdict="""
        Inert, so dropped. For an uncertainty signal, use the one the sampler gives you.
        The near-silence is partly by construction, since rec4 is engineered to be
        decisive, hence rec4_pos_nei.
    """,
))

# ------------------------------------------------------------------ rec4_pos_nei
def _pos_sub(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError("rec4_pos_nei anchor missing: %r..." % old[:70])
    return text.replace(old, new, 1)


_pos = REC4_USER_PROMPT_TEMPLATE
_pos = _pos_sub(
    _pos,
    "and context as support. Decide SUPPORTED or REFUTED.",
    "and context as support. Decide SUPPORTED, REFUTED, or NEI (not enough info).",
)
_pos = _pos_sub(
    _pos,
    "DECISION RULE (contradiction-based)\n"
    "- SUPPORTED: the visible evidence is consistent with the claim and nothing in the "
    "image, caption, or context contradicts it.\n"
    "- REFUTED: a value, ordering, direction, ranking, or relationship in the evidence "
    "actually CONTRADICTS the claim.\n"
    "- Being unable to positively confirm a detail is NOT grounds to refute. Only an "
    "actual conflict refutes. Absence of proof is not proof of the opposite.",
    "DECISION RULE (three-way, positive-support based)\n"
    "- SUPPORTED: the claim's CORE assertion -- the value, ordering, direction, ranking, "
    "or relationship it asserts -- is VISIBLE in the evidence and MATCHES. You can point "
    "to the axis, bar, curve, cell, or legend entry that shows it, and it agrees.\n"
    "- REFUTED: the visible evidence CONTRADICTS the claim -- a value, ordering, "
    "direction, ranking, or relationship in the image conflicts with it.\n"
    "- NEI (Not Enough Info): the claim's CORE assertion is something THIS image could "
    "carry, but the image NEITHER confirms it NOR contradicts it -- the value/order/"
    "relation is absent, or present but unresolvable, so you can point to NO positive "
    "support AND NO conflict. The band \"nothing here contradicts it, but nothing here "
    "shows it either\" lands HERE, not in SUPPORTED.\n"
    "- A genuinely SUBORDINATE or external part (citation, appendix, unprinted p-value / "
    "significance, \"since / because / despite\" rationale clause) does NOT need positive "
    "support and does NOT trigger NEI or REFUTED on its own -- judge the claim on its "
    "CORE. Peripheral-and-uncontradicted stays SUPPORTED.",
)
_pos = _pos_sub(
    _pos,
    "PLOTTABILITY TEST (apply to each part of the claim before judging it)\n"
    "Ask: \"Does THIS image physically contain the thing this part of the claim is about "
    "(an axis, bar, curve, cell, or legend entry)?\"\n"
    "- NO -> the part is PERIPHERAL: e.g. an appendix or a different table/figure, a "
    "citation, a p-value / confidence interval / significance not printed here, or a "
    "\"since / because / indicating / despite / although\" rationale clause. You may NOT "
    "refute on a peripheral part; refute only if the image shows the OPPOSITE of it.\n"
    "- YES -> the part is CORE: check whether the plotted value/order actually "
    "contradicts the claim.",
    "PLOTTABILITY TEST (apply to each part of the claim before judging it)\n"
    "Ask: \"Is THIS part the claim's CORE assertion, or a subordinate/external detail?\" "
    "and \"Does THIS image physically contain the thing this part is about (an axis, bar, "
    "curve, cell, or legend entry)?\"\n"
    "- SUBORDINATE / EXTERNAL part -- a citation, an appendix or a different table/figure, "
    "a p-value / confidence interval / significance not printed here, or a \"since / "
    "because / indicating / despite / although\" rationale clause. This is NOT what the "
    "claim is being verified on. Do NOT require positive support for it, and do NOT refute "
    "on it, UNLESS the image shows the OPPOSITE of it.\n"
    "- CORE part that the image CAN carry -- the value, ordering, comparison, or "
    "relationship the claim actually asserts. This MUST be positively shown:\n"
    "    * present and matches -> supports SUPPORTED;\n"
    "    * present and conflicts -> REFUTED;\n"
    "    * capable of being shown here but NOT shown, or present-but-unresolvable -> NEI "
    "(neither positively supported nor contradicted).",
)
_pos = _pos_sub(
    _pos,
    "stated value. These permit exceptions and are never exact assertions.",
    "stated value. These permit exceptions and are never exact assertions. A hedge "
    "loosens the THRESHOLD of an assertion whose evidence is PRESENT; it does NOT "
    "substitute for absent evidence -- a hedged core assertion still needs to be visibly "
    "shown to be SUPPORTED.",
)
_pos = _pos_sub(
    _pos,
    "Distinguish carefully: \"a detail is merely unshown, paraphrased, or an unplottable "
    "framing/magnitude word\" (do NOT refute) versus \"a specific value, ordering, "
    "direction, or single-item superlative that the image actually plots is contradicted\" "
    "(REFUTE).",
    "Distinguish carefully: a SUBORDINATE detail that is merely unshown, paraphrased, or "
    "an unplottable framing/magnitude word stays SUPPORTED (do NOT refute or mark NEI); a "
    "CORE value, ordering, direction, or relationship the image PLOTS and CONTRADICTS is "
    "REFUTED; a CORE assertion this image COULD show but does NOT (neither confirmed nor "
    "contradicted) is NEI.",
)
_REC4_POS_NEI_TAIL = """\
NOT_ENOUGH_INFO (NEI) reason codes -- pick exactly one when your Verdict is NEI:
- unconfirmed: the claim's CORE value/order/relation is not shown in this image (the \
element that would carry it is absent, or shows something unrelated), so there is no \
positive support and also no conflict. You MUST name the element that is missing. This \
is the PRIMARY NEI case -- the "uncontradicted but not positively confirmed" band.
- perception: the CORE element physically EXISTS in this image but you genuinely cannot \
resolve the value you need (overlapping/occluded curves, segments too small, digits too \
low resolution). You MUST name the specific element you cannot read.
- ambiguous: the claim admits TWO OPPOSITE verdicts because it is under-specified \
against what the image shows (a referent mapping to two rows/series; a comparison with \
no single defensible reading). You MUST state both readings.

NEI is NOT a dumping ground. It does NOT apply when: the CORE assertion IS visibly shown \
and matches (that is SUPPORTED, commit to it); the evidence conflicts (that is REFUTED); \
or the part is SUBORDINATE/external or the wording is hedged/approximate (that is \
SUPPORTED via the carve-out and Recall Principle 1; peripheral-uncontradicted is \
SUPPORTED, never NEI). Use NEI only for a CORE, plottable assertion that this image can \
neither confirm nor contradict.

Reason step by step, then conclude with EXACTLY these three lines, each on its own line \
and nothing after them:
Verdict: $VERDICT
Reason code: $CODE
Answer: $ANSWER
where $VERDICT is 'Supported', 'Refuted', or 'NEI'; $CODE is 'unconfirmed', 'perception', \
or 'ambiguous' when $VERDICT is 'NEI', otherwise 'none'; and $ANSWER is a forced binary \
best guess, ALWAYS 'Supported' or 'Refuted' (when $VERDICT is 'Supported' or 'Refuted', \
$ANSWER must equal it; when $VERDICT is 'NEI', lean 'Supported' unless you can point to \
an actual conflict)."""
_pos = _pos_sub(
    _pos,
    "Reason step by step, then conclude by stating: 'Therefore, the final answer is: "
    "Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.",
    _REC4_POS_NEI_TAIL,
)
REC4_POS_NEI_USER_PROMPT_TEMPLATE = _pos

HISTORY.append(Step(
    id="rec4_pos_nei",
    parent="rec4",
    date="2026-07-03",
    status="rejected",
    headline="32B: acc 71.9 (-11.1 vs rec4), FP 153; NEI on 13.3%",
    prompt=REC4_POS_NEI_USER_PROMPT_TEMPLATE,
    why="""
        Give NEI a real job. Remove rec4's "nothing contradicts -> SUPPORTED" lean so
        that the uncontradicted-but-unconfirmed band goes to NEI instead. It was expected
        to score lower on SciClaimEval; it was built to measure by how much.
    """,
    changes="""
        - "Decide SUPPORTED, REFUTED, or NEI (not enough info)."
        - DECISION RULE (three-way, positive-support based). SUPPORTED only if the core
          assertion is visible and matches ("You can point to the axis, bar, curve,
          cell, or legend entry that shows it"). "nothing here contradicts it, but
          nothing here shows it either" -> NEI.
        - The plottability test is rewritten as core vs subordinate: a core part "MUST
          be positively shown".
        - Recall principle 1: a hedge "does NOT substitute for absent evidence". The
          closing line becomes three-way.
        - New tail with three reason codes (unconfirmed, perception, ambiguous) and a
          forced binary guess. Guardrails A-F and recall principles 2-3 are
          byte-identical to rec4.
    """,
    result="""
        32B greedy, scored on the forced guess: acc 71.89, pair 44.32, FP 153 (rec4: 53),
        FN 57. NEI on 99 claims (13.3%), all "unconfirmed".
    """,
    verdict="""
        Rejected. Asked to look for support, the model stopped looking for a mismatch, and
        on this data that search is the task. On the 648 claims it answered outright it
        made 109 FP where rec4 made 38 on the same claims. In a 30-FP sample, drawn and
        classified on our internally corrected labels, 27 were read as missed
        contradictions, the model latching onto a confirming detail. The other 44 FP come
        from the forced guess leaning Supported on the 99 NEI answers.
    """,
))

# ------------------------------------------------------------------ rec4_nei2
_n2 = REC4_USER_PROMPT_TEMPLATE
_n2 = _pos_sub(
    _n2,
    "and context as support. Decide SUPPORTED or REFUTED.",
    "and context as support. Decide SUPPORTED, REFUTED, or NEI (not enough info).",
)
_n2 = _pos_sub(
    _n2,
    "DECISION RULE (contradiction-based)\n"
    "- SUPPORTED: the visible evidence is consistent with the claim and nothing in the "
    "image, caption, or context contradicts it.\n"
    "- REFUTED: a value, ordering, direction, ranking, or relationship in the evidence "
    "actually CONTRADICTS the claim.\n"
    "- Being unable to positively confirm a detail is NOT grounds to refute. Only an "
    "actual conflict refutes. Absence of proof is not proof of the opposite.",
    "DECISION RULE (contradiction-first, three-way)\n"
    "FIRST hunt for a contradiction across EVERY core part of the claim, applying the "
    "CONTRADICTION GUARDRAILS below. Finding ONE detail that matches is NOT enough to "
    "support -- a genuine conflict in ANY core part means REFUTED. Do not settle on "
    "SUPPORTED before you have checked the whole claim for conflicts.\n"
    "- REFUTED: a value, ordering, direction, ranking, or relationship in the evidence "
    "actually CONTRADICTS the claim.\n"
    "- Being unable to positively confirm a detail is NOT grounds to refute. Only an "
    "actual conflict refutes.\n"
    "- Only once you are sure NOTHING in the image contradicts the claim, split the "
    "remaining (uncontradicted) case:\n"
    "    * SUPPORTED: you can point to POSITIVE visible evidence for the claim's CORE "
    "assertion -- the value/order/relationship it asserts is actually shown and matches.\n"
    "    * NEI (Not Enough Info, reason 'unconfirmed'): nothing contradicts the claim, but "
    "its CORE assertion is NOT positively shown here (the value/order/relation is absent "
    "or unresolvable), so you cannot point to evidence FOR it. A merely SUBORDINATE or "
    "external unshown detail (a citation, an appendix, an unprinted p-value, a \"since / "
    "because / despite\" clause) does NOT trigger NEI -- that stays SUPPORTED.",
)
_n2 = _pos_sub(
    _n2,
    "Reason step by step, then conclude by stating: 'Therefore, the final answer is: "
    "Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.",
    _REC4_POS_NEI_TAIL,
)
REC4_NEI2_USER_PROMPT_TEMPLATE = _n2

HISTORY.append(Step(
    id="rec4_nei2",
    parent="rec4",
    date="2026-07-03",
    status="rejected",
    headline="32B: acc 77.5 (-5.5 vs rec4), FP 94; NEI on 9.4%",
    prompt=REC4_NEI2_USER_PROMPT_TEMPLATE,
    why="""
        Fix rec4_pos_nei, whose new errors were mostly missed contradictions. Keep rec4's
        contradiction machinery intact, make the contradiction hunt explicitly come first,
        and only then split the uncontradicted case into Supported and NEI.
    """,
    changes="""
        - Same "SUPPORTED, REFUTED, or NEI" intro and three-reason tail as rec4_pos_nei.
        - DECISION RULE (contradiction-first, three-way): "FIRST hunt for a contradiction
          across EVERY core part of the claim ... Finding ONE detail that matches is NOT
          enough". Only once nothing contradicts does the split apply: SUPPORTED needs
          positive evidence for the core assertion, otherwise NEI ("unconfirmed"). rec4's
          SUPPORTED definition and its sentence "Absence of proof is not proof of the
          opposite." are removed; "Being unable to positively confirm a detail is NOT
          grounds to refute. Only an actual conflict refutes." is kept.
        - Plottability test, recall principles, guardrails A-F and the closing line are
          byte-identical to rec4.
    """,
    result="""
        32B greedy, forced guess: acc 77.51, pair 56.25, FP 94, FN 74. NEI on 70 claims
        (9.4%).
    """,
    verdict="""
        Rejected. Putting the contradiction hunt first recovered about half of
        rec4_pos_nei's loss (McNemar 64/22 against it, p = 6.5e-6) but still fell 5.5
        points short of rec4 (McNemar 43/84, p = 3.5e-4): with a live NEI channel the
        model still misses contradictions rec4 catches. NEI and rec4's contradiction
        detection look entangled.
    """,
))

# ------------------------------------------------------------------ localize
LOCALIZE_PROMPT_TEMPLATE = """\
You are given a scientific figure and a claim about it. Output the smallest axis-aligned \
bounding box around the region of the figure that contains the evidence needed to verify the \
claim -- the specific curves, bars, cells, markers, legend entries, or axis span the claim \
refers to.

Respond with ONE line of JSON and nothing else:
{{"bbox_2d": [x1, y1, x2, y2], "found": true}}
where x1,y1 is the top-left pixel and x2,y2 the bottom-right pixel of that region. If the whole \
figure is needed, or you cannot localize a specific region, respond exactly:
{{"bbox_2d": null, "found": false}}

Claim:
{claim}"""

HISTORY.append(Step(
    id="localize",
    parent=None,
    date="2026-07-03",
    status="side experiment",
    headline="pass 1 of crop-and-zoom: a bounding box for the claim's evidence",
    prompt=LOCALIZE_PROMPT_TEMPLATE,
    why="""
        Scale, quantization, global resolution and table injection had been tried. What
        remained on the 32B included perception errors, mostly on figures. The idea: ask
        the model where the evidence is, crop that region, and verify again at high
        resolution.
    """,
    changes="""
        - Not a verifier prompt. It is a separate first call that returns one JSON line,
          {"bbox_2d": [x1, y1, x2, y2], "found": true}, or found = false when the whole
          figure is needed. Greedy, 256 tokens. The box is checked, padded and upscaled
          in code, then passed to rec4_crop or rec4_croponly.
    """,
    result="""
        On an enriched 200-claim set the box was usable for 121 of 140 figures and 52 of
        60 tables. Verification results are under rec4_crop.
    """,
    verdict="""
        Worked mechanically. Replaced by localize_table after the 200-claim set, scored at
        the time on our internally corrected labels, suggested cropping tables only (see
        rec4_crop).
    """,
))

# ------------------------------------------------------------------ localize_table
LOCALIZE_TABLE_PROMPT_TEMPLATE = """\
You are given a scientific image (a table, a chart/plot, or a multi-panel figure) and a claim \
about it. Decide where the evidence needed to verify the claim lives.

- If that evidence is in a TABLE (rows/columns/cells of numbers or text), output the smallest \
axis-aligned bounding box around the specific rows, columns, or cells the claim refers to.
- If that evidence is in a chart, plot, curve, scatter, or any other NON-tabular figure, do NOT \
localize.

Respond with ONE line of JSON and nothing else. If the evidence is in a table:
{{"bbox_2d": [x1, y1, x2, y2], "found": true}}
where x1,y1 is the top-left pixel and x2,y2 the bottom-right pixel of that table region. \
Otherwise (evidence is in a chart/plot/non-tabular figure, or you cannot localize a table \
region), respond exactly:
{{"bbox_2d": null, "found": false}}

Claim:
{claim}"""

HISTORY.append(Step(
    id="localize_table",
    parent="localize",
    date="2026-07-04",
    status="side experiment",
    headline="table gate: boxes 94.4% of tables, abstains on 93.2% of figures",
    prompt=LOCALIZE_TABLE_PROMPT_TEMPLATE,
    why="""
        Fold the "crop tables only" policy into the single localisation call. It is also
        meant to cover a table inside a multi-panel image, which a per-image modality flag
        cannot see (not measured).
    """,
    changes="""
        - Box only the rows, columns or cells the claim refers to, and only IF the
          evidence is a table. For a chart, plot or any non-tabular figure, "do NOT
          localize": found = false, and the caller keeps the full image (a no-op).
    """,
    result="""
        Full dev, 32B greedy: boxes 455 of 482 tables (94.4%) and abstains on 247 of 265
        figures (93.2%). 425 tables pass the crop checks.
    """,
    verdict="""
        An accurate gate, but the crop it gates did not pay off (rec4_crop), so it shipped
        nowhere.
    """,
))

# ------------------------------------------------------------------ rec4_crop, rec4_croponly
_CROP_LINE_A = (
    "You are given TWO images of the SAME figure: image 1 is the full figure (overall "
    "context); image 2 is a high-resolution ZOOM into the region most relevant to the claim. "
    "Read fine details -- overlapping curves, small bars, markers, legend entries, exact axis "
    "values -- from image 2; use image 1 for global structure. They show the same data; if they "
    "appear to disagree, trust the zoom for fine values.\n\n"
)
_CROP_LINE_B = (
    "This image is a high-resolution ZOOM into the region of a scientific figure most relevant "
    "to the claim.\n\n"
)
REC4_CROP_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _CROP_LINE_A + "Evaluate the claim against the evidence in the IMAGES (primary),", 1,
)
REC4_CROPONLY_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _CROP_LINE_B + "Evaluate the claim against the evidence in the IMAGE (primary),", 1,
)

HISTORY.append(Step(
    id="rec4_crop",
    parent="rec4",
    date="2026-07-03",
    status="rejected",
    headline="425 cropped tables: 24 fixed / 12 broken (p=0.065); replication p=0.54",
    prompt=REC4_CROP_USER_PROMPT_TEMPLATE,
    why="""
        Spend the pixel budget where the claim's evidence is: give the model the full
        image plus a high-resolution crop of the localised region.
    """,
    changes="""
        - One paragraph inserted before "Evaluate the claim ...": "You are given TWO
          images of the SAME figure: image 1 is the full figure (overall context); image
          2 is a high-resolution ZOOM into the region most relevant to the claim ...
          trust the zoom for fine values."
        - "the evidence in the IMAGE" -> "the evidence in the IMAGES". The rest of rec4 is
          unchanged. The crop goes in as a second image. Without a usable box, the claim
          runs on plain rec4.
    """,
    result="""
        On the enriched 200-claim set, scored at the time against our internally corrected
        labels and reweighted to dev base rates, it looked useful on tables but not on
        figures, hence localize_table. Against the released labels that set shows no
        table-specific advantage (tables 9 fixed / 4 broken, figures 17 / 10, both not
        significant). On all 482 dev tables (425 cropped), against a same-session rec4
        control: 24 fixed / 12 broken, p = 0.065. The replication gave 24 / 19, p = 0.54.
        A header-preserving stitched crop with the same prompt did no better.
    """,
    verdict="""
        Rejected: not significant, did not replicate, and doubles inference. The
        replication's rec4 control matched run 1's verdict for verdict, so the drop came
        from the crop arm itself. Across sessions, the rec4 control on the same 482 tables
        differs by 5.6-7.3% between most pairs of sessions, so effects of this size can be
        compared only within a session.
    """,
))

HISTORY.append(Step(
    id="rec4_croponly",
    parent="rec4",
    date="2026-07-03",
    status="rejected",
    headline="200-claim set, figures: crop alone 19 fixed / 29 broken (p=0.19, n.s.)",
    prompt=REC4_CROPONLY_USER_PROMPT_TEMPLATE,
    why="""
        Ablation of rec4_crop: does the zoom work on its own, without the full image?
    """,
    changes="""
        - One line inserted before "Evaluate the claim ...": "This image is a
          high-resolution ZOOM into the region of a scientific figure most relevant to
          the claim." The crop replaces the full image.
    """,
    result="""
        Enriched 200-claim set, against a separately run rec4 control on the same claims:
        figures 19 fixed / 29 broken (p = 0.19; 19 / 26 on the 121 cropped figures).
    """,
    verdict="""
        Rejected. On figures the crop alone scored lower than the full image, but against
        a separately run control and not significantly, so not conclusive: 3 of the 10
        net breaks fall on uncropped figures and are session noise. Losing the axes,
        legend and other series was the suspected cause, so the crop was kept only as a
        second image beside the full one (rec4_crop).
    """,
))

# ------------------------------------------------------------------ rec4_highlight
_HL_LINE = (
    "This image is the FULL table with a translucent highlight box (red border, yellow shading) "
    "marking the region most relevant to the claim. Read the highlighted cells carefully, but use "
    "the ENTIRE table for context -- row and column headers, and the other rows -- since the "
    "highlight only points at the evidence, it does not remove the rest.\n\n"
)
REC4_HIGHLIGHT_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _HL_LINE + "Evaluate the claim against the evidence in the IMAGE (primary),", 1,
)

HISTORY.append(Step(
    id="rec4_highlight",
    parent="rec4",
    date="2026-07-04",
    status="rejected",
    headline="408 highlighted tables: 20 fixed / 20 broken (p=1.0)",
    prompt=REC4_HIGHLIGHT_USER_PROMPT_TEMPLATE,
    why="""
        Keep the whole table but point at the evidence. Can emphasis without excision get
        what cropping fixes without what cropping breaks?
    """,
    changes="""
        - One paragraph inserted before "Evaluate the claim ...": the image is the FULL
          table with a translucent highlight box (red border, yellow shading) over the
          region most relevant to the claim; read the highlighted cells, but use the
          entire table for context. The box comes from localize_table and is drawn onto
          the image in code.
    """,
    result="""
        All dev tables, 408 highlighted, against a same-session rec4 control: 20 fixed /
        20 broken, p = 1.0.
    """,
    verdict="""
        Null. Together with a re-render of the tables at crop-equivalent resolution (no
        prompt change: 24 / 24, p = 1.0), it closed the presentation levers. Presentation
        is a dead end; the rubric is not.
    """,
))

# ------------------------------------------------------------------ rec4_conf
REC4_CONF_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Reason step by step, then conclude by stating: 'Therefore, the final answer is: "
    "Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.",
    "Reason step by step. Then conclude with these TWO lines, exactly:\n"
    "'Therefore, the final answer is: Answer: $ANSWER' (without quotes), where $ANSWER is "
    "either 'Supported' or 'Refuted'.\n"
    "'Support score: $N' (without quotes), where $N is an integer from 0 to 100 giving how "
    "strongly the visible evidence in the IMAGE supports the claim: 0 = the image clearly "
    "contradicts the claim, 50 = genuinely borderline, 100 = the image clearly and fully "
    "supports it. Use the full range and avoid defaulting to round numbers.",
)

HISTORY.append(Step(
    id="rec4_conf",
    parent="rec4",
    date="2026-07-05",
    status="side experiment",
    headline="Subtask-2 ties (5-vote run): equal scores on 45%; 92.0 vs 93.2",
    prompt=REC4_CONF_USER_PROMPT_TEMPLATE,
    why="""
        Subtask 2 runs rec4 on each of two candidate images. In the exploratory Subtask-2
        sweep, which judged each image with 5 samples at T=0.7, both images got the same
        majority verdict on 104 of the 352 dev claims. A graded per-image score might
        break these ties without a joint two-image call. Like rec4, the prompt sees one
        image and says nothing about the other.
    """,
    changes="""
        - rec4's answer line is replaced by two lines: the same "Therefore, the final
          answer is: Answer: $ANSWER", plus "Support score: $N", an integer 0-100 (0 =
          the image clearly contradicts the claim, 50 = borderline, 100 = clearly and
          fully supports it; "Use the full range and avoid defaulting to round
          numbers"). The rest of rec4 is byte-identical.
    """,
    result="""
        32B greedy, on those 104 ties: identical scores on 47 (45%), an unparsable score
        on 1, and 43 of the 56 ties it did separate were correct. Breaking ties by score
        first and sending the remaining 48 to the joint two-image call scored 92.0 on
        Subtask-2 dev (score alone: 81.5), against 93.2 for the joint call on every tie,
        both with the 5-vote first step. The shipped greedy cascade (SIBTM_T2_R01) ties on
        97 of 352 and scores 92.9; rec4_conf was not run on its ties.
    """,
    verdict="""
        Not used. SIBTM_T2_R01 breaks ties with the joint call (see task2_standalone.py).
        Judged one image at a time, the score leaves 45% of the ties unbroken, and where
        it does break them it is right less often than the joint call (43 vs 47 of 56).
    """,
))


# ------------------------------------------------------------------ CLI
STEPS = {step.id: step for step in HISTORY}

# Sentence boundary inside a prompt line: diffing sentence by sentence points at the
# edited sentence instead of flagging the whole paragraph. A list marker ("1.", "A.")
# is not a sentence and stays attached to the text that follows it.
_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")
_LIST_MARKER = re.compile(r"(?:\d+|[A-Z])\.")


def _prose(text: str) -> str:
    return textwrap.indent(textwrap.dedent(text).strip(), "  ")


def lineage() -> str:
    rows = [("id", "parent", "date", "status", "result")]
    rows += [(s.id, s.parent or "-", s.date, s.status, s.headline) for s in HISTORY]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[4]
        for row in rows
    )


def show(step: Step) -> str:
    against = f" vs {step.parent}" if step.parent else ""
    return "\n\n".join([
        f"{step.id}  (parent: {step.parent or '-'}, {step.date}, {step.status})\n"
        f"  {step.headline}",
        "WHY\n" + _prose(step.why),
        f"CHANGES{against}\n" + _prose(step.changes),
        "RESULT\n" + _prose(step.result),
        "VERDICT\n" + _prose(step.verdict),
        "PROMPT\n" + "-" * 78 + "\n" + step.prompt + "\n" + "-" * 78,
    ])


def _sentences(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        indent = line[: len(line) - len(line.lstrip())]
        parts: list[str] = []
        for part in _SENTENCE_BREAK.split(line.strip()):
            if parts and _LIST_MARKER.fullmatch(parts[-1]):
                parts[-1] += " " + part
            else:
                parts.append(part)
        lines += [indent + part for part in parts]
    return lines


def diff(old: Step, new: Step) -> str:
    return "\n".join(difflib.unified_diff(
        _sentences(old.prompt), _sentences(new.prompt),
        fromfile=old.id, tofile=new.id, n=1, lineterm="",
    ))


def check() -> bool:
    """True if task1_standalone.py ships exactly the rec4 recorded here."""
    tree = ast.parse((ROOT / "task1_standalone.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "REC4_USER_PROMPT_TEMPLATE" for target in node.targets
        ):
            return ast.literal_eval(node.value) == REC4_USER_PROMPT_TEMPLATE
    raise LookupError("REC4_USER_PROMPT_TEMPLATE not found in task1_standalone.py")


def main():
    ap = argparse.ArgumentParser(description="History of the Subtask-1 prompts behind SIBTM_T1_R01")
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("list", help="lineage table (the default)")
    p = sub.add_parser("show", help="why, changes, result, verdict and full text of one prompt")
    p.add_argument("id", choices=list(STEPS))
    p = sub.add_parser("diff", help="sentence-level diff between two prompts")
    p.add_argument("old", choices=list(STEPS))
    p.add_argument("new", nargs="?", choices=list(STEPS),
                   help="omit to compare OLD against its parent")
    sub.add_parser("check", help="verify rec4 is the prompt task1_standalone.py runs")
    args = ap.parse_args()

    if args.command == "show":
        print(show(STEPS[args.id]))
    elif args.command == "diff":
        old, new = STEPS[args.old], STEPS[args.new] if args.new else None
        if new is None:
            if old.parent is None:
                ap.error(f"{old.id} has no parent; name two prompts")
            old, new = STEPS[old.parent], old
        print(diff(old, new))
    elif args.command == "check":
        ok = check()
        print("rec4 matches task1_standalone.py" if ok
              else "MISMATCH: rec4 differs from task1_standalone.py")
        raise SystemExit(0 if ok else 1)
    else:
        print(lineage())


if __name__ == "__main__":
    main()
