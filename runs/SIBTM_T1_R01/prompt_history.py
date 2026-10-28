#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Prompt history of SIBTM_T1_R01: every Subtask-1 prompt behind this run, and why each changed.

The task (NTCIR-19 SciClaimEval, Subtask 1): given a claim, the chart or table image it is about,
the image's caption and some text from the paper, answer Supported or Refuted. Our submitted run,
SIBTM_T1_R01, sends one prompt, rec4, to Qwen3-VL-32B (REC4_USER_PROMPT_TEMPLATE in
task1_standalone.py). This file records how rec4 came about and what we tried after it.

HOW TO READ THIS FILE
  Top to bottom, as text; nothing needs to be run.
  1. This overview: results, the story in brief, how the prompt text evolved, terms.
  2. MAIN LINE: 8 steps, oldest first. To jump to one, search "# STEP <id> " (with the "# "
     and a trailing space, e.g. "# STEP rec4 ").
     Each step is a record (why / changes / result / verdict), then the prompt text as sent.
  3. SIDE EXPERIMENTS: 10 steps in 4 groups (search "GROUP").
  4. THE TOOL (list / show / diff / check), at the end. Optional: it prints what the file says.
  Short path: this overview, then the why and verdict of STEP baseline, rec2 and rec4.

MAIN LINE  (full dev split, 747 claims; acc / pair, in %)
  | step     | from     | status     | what changed                 | 8B            | 32B greedy  |
  |----------|----------|------------|------------------------------|---------------|-------------|
  | v0       | -        | superseded | JSON verdict, claim only     | 59.8 (7B)     | not run     |
  | v1       | v0       | superseded | 4-step procedure, 3 examples | 69.1 / 40.9   | not run     |
  | baseline | v1       | superseded | SciVer's strict CoT prompt   | 69.7 / 45.2 T | not run     |
  | rec1     | baseline | rejected   | refute only on conflict      | slice only    | not run     |
  | rec2     | rec1     | superseded | 6 guardrails; must confirm   | 75.2 / 52.8   | 80.7 / 64.5 |
  | rec3     | rec2     | rejected   | guardrails narrowed in prose | 76.3 / 55.4 T | not run     |
  | rec4     | rec2     | submitted  | conflict only + plottability | 76.3 / 55.1   | 83.0 / 69.0 |
  | rec5     | rec4     | rejected   | rec4 + a precision guard     | not run       | 82.7 / 66.8 |
  8B runs are greedy unless marked T (temperature 1.0, one sample); rec3 is compared with rec2's
  T=1.0 run, which also scored 76.3. v0 ran on Qwen2.5-VL-7B (accuracy only). Official test,
  rec4 (SIBTM_T1_R01): acc 81.1 / pair 64.1.

THE STORY IN BRIEF
  The organizers' zero-shot results follow SciVer's chain-of-thought prompting, so after two
  quick first attempts (v0, v1) we adopted SciVer's prompt (baseline), the root of every rec*
  prompt. It refutes whatever it cannot fully confirm: on the 8B it found only 49.9% of the
  Supported claims. An audit of its errors found that most false
  Refuted answers were correct readings of the image followed by a refusal. rec1 refuted only
  on a contradiction; on a 150-claim slice it recovered recall but let through too many false
  Supported answers. rec2 added six guardrails that closed those, but it also refuted plottable
  claims the image did not positively confirm, and its guardrail A fired on parts the image
  cannot show (citations, unprinted p-values, "since / because" clauses): the refusals came
  back. rec3 narrowed the guardrails in prose and gained nothing. rec4 went back to rec1's
  contradiction-only rule, took most of rec3's narrower guardrail wording, and added a
  plottability test: a part of the claim the image cannot show may not refute it. The
  reason: the labels are binary, and the dev labels appear to put claims that the image
  neither confirms nor contradicts on Supported. On the 8B, rec4 was a wash; on the 32B it
  gained 2.3 acc and 4.5 pair points over rec2 (not significant, p = 0.125) and was
  submitted. Every later variant lost or was null.

  Where 69.7 -> 83.0 came from (each row changes one thing; see Terms for "fixed / broken"):
  | what changes | held fixed   | from -> to       | acc          | fixed / broken | p      |
  |--------------|--------------|------------------|--------------|----------------|--------|
  | prompt       | 8B, T=1.0    | baseline -> rec2 | 69.7 -> 76.3 | 100 / 51       | 8.2e-5 |
  | decoding     | rec2, 8B     | T=1.0 -> greedy  | 76.3 -> 75.2 | 54 / 62        | 0.52   |
  | model        | rec2, greedy | 8B -> 32B        | 75.2 -> 80.7 | 81 / 40        | 2.4e-4 |
  | prompt       | 32B, greedy  | rec2 -> rec4     | 80.7 -> 83.0 | 63 / 46        | 0.125  |
  The baseline prompt was never run on the 32B.

HOW THE PROMPT TEXT EVOLVED  (main line, part by part)
  | part              | baseline | rec1          | rec2         | rec3     | rec4         | rec5   |
  |-------------------|----------|---------------|--------------|----------|--------------|--------|
  | header and inputs | SciVer's | = baseline    | new          | = rec2   | = rec2       | = rec4 |
  | decision rule     | strict   | conflict only | must confirm | = rec2   | rec1's (a)   | = rec4 |
  | plottability test | -        | -             | -            | -        | new          | = rec4 |
  | precision guard   | -        | -             | -            | -        | -            | new    |
  | recall principles | -        | 4, new        | 4, reworded  | = rec2   | 3 (b)        | = rec4 |
  | guardrails A-F    | -        | -             | new (c)      | narrowed | reworked (d) | = rec4 |
  | closing line      | -        | -             | new          | edited   | edited       | = rec4 |
  | answer sentence   | SciVer's | = baseline    | new          | = rec2   | = rec2       | = rec4 |
  (a) rec1's contradiction-only rule, plus "Absence of proof is not proof of the opposite".
  (b) rec2's first principle moves into the plottability test; "permit exceptions" returns.
  (c) the guardrails override the recall principles.
  (d) A becomes an ordering check; B-F take rec3's narrowed scopes, condensed, plus new B and
      F rules; the guardrails no longer override the principles.
  "= x" means identical to x; "-" means absent. `diff <a> <b>` prints any pair sentence by
  sentence.

SIDE EXPERIMENTS  (32B unless stated; "N / M" = fixed / broken against a rec4 control)
  | group       | step           | question                   | answer                             |
  |-------------|----------------|----------------------------|------------------------------------|
  | table text  | rec2_hybrid    | Extracted table as text?   | 8B yes (+12.1, p = 0.012), 32B no  |
  | abstention  | rec4_nei       | NEI answer flags errors?   | no: NEI on 0.7%, error AUC 0.50    |
  |             | rec4_pos_nei   | Unconfirmed claims -> NEI? | no: acc 71.9 (rec4: 83.0)          |
  |             | rec4_nei2      | Same, conflict hunt first? | no: acc 77.5                       |
  | crop / zoom | localize       | Box the evidence (helper)  | usable on 121 of 140 figures       |
  |             | rec4_crop      | Full image + zoomed crop?  | 24 / 12, p = 0.065, not replicated |
  |             | rec4_croponly  | The zoomed crop alone?     | figures 19 / 29 (n.s.)             |
  |             | localize_table | Box tables only (helper)   | boxes 94.4% of tables              |
  |             | rec4_highlight | Table + highlighted cells? | 20 / 20 (p = 1.0)                  |
  | Subtask 2   | rec4_conf      | 0-100 score breaks ties?   | no: 45% stay tied                  |

TERMS
  dev split         the released dev split: 747 claims (395 Supported / 352 Refuted), forming
                    352 Supported/Refuted pairs plus 43 unpaired claims, all Supported. Every
                    number uses the released labels (revision pinned in task1_standalone.py)
                    unless it says otherwise. A team member re-checked some results and the
                    scoring by hand, to an extent not recorded.
  acc, pair         accuracy; pair accuracy = share of the 352 pairs with both claims right.
  P, R, FP, FN      precision and recall of SUPPORTED. FP = a Refuted claim answered Supported;
                    FN = a Supported claim answered Refuted.
  8B, 30B-A3B, 32B  Qwen3-VL-8B-Instruct (bf16), Qwen3-VL-30B-A3B-Instruct (W8A16) and
                    Qwen3-VL-32B-Instruct AWQ W8A16, the submission model.
  greedy, T=1.0     temperature 0; or SciVer's temperature 1.0, seed 215, one sample.
  fixed / broken    claims a change turned from wrong to right / from right to wrong, against a
                    named reference; p is an exact two-sided McNemar test on those two counts.
  session           one launch of the model server. Greedy answers are not bit-identical across
                    sessions: across table sessions, rec4's answers on the same 482 tables
                    differ by up to 7.3% (two sessions happened to agree exactly). The side
                    experiments therefore compare arms within one session; the main-line
                    comparisons, rec2 -> rec4 included, are across sessions.
  slice             150 dev claims (75 Supported / 75 Refuted, including the 50 audited
                    baseline errors), used for the quick rec1 and rec2 comparisons.
  audit             an error analysis done by Claude agents in multi-agent Claude Code
                    workflows, usually one agent per error, shown the image, the claim, its
                    caption and context, the gold label and the model's reasoning (so not
                    blind). "Auditors" are these agents. There is no record of a human
                    re-checking their classifications.
  precision floor   about 85% SUPPORTED precision, the bar a recall-recovering prompt had to
                    clear on the 8B. The Claude agent running the experiments proposed it on
                    2026-07-01, a little below the baseline's 87.6%, with no further
                    justification recorded; it later called it "85% +/- noise, not a hard line".
  guardrails A-F    rec2's six rules: A positive support (rec4: ordering / ranking), B exact
                    values, C universals ("all", "every"), D superlatives, E directional
                    comparisons, F perception discipline.
  plottable         a part of the claim that the image physically shows (an axis, bar, curve,
                    cell or legend entry).
  NEI               NOT_ENOUGH_INFO, a third answer; SciClaimEval scores only Supported and
                    Refuted.
  corrected labels  a re-adjudication of dev labels by Claude agents acting as blind judges (no
                    gold label or model answer shown), followed by an adversarial review: 50
                    labels changed, 8 later reverted, 42 kept. No record of a human re-check.
                    Used to screen side experiments and in the decision to adopt rec4 (STEP
                    rec4). No number in this file uses them unless it says so.
  Subtask 2         pick which of two images supports a claim; SIBTM_T2_R01 runs rec4 on each.
  status            submitted; superseded (was the development default, later replaced);
                    rejected (scored, not adopted); side experiment (answered a question or
                    served as a helper, never a submission candidate).

READING THE PROMPT TEXTS
  Each prompt is a Python string, byte for byte as it was sent.
  - A line ending in a backslash continues on the next line: no line break is sent there.
    Every other line break is real (the NEI block, for instance, is sent hard-wrapped).
  - {claim}, {caption} and {context} are filled in for each claim; from baseline on, an empty
    caption or context is sent as "N/A". rec2_hybrid also takes {parsed_table}; v0 and the
    localize prompts take only {claim}. In the localize prompts, {{ and }} are sent as single
    braces; the v0 and v1 system messages are sent as written, so their braces are literal.
  - From baseline on, each claim is one user message, no system message, image before text
    (rec4_crop: the full image, then the crop; rec4_croponly: the crop only; rec4_highlight:
    the box drawn onto the image).
  - A prompt built from its parent shows only the inserted or replaced text, then one
    .replace(...) line; `show <id>` prints the assembled text. rec4_pos_nei and rec4_nei2
    change too many places for that, so they are written out in full.
  - "(sic)" marks wording that was sent as written.
  - Every prompt is identical to the development repository's; `check` verifies them against
    SHA-256 hashes recorded at the end of this file.

NOT IN THIS FILE
  - The team's other Subtask-1 runs, from its adaptation line (github.com/mottinl/ntcir2026):
    SIBTM_T1_R02, the un-adapted Qwen3-VL-32B control, and SIBTM_T1_R03, QLoRA fine-tuning.
    Also the Subtask-2 two-image prompts (see runs/SIBTM_T2_R01/task2_standalone.py).
  - Experiments that changed the model, quantization, decoding, image resolution or serving
    stack but not the prompt (bf16 vs W8A16, max_pixels, a Thinking model, vLLM 0.24), except
    where a prompt comparison needs them (8B vs 32B, the 30B-A3B runs, the table re-render
    under STEP rec4_highlight).
  - A first score reported by the commit that introduced baseline (2026-03-12, "Three prompts
    per claim": acc 71.2 / pair 44.0). Its FP and TN counts equal v1's and no results file
    survives, so it is not used.

RUNNING IT  (optional; standard library only, `python3` works too)
    uv run runs/SIBTM_T1_R01/prompt_history.py                  # lineage table
    uv run runs/SIBTM_T1_R01/prompt_history.py show rec4        # one step, prompt included
    uv run runs/SIBTM_T1_R01/prompt_history.py diff rec2 rec4   # sentence-level diff
    uv run runs/SIBTM_T1_R01/prompt_history.py diff rec4        # ... against its parent
    uv run runs/SIBTM_T1_R01/prompt_history.py check            # prompt hashes; rec4 == runner
"""

HISTORY = []  # one record per prompt, in reading order; the tool that reads them is at the end


# ##################################################################################################
# MAIN LINE: 8 steps, oldest first
# ##################################################################################################

# ==================================================================================================
# STEP v0 · main line 1 of 8 · superseded · 2026-03-12
# ==================================================================================================
HISTORY.append(dict(
    id="v0", parent=None, date="2026-03-12", status="superseded",
    headline="Qwen2.5-VL-7B greedy: acc 59.8",
    why="""
        The first working pipeline, written before any error analysis: a system prompt asking
        for a JSON verdict, with the claim alone as the user message.
    """,
    changes="""
        First prompt.
        - A system message asks for JSON {verdict, evidence, reasoning} with three possible
          verdicts: SUPPORTED, REFUTED or NOT_ENOUGH_INFO.
        - The user message is the image and `Claim: "<claim>"`, with no caption and no
          context.
    """,
    result="""
        full dev, Qwen2.5-VL-7B-Instruct greedy (512 max tokens): acc 59.8 (447/747).
        Nine answers were NOT_ENOUGH_INFO, all counted wrong.
    """,
    verdict="""
        Superseded the same day. SciClaimEval labels are binary, so a third answer can only
        lose points. Abstention came back in July in three variants of rec4 (GROUP abstention);
        all three were dropped.
    """,
    prompt=("V0_SYSTEM_PROMPT", "V0_USER_PROMPT_TEMPLATE"),
))

# The prompt as sent: a system message, then the user message.
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


# ==================================================================================================
# STEP v1 · main line 2 of 8 · superseded · from v0 · 2026-03-12
# ==================================================================================================
HISTORY.append(dict(
    id="v1", parent="v0", date="2026-03-12", status="superseded",
    headline="8B greedy: acc 69.1 / pair 40.9",
    why="""
        Get closer to the organizers' published zero-shot result for Qwen3-VL-8B (SciClaimEval
        paper, Table 4, validation average: macro-F1 72.1, pair 46.9) by moving to their model
        and to their binary label set.
    """,
    changes="""
        - Model switched to Qwen3-VL-8B-Instruct, the model the organizers report.
        - System prompt rewritten as a four-step procedure (EXTRACT, PARSE CLAIM, CHECK, DECIDE:
          "If ANY assertion is contradicted, verdict is REFUTED") plus three worked examples: a
          "consistently ... across all datasets" claim refuted by one dataset, a "reduces error
          by 15%" claim refuted by computing 9.5%, a "highest F1" claim supported.
        - NOT_ENOUGH_INFO removed ("You MUST choose either SUPPORTED or REFUTED"). The JSON
          drops reasoning and gains extracted_data, claim_analysis and contradictions, enforced
          by guided (xgrammar) decoding.
        - The user message now carries the caption and the paper text, each line left out when
          empty.
        - Non-prompt changes in the same step: max tokens 512 -> 1024, an explicit image budget
          (200,704-2,007,040 pixels), and an unparsable answer now counts as REFUTED.
    """,
    result="""
        full dev, 8B greedy (guided JSON, 1024 max tokens): acc 69.1 | pair 40.9 | macro-F1 69.0
        | P 71.6 | R 68.9 | FP 108 | FN 123.
        Organizers' Qwen3-VL-8B: macro-F1 72.1, pair 46.9.
    """,
    verdict="""
        Superseded 47 minutes later by baseline, adopted to follow the organizers' protocol
        rather than to beat v1; v1 was still 3.1 F1 and 6.0 pair points short of them. This
        system prompt, minus the examples, stayed in the development code as an unused
        guided-decoding option. Its DECIDE step made a contradiction the trigger for REFUTED,
        but it still required every assertion to match before answering SUPPORTED; rec1 and
        rec4 dropped that requirement.
    """,
    prompt=("V1_SYSTEM_PROMPT", "V1_USER_PROMPT_TEMPLATE"),
))

# The prompt as sent: a system message, then the user message.
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

# (The caption and context lines were left out when empty.)
V1_USER_PROMPT_TEMPLATE = """\
Table/Figure caption: {caption}

Claim to verify: "{claim}"

Context from the paper: {context}\
"""


# ==================================================================================================
# STEP baseline · main line 3 of 8 · superseded · replaces v1 · 2026-03-12
# ==================================================================================================
HISTORY.append(dict(
    id="baseline", parent="v1", date="2026-03-12", status="superseded",
    headline="8B T=1.0: acc 69.7 / pair 45.2 (P 87.6 / R 49.9)",
    why="""
        Approximate the organizers' evaluation protocol. The SciClaimEval paper says its
        zero-shot chain-of-thought prompting follows SciVer (Wang et al., 2025) but prints no
        prompt, so we took SciVer's own OPEN_ENDED_COT_PROMPT1 (github.com/QDRhhhh/SciVer,
        utils/constant.py) and SciVer's decoding settings.
    """,
    changes="""
        - Replaces v1 wholesale: no system message, no JSON, no procedure, no examples; one user
          message, image first.
        - The text is SciVer's prompt with three edits: "respond with 'no'" -> "respond with
          'Refuted'"; "Only respond with 'yes'" -> "Only respond with 'Supported'"; "where
          $ANSWER is your final answer." -> "where $ANSWER is either 'Supported' or 'Refuted'."
        - The decision rule is SciVer's strict one: "if there is any inconsistency, missing
          evidence, or ambiguity, consider the claim incorrect ... Only respond with 'Supported'
          if the claim is fully and clearly supported."
        - Free-text answer, parsed from the last "Answer: Supported|Refuted" (the parser
          task1_standalone.py still uses).
        - SciVer's decoding: temperature 1.0, seed 215, 10,240 max tokens.
        - The context is always sent ("N/A" when empty). The organizers send it only to the 253
          dev claims annotated as needing it, and report those and the other 494 separately and
          pooled.
    """,
    result="""
        full dev, 8B T=1.0 (the code default then; the run file does not record it): acc 69.7 |
        pair 45.2 | macro-F1 68.8 | P 87.6 | R 49.9 | FP 28 | FN 198.
        Organizers' Qwen3-VL-8B: macro-F1 72.1, pair 46.9 (their context handling differs).
        vs v1: 125 fixed / 120 broken, p = 0.80 (the decoding differs too); the errors move
        almost entirely to FN.
        full dev, 30B-A3B greedy: acc 69.7 | P 85.7 | R 51.4.
    """,
    verdict="""
        Superseded by rec2 on 2026-07-01, after serving as the development default since March;
        the root of every rec* prompt. Strongly biased toward Refuted: it withholds Supported
        until every detail is positively confirmed. The labels are binary with no abstention
        class, so a claim the image neither confirms nor contradicts has to go to one side.
        Sending it to Refuted forfeits claims like the 43 unpaired dev claims, all labeled
        Supported, of which this prompt gets 19.
    """,
    prompt="USER_PROMPT_TEMPLATE",
))

# The prompt as sent. (Named as in the development repository, where baseline has no prefix.)
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


# ==================================================================================================
# STEP rec1 · main line 4 of 8 · rejected · from baseline · 2026-07-01
# ==================================================================================================
HISTORY.append(dict(
    id="rec1", parent="baseline", date="2026-07-01", status="rejected",
    headline="8B T=1.0, 150-claim slice only: acc 74.0 (baseline 61.3)",
    why="""
        The first error audit covered 50 of the baseline's 8B errors (10 FP, 40 FN). 34 of the
        40 FN were classed as over-strict verification: in most, the model read the evidence
        correctly and then refused. About 10 of these were table-encoding misreads (a delta read
        as an absolute value, a two-sample cell) that the auditors expected no decision rule to
        repair. The other 6 FN were perception errors. Typical triggers: a detail not restated
        in the image (an appendix, a supplemental p-value, a citation), hedged wording ("almost
        all", "notable") held to an absolute standard, a cosmetic caption mismatch, and the
        model reversing itself after conceding support.
    """,
    changes="""
        - Removed the strict rule: "Be skeptical and cautious: if there is any inconsistency,
          missing evidence, or ambiguity, consider the claim incorrect", "Only respond with
          'Supported' if the claim is fully and clearly supported", "If the claim is unsupported
          or contradicted by the caption and image, respond with 'Refuted'", the "focusing on
          identifying potential contradictions, lack of support, or misleading interpretations"
          clause, and "Carefully examine whether the information in the caption truly supports
          the claim".
        - The image becomes the primary evidence, supported by the caption and context; "read
          exact values, rows, columns, series, and points where you can".
        - New contradiction-based decision rule: Supported if the evidence is consistent and
          nothing contradicts it; Refuted ONLY if a value, direction, ranking or relationship
          actually conflicts.
        - Four principles, one per audited failure mode: (1) absence is not contradiction
          (appendix, supplemental p-values, citation, caption term, domain descriptor); (2) read
          hedged or qualitative wording charitably; (3) a cosmetic caption inconsistency (e.g.
          swapped color labels) does not override the image; (4) do not reverse yourself on
          wording, and trust the image's labels over the caption.
        - Header and answer sentence unchanged.
    """,
    result="""
        slice, 8B T=1.0, vs a fresh baseline run on the same slice: acc 74.0 (baseline 61.3).
        It fixed 30 of that baseline's 47 slice FN. Of the 40 audited FN, it answered Supported
        on 25 of the 34 over-strict ones but only 1 of the 6 perception errors (and on 5 of the
        7 table-encoding misreads the auditors named). But it turned 10 of the baseline's 64
        correct Refuted answers (15.6%) into Supported.
        Projected to the full dev split: acc ~77.7 | P ~80.4 | R ~76.5, below the precision
        floor. An upper estimate: the same projection put rec2 at ~78.4 / P ~89.7, 2-3 points
        above its real full run.
    """,
    verdict="""
        Rejected the same day and never run on the full split: too permissive. The recall
        principles work, but "refute only on a contradiction" turned into "accept unless a
        contradiction is perceived". The contradiction-only rule came back in rec4.
    """,
    prompt="REC1_USER_PROMPT_TEMPLATE",
))

# The prompt as sent. Same header and answer sentence as baseline; everything between is new.
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


# ==================================================================================================
# STEP rec2 · main line 5 of 8 · superseded · from rec1 · 2026-07-01
# ==================================================================================================
HISTORY.append(dict(
    id="rec2", parent="rec1", date="2026-07-01", status="superseded",
    headline="8B greedy: acc 75.2 / pair 52.8; 32B greedy: 80.7 / 64.5",
    why="""
        The auditors looked at the 10 slice claims rec1 flipped from a correct Refuted to
        Supported. They judged nine of them genuine FP (and the tenth mislabeled). Four were a
        quantifier read too charitably ("broadly"
        allowed to dissolve "in all metrics"; "particularly large tables" when medium tables gain
        more). Three were a contradiction missed by perception (a curve crossing, a value
        sitting in the wrong column, an inverted legend mapping). Two were a consistency rule too
        weak to catch an inverted ranking, or a claimed average (20.80%) that the table does not
        yield.
    """,
    changes="""
        - New header: "You are verifying a scientific claim against a figure or table."; Claim /
          Caption / Context each on its own line (Caption now before Context); "Decide SUPPORTED
          or REFUTED."
        - Decision rule rewritten, and no longer contradiction-only. SUPPORTED needs the specific
          assertion to be "positively confirmed by the visible evidence (or is a genuinely
          peripheral detail that nothing contradicts)". REFUTED also covers a value / ordering /
          comparison / causal link the image could show but does "NOT positively confirm".
        - Recall principles kept but reworded. (1) keeps "not restated" and "appendix", and adds
          "unplotted axis" and that a peripheral, unplotted detail stays SUPPORTED, but the new
          decision rule and guardrail A now override it for anything the image plots; rec1's
          concrete examples (citation, supplemental p-values, caption term, domain descriptor)
          are gone. (2) adds "typically" and rounding, and drops "notable" (moved into C) and
          "permit exceptions". (3) moves from a cosmetic caption inconsistency to cosmetic
          wording or term paraphrase (renamed category, synonym, label variant). (4) is
          shortened.
        - Six guardrails that OVERRIDE the recall principles, each taken from the auditors' fix
          for one or two of the nine FP: A positive support ("Nothing explicitly contradicts it"
          is NOT enough); B exact values at the named row and column; C a soft adverb cannot
          dissolve an explicit universal ("all", "every", "consistently"); D superlatives and
          "particularly X" are comparative, so the named item must be uniquely at the extreme;
          E directional comparisons decided by the majority of head-to-head pairs; F perception
          discipline (enumerate every tick for "consistently" claims, resolve every legend
          abbreviation).
        - A closing "Distinguish carefully" line. The answer sentence becomes "Reason step by
          step, then conclude by stating: ...".
    """,
    result="""
        slice, 8B T=1.0: all nine FP closed (correct Refuted turned Supported: 10/64 -> 2/64),
        but it fixed only 18 of the 34 audited over-strict FN (rec1: 25); 14 of rec1's 25 fixes
        survived and 4 were new.
        | full dev                       | acc  | pair | P    | R    | FP | FN  |
        |--------------------------------|------|------|------|------|----|-----|
        | 8B greedy                      | 75.2 | 52.8 | 85.5 | 64.1 | 43 | 142 |
        | 8B T=1.0 (adopted on this run) | 76.3 | 56.0 | 86.8 | 65.1 | 39 | 138 |
        | 30B-A3B greedy                 | 76.0 | 56.8 | 85.5 | 65.8 | 44 | 135 |
        | 32B greedy                     | 80.7 | 64.5 | 88.6 | 72.9 | 37 | 107 |
        vs baseline, 8B T=1.0: 100 fixed / 51 broken, p = 8.2e-5.
        32B vs 8B, both greedy (model change only): 81 fixed / 40 broken, p = 2.4e-4.
    """,
    verdict="""
        Superseded by rec4 on 2026-07-02, after one day as the development default: the only
        one of the recall-recovering prompts rec1-rec4 that stayed above the precision floor on
        the 8B. The guardrails closed the FP they targeted, but the positive-confirmation rule
        and guardrail A brought back the refusals rec1 had removed: A fired on uncontradicted
        details outside the image, so a citation or an unprinted p-value became grounds to
        refute again (seen most clearly in the 32B audit, STEP rec4). rec3 and rec4 both start
        from that problem.
    """,
    prompt="REC2_USER_PROMPT_TEMPLATE",
))

# The prompt as sent. New header; the decision rule, principles and guardrails are rewritten.
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


# ==================================================================================================
# STEP rec3 · main line 6 of 8 · rejected · from rec2 · 2026-07-01
# ==================================================================================================
HISTORY.append(dict(
    id="rec3", parent="rec2", date="2026-07-01", status="rejected",
    headline="8B T=1.0: acc 76.3 / pair 55.4 (P 82.6)",
    why="""
        The auditors looked at the 27 claims the baseline got right as Supported and rec2
        refuted (full dev, 8B, T=1.0). They judged all 27 labels correct. In 21, a guardrail
        caused the
        wrong Refuted (A 9, C 3, D 3, E 3, B 2, F 1); 6 were a perception miss. A fired on
        "since / because / indicating" rationale clauses, on values printed in the legend, and
        on magnitude words such as "slightly" or "almost overlaps". C fired on trends with no
        explicit universal and on counterexamples from inside the claimed group, D on top-N lists
        and plain comparatives, and F on an epoch-0 crossing.
    """,
    changes="""
        Where: identical to rec2 through RECALL PRINCIPLES and in the answer sentence; the
        guardrail header, A-F and the closing line are rewritten.
        - Guardrail header: they now "fire ONLY within the narrowed scope described ... a
          guardrail must not fire on something the chart cannot plot".
        - A keeps its core but gains three scope limits: it binds only the load-bearing PLOTTED
          quantity, so rationale clauses are not checked separately; legend text and printed
          labels are first-class evidence; magnitude adverbs and hedged relations are
          non-falsifiable. It also gains a rule for negated causal sources.
        - B: only PRECISE numbers (specific decimals); search all matching sub-blocks (e.g.
          Development and Test) before refuting; approximate multipliers ("~10x") are confirmed
          within about +/-25%.
        - C: fires only on an explicit universal marker. A general trend that holds for the
          clear majority is SUPPORTED. A counterexample must be against the named baseline and
          inside the set "all" ranges over. Footnoted or by-construction cells are excluded. The
          soft-adverb examples (broadly, generally, largely, notable) go.
        - D: only a SINGLE named item at the extreme; exempts collective top-N lists, plain
          comparatives, and exclusivity gated by a qualitative bar. Drops rec2's "primarily /
          particularly / especially / most notably X" clause (the fix for the "particularly
          large tables" FP).
        - E: compare only rows sharing the same condition; ties are not counterexamples; read
          each cited value from its own cell; a roughly constant large margin is confirmation;
          refute only when B strictly exceeds A. Drops "do not rescue it by narrowing to a
          favorable sub-setting".
        - F: the every-tick rule applies only to explicit "consistently / always" claims;
          general claims are judged on the converged trend; bind each curve to its legend label.
        - The closing line also exempts "an unplottable framing or magnitude word" and narrows
          the superlative case to a single item.
        - 4,225 -> 7,855 characters.
    """,
    result="""
        full dev, 8B T=1.0 (its only run): acc 76.3 | pair 55.4 | P 82.6 | R 69.9 | FP 58 |
        FN 119.
        vs rec2, 8B T=1.0: 62 fixed / 62 broken, p = 1.0 (570/747 correct for both); recall
        +4.8, precision -4.2.
        18 of the 27 audited claims came back Supported, the count the auditors predicted; but
        rec2 itself, decoded greedily, gets 17 of the same 27 right, so this does not show that
        the narrowing worked. It turned 38 of the baseline's 324 correct Refuted answers into
        Supported (rec2: 24), where the auditors had predicted a precision cost of only 0.5-1.5
        points.
    """,
    verdict="""
        Rejected: below the precision floor. rec3 moved the model toward Supported across the
        board (Supported answers 296 -> 334, FP 39 -> 58), not only on the targeted claims:
        narrowing guardrails in prose loosened behavior globally. With one T=1.0 sample per
        prompt, which individual claims flipped is mostly noise. rec4 later reused rec3's
        narrowed B-F wording (STEP rec4).
    """,
    prompt="REC3_USER_PROMPT_TEMPLATE",
))

# The prompt as sent. Everything down to the end of "RECALL PRINCIPLES" repeats rec2 word for
# word; the changes start at "GUARDRAILS".
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


# ==================================================================================================
# STEP rec4 · main line 7 of 8 · SUBMITTED · from rec2, borrows from rec1 and rec3 · 2026-07-01
# ==================================================================================================
HISTORY.append(dict(
    id="rec4", parent="rec2", date="2026-07-01", status="submitted",
    borrows="rec1 (decision rule), rec3 (guardrail B-F wording)",
    headline="32B greedy: acc 83.0 / pair 69.0; test: 81.1 / 64.1",
    why="""
        The idea. rec2 answered REFUTED in two different situations: when the evidence
        contradicted the claim, and when a claim about something the image plots was merely not
        confirmed by it. The labels are binary with no abstention class, so a claim the image
        neither confirms nor contradicts must go to one side, and the dev labels appear to put
        such claims on Supported (e.g. significance claims labeled Supported with no p-value
        printed). "Not positively confirmed -> REFUTED" therefore manufactures FN. rec4 refutes
        only on an actual contradiction, and a plottability test decides which parts of a claim
        the image can show at all.
        Where the idea came from. An outside review of rec2, by Claude on claude.ai without
        project context, argued that REFUTED was overloaded. Three Claude agents then checked
        it against the audit data. By the auditors' attribution, guardrail A was the fix for 1
        of the nine rec1 FP (STEP rec2) and the cause of 9 of the 27 rec2 regressions (STEP
        rec3), the largest single cause. The Claude session that ran them wrote the synthesis:
        refute only on a contradiction (strip A's positive confirmation), add a yes/no
        plottability test, narrow B against values read by eye or computed by the model, split
        F, and keep C/D/E firm, "only anchor-scope (as rec3 does)".
        Where the wording came from. The decision rule is rec1's almost word for word; the
        header and the six-guardrail structure are rec2's; B-F take rec3's narrowed scopes in
        condensed form; the plottability test follows the synthesis. The code comment written
        with rec4 calls C/D/E "kept FIRM ... (only anchor-scoped, not loosened)", but in the
        text that anchor-scoping is rec3's narrowing (a general trend holding for the clear
        majority is SUPPORTED, single-item superlatives only, "strictly exceeds"). So rec4
        departs from the synthesis on C-E: they carry the narrowing that the rec3 run had shown
        loosens behavior overall, and no run separates that from the new decision rule. The
        outside reviewer, shown the draft with that comment at its top and without rec3,
        repeated that C/D/E were not loosened.
    """,
    changes="""
        From rec1:
        1. The contradiction-based DECISION RULE: SUPPORTED when the evidence is consistent and
           nothing contradicts it; REFUTED when a value, ordering, direction, ranking or
           relationship actually CONTRADICTS the claim (rec1's list, plus "ordering"). Recall
           principle 1 restores rec1's "These permit exceptions".
        From rec3, condensed:
        2. B: PRECISE printed numbers only, searched in all matching sub-blocks.
        3. C, D, E: explicit universal markers only, majority trends supported, single-item
           superlatives, same-condition comparisons, ties not counterexamples, "strictly
           exceeds"; and C's counterexample rule (only against the exact named baseline, not an
           in-group row, not a by-construction or footnoted cell).
        4. F: legend binding first; the converged-trend rule for general claims.
        5. The closing line's exemption for unplottable framing / magnitude words, and the
           single-item superlative.
        New:
        6. After the decision rule: "Being unable to positively confirm a detail is NOT grounds
           to refute. Only an actual conflict refutes. Absence of proof is not proof of the
           opposite."
        7. A PLOTTABILITY TEST, replacing rec2's undefined "genuinely peripheral detail" and its
           first recall principle: does THIS image physically contain the axis, bar, curve, cell
           or legend entry this part of the claim is about? No -> peripheral (another table or
           figure, an appendix, a citation, an unprinted p-value, or a "since / because"
           rationale, which is rec3's exemption for A moved here): it may not refute unless the
           image shows its opposite. Yes -> core: check it.
        8. GUARDRAILS -> CONTRADICTION GUARDRAILS ("... but do NOT manufacture a conflict");
           they no longer override the recall principles.
        9. A changes from "positive-support requirement" to "ordering / ranking / comparison":
           refute only if the plotted order is reversed; "inability to read is not a
           contradiction".
        10. B: do not refute on a value estimated by eye or a delta the model computed itself
            when the claim is within reading tolerance: about one minor gridline, and +/-25%
            for any "~" or order-of-magnitude figure (rec3 allowed +/-25% only for approximate
            multipliers such as "~10x").
        11. F: for an explicit "consistently / always" claim, judge only the ticks that can
            actually be resolved; never narrate unplotted ticks or assert "no crossing" by
            default.
        Dropped:
        12. rec2's positive-confirmation rule, and "OR has no positive support in what the image
            plots" in the closing line; "causal link"; rec2's "particularly / especially / most
            notably X" (D) and "do not rescue it by narrowing to a favorable sub-setting" (E);
            rec3's legend-value, magnitude-adverb and negated-cause rules for A, D's
            qualitative-bar exemption, E's constant-margin exemption and "Read each cited value
            from its own cell first", and B's "a caption emphasizing one subset does not excuse
            skipping the others".
        4,225 -> 5,426 characters (rec2 -> rec4); header, inputs and answer sentence unchanged.
    """,
    result="""
        | setting              | acc  | pair | P    | R    | FP | FN |
        |----------------------|------|------|------|------|----|----|
        | full dev, 8B greedy  | 76.3 | 55.1 | 78.8 | 75.4 | 80 | 97 |
        | full dev, 32B greedy | 83.0 | 69.0 | 85.8 | 81.3 | 53 | 74 |
        | official test (a)    | 81.1 | 64.1 |      |      |    |    |
        (a) SIBTM_T1_R01, 32B greedy; macro-F1 81.1. 917 claims submitted; the scored population
        was not confirmed by the organizers.
        vs rec2, 8B greedy: 79 fixed / 71 broken, p = 0.57. On the 8B it only traded precision
        for recall.
        vs rec2, 32B greedy: 63 fixed / 46 broken, p = 0.125 (pairs: 45 fixed / 29 broken,
        p = 0.081): acc +2.3, recall +8.4, precision -2.8, pair +4.5. It recovers 51 of rec2's
        107 FN.
    """,
    verdict="""
        Submitted. Development default from 2026-07-02, together with the 32B, and the per-image
        verifier inside SIBTM_T2_R01.
        On the 8B, rec4 fell below the precision floor (P 78.8, under rec3's 82.6) and read as a
        wash. What sent it to the 32B was an audit of 40 of the 32B's 144 rec2 errors (27 FN,
        13 FP): it blamed 26 on the verification rules (20 FN and 6 FP; the FN split between
        guardrails firing on uncontradicted details and over-strict reading of hedges and
        quantifiers), 6 on perception, 8 on other causes and none on model capacity, and
        predicted that recall would move. On the 32B its precision was 85.8. rec3 was never run
        on the 32B.
        At the time, rec4 was adopted for its recall gain (+8.4) and because a re-adjudication
        of its errors put most of its precision loss down to label errors: on the corrected
        labels it led rec2 86.2 to 83.4 (65 fixed / 44 broken, p = 0.055). On the released
        labels the gain is not significant (p = 0.125); the paper keeps rec4 on pair accuracy
        and on the audit's prediction holding.
        The outside review of the draft flagged B's +/-25% tolerance and C's footnote /
        by-construction exclusion as untested constants worth ablating; neither was ablated.
        Lesson: a rule that recovers recall pays only when the model reads the evidence
        correctly. On the 8B, misreading was the binding constraint.
    """,
    prompt="REC4_USER_PROMPT_TEMPLATE",
))

# The prompt as sent, identical to REC4_USER_PROMPT_TEMPLATE in task1_standalone.py (`check`
# verifies it). Same header and answer sentence as rec2; everything between is reworked.
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


# ==================================================================================================
# STEP rec5 · main line 8 of 8 · rejected · from rec4 · 2026-07-02
# ==================================================================================================
HISTORY.append(dict(
    id="rec5", parent="rec4", date="2026-07-02", status="rejected",
    headline="32B greedy: acc 82.7 / pair 66.8 (P 89.3 / R 76.5)",
    why="""
        rec4's own reasoning on the 32B showed it being too lenient. It supported a claim whose
        core experiment the image does not show ("it simply does not show the experiment. So,
        nothing contradicts the claim. Therefore, the claim is SUPPORTED"). It called mislabeled
        panel titles "a cosmetic error, not a contradiction". It excused a false "despite"
        clause as "a general statement".
    """,
    changes="""
        Where: one new block, PRECISION GUARD, inserted before RECALL PRINCIPLES; every other
        line is rec4's.
        - "Nothing contradicts it" is NOT sufficient on its own: if the claim's CENTRAL
          quantity, experiment, comparison or result is absent from the image, answer REFUTED.
        - The plottability test's protection of peripheral parts covers only SUBORDINATE
          details, and only while the main assertion is shown.
        - A visible conflict (swapped labels, a wrong value, a reversed order, a false
          subordinate clause) may not be excused as cosmetic, a typo, a paraphrase or a general
          statement.
        - A hedge loosens an assertion's strength; it does not make an unshown assertion
          supported.
        In effect, a scoped return of rec2's guardrail A (positive support), while rec4's
        "Absence of proof is not proof of the opposite" stays in place.
    """,
    result="""
        full dev, 32B greedy: acc 82.7 | pair 66.8 | P 89.3 | R 76.5 | FP 36 | FN 93.
        vs rec4, 32B greedy: 45 fixed / 47 broken, p = 0.92.
        It removed 25 of rec4's 53 FP and added 8, but turned 39 of rec4's correct Supported
        answers into FN, 22 of them among the 51 that rec4 had recovered from rec2.
    """,
    verdict="""
        Rejected: net -2 claims and -8 pairs (-2.3 points); on the corrected labels it also
        trailed rec4 (85.4 vs 86.2). Asking again for positive support
        gives back much of the recall rec4 recovered (R 81.3 -> 76.5). The two NEI variants that
        also ask for positive support (STEP rec4_pos_nei, STEP rec4_nei2) lose too, but through
        FP (153 and 94; rec4: 53), not recall.
    """,
    prompt="REC5_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with the block below inserted just before "RECALL PRINCIPLES";
# nothing else changes. (sic) "may-not- refute", with the stray space, was sent that way; the
# line break after it here is a backslash continuation and was not sent.
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


# ##################################################################################################
# SIDE EXPERIMENTS: 10 steps in 4 groups. Each asks one question of rec2 or rec4; none was
# submitted.
# ##################################################################################################

# --------------------------------------------------------------------------------------------------
# GROUP table text: would an extracted copy of the table, given as text, help the model read it?
# Steps: rec2_hybrid. Answer: on the 8B, yes; on the 32B, no.
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# STEP rec2_hybrid · GROUP table text · side experiment · from rec2 · 2026-07-02
# ==================================================================================================
HISTORY.append(dict(
    id="rec2_hybrid", parent="rec2", date="2026-07-02", status="side experiment",
    headline="124 curated tables: 8B 36.3 -> 48.4 (p = 0.012); 32B null",
    why="""
        The baseline audit put roughly a quarter of the FN (about 10 of 40) down to
        table-encoding misreads: digits read correctly, but notation, row and column roles or
        scope misunderstood (a "(+28.9)" delta read as an absolute value, a "37/47" cell
        holding two samples). The auditors expected a less strict prompt not to repair this
        class, and under rec2 (greedy) the 8B still missed 89 tables as FN.
    """,
    changes="""
        Where: one block, PARSED TABLE, inserted before "Evaluate the claim against the
        evidence"; the rest of rec2 is unchanged.
        - The block holds a span-preserving HTML table (rowspan / colspan, bold cells) extracted
          by PaddleOCR 3.x (TableRecognitionPipelineV2), labeled STRUCTURE AID ONLY. Digits are
          still read from the image. A notation key covers "value (+d)" and "a / b". A
          table-vs-image mismatch never by itself refutes.
        - When the extraction is empty, the claim runs on plain rec2.
    """,
    result="""
        124 curated tables (89 the 8B got wrong as FN + 35 controls), greedy: 8B 36.3% ->
        48.4%, p = 0.012; 32B 63.7% -> 64.5%, p = 1.0. As recorded at the time: the per-claim
        predictions were not kept, so unlike the other results here these cannot be re-scored.
    """,
    verdict="""
        Not pursued. It helped the 8B on a set built from the 8B's own table errors; on the
        same set the 32B started far higher, partly because of that selection, and the table
        added nothing measurable. Never ported to rec4, never run on the full dev split.
    """,
    prompt="REC2_HYBRID_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec2 with the block below inserted before "Evaluate the claim against the
# evidence in the IMAGE (primary),"; nothing else changes.
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


# --------------------------------------------------------------------------------------------------
# GROUP abstention (NEI): would a third answer, NOT_ENOUGH_INFO, help?
# Steps: rec4_nei (as an error detector), rec4_pos_nei and rec4_nei2 (as a three-way rule).
# Answer: no. As a detector it barely fires; as a rule it costs the contradictions rec4 catches.
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# STEP rec4_nei · GROUP abstention · side experiment · from rec4 · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_nei", parent="rec4", date="2026-07-03", status="side experiment",
    headline="32B, 5 samples: NEI on 0.7%; error AUC 0.504 vs 0.687",
    why="""
        Would an explicit NOT_ENOUGH_INFO answer flag the claims the model is likely to get
        wrong better than the free signal of disagreement between samples?
    """,
    changes="""
        Where: rec4's answer sentence is replaced by an NEI block and a three-line answer
        (Verdict / Reason code / Answer); the rest of rec4 is unchanged.
        - NEI is a last resort with exactly two reasons, each of which must be named:
          perception (a core element is present but unreadable) or ambiguous (two opposite
          readings). Peripheral, unshown and hedged parts are never NEI; plain uncertainty is
          not NEI: "When in doubt between NEI and a verdict, give the verdict."
        - A forced best guess (leaning Supported unless there is a conflict) keeps every answer
          scoreable with the same parser.
    """,
    result="""
        full dev, 32B, 5 samples per claim at T=0.7: NEI on 5 of 747 claims (0.7%), never a
        majority. As an error detector: AUC 0.504, against 0.687 for the disagreement between
        the same 5 samples.
    """,
    verdict="""
        Dropped: the channel is inert. For an uncertainty signal, use the sampler's. The
        near-silence is partly by construction, since rec4 is built to be decisive; hence
        STEP rec4_pos_nei.
    """,
    prompt="REC4_NEI_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with its answer sentence (below) replaced by the NEI block and the
# three answer lines; nothing else changes. The NEI block is sent with its line breaks.
_REC4_ANSWER_SENTENCE = """\
Reason step by step, then conclude by stating: 'Therefore, the final answer is: \
Answer: $ANSWER' (without quotes), where $ANSWER is either 'Supported' or 'Refuted'.\
"""

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
    _REC4_ANSWER_SENTENCE, _NEI_BLOCK + _REC4_NEI_TAIL, 1)


# ==================================================================================================
# STEP rec4_pos_nei · GROUP abstention · side experiment · from rec4 · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_pos_nei", parent="rec4", date="2026-07-03", status="side experiment",
    headline="32B greedy: acc 71.9 / pair 44.3; FP 153 (rec4: 53)",
    why="""
        Give NEI a real job: remove rec4's "nothing contradicts -> SUPPORTED" lean, so that
        claims the image neither confirms nor contradicts go to NEI instead. It was expected to
        score lower on SciClaimEval; it was built to measure by how much.
    """,
    changes="""
        Written out in full below: it changes too many places for a single splice. Guardrails
        A-F and recall principles 2-3 are identical to rec4; principle 1 gains one sentence.
        - "Decide SUPPORTED, REFUTED, or NEI (not enough info)."
        - DECISION RULE (three-way, positive-support based): SUPPORTED only if the core
          assertion is visible and matches ("You can point to the axis, bar, curve, cell, or
          legend entry that shows it"); "nothing here contradicts it, but nothing here shows it
          either" -> NEI.
        - The plottability test is rewritten as core vs subordinate: a core part "MUST be
          positively shown".
        - Recall principle 1 gains: a hedge "does NOT substitute for absent evidence". The
          closing line becomes three-way.
        - rec4's answer sentence is replaced by an NEI answer section: a reason-code block
          (unconfirmed, perception, ambiguous), a paragraph "NEI is NOT a dumping ground", and
          three answer lines ending in a forced Supported / Refuted guess.
    """,
    result="""
        full dev, 32B greedy, scored on the forced guess: acc 71.9 | pair 44.3 | FP 153 |
        FN 57. NEI on 99 claims (13.3%), all "unconfirmed".
    """,
    verdict="""
        Dropped. Asked to look for support, the model stopped looking for a mismatch, and on
        this data that search is the task. On the 648 claims it answered outright it made 109
        FP where rec4 made 38 on the same claims; in a 30-FP sample that the auditors
        classified on the corrected labels, they read 27 as missed contradictions, the model
        latching onto a confirming detail. The other 44 FP come from the forced guess leaning
        Supported on the 99 NEI answers.
    """,
    prompt="REC4_POS_NEI_USER_PROMPT_TEMPLATE",
))

# The prompt as sent, in full. Same as rec4 in the header, recall principles and guardrails A-F
# (principle 1 gains one sentence); the intro sentence, decision rule, plottability test and
# closing line are rewritten, and an NEI answer section replaces rec4's answer sentence.
REC4_POS_NEI_USER_PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a figure or table.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Evaluate the claim against the evidence in the IMAGE (primary), using the caption \
and context as support. Decide SUPPORTED, REFUTED, or NEI (not enough info).

DECISION RULE (three-way, positive-support based)
- SUPPORTED: the claim's CORE assertion -- the value, ordering, direction, ranking, \
or relationship it asserts -- is VISIBLE in the evidence and MATCHES. You can point \
to the axis, bar, curve, cell, or legend entry that shows it, and it agrees.
- REFUTED: the visible evidence CONTRADICTS the claim -- a value, ordering, \
direction, ranking, or relationship in the image conflicts with it.
- NEI (Not Enough Info): the claim's CORE assertion is something THIS image could \
carry, but the image NEITHER confirms it NOR contradicts it -- the \
value/order/relation is absent, or present but unresolvable, so you can point to NO \
positive support AND NO conflict. The band "nothing here contradicts it, but nothing \
here shows it either" lands HERE, not in SUPPORTED.
- A genuinely SUBORDINATE or external part (citation, appendix, unprinted p-value / \
significance, "since / because / despite" rationale clause) does NOT need positive \
support and does NOT trigger NEI or REFUTED on its own -- judge the claim on its \
CORE. Peripheral-and-uncontradicted stays SUPPORTED.

PLOTTABILITY TEST (apply to each part of the claim before judging it)
Ask: "Is THIS part the claim's CORE assertion, or a subordinate/external detail?" \
and "Does THIS image physically contain the thing this part is about (an axis, bar, \
curve, cell, or legend entry)?"
- SUBORDINATE / EXTERNAL part -- a citation, an appendix or a different \
table/figure, a p-value / confidence interval / significance not printed here, or a \
"since / because / indicating / despite / although" rationale clause. This is NOT \
what the claim is being verified on. Do NOT require positive support for it, and do \
NOT refute on it, UNLESS the image shows the OPPOSITE of it.
- CORE part that the image CAN carry -- the value, ordering, comparison, or \
relationship the claim actually asserts. This MUST be positively shown:
    * present and matches -> supports SUPPORTED;
    * present and conflicts -> REFUTED;
    * capable of being shown here but NOT shown, or present-but-unresolvable -> NEI \
(neither positively supported nor contradicted).

RECALL PRINCIPLES
1. Read hedged / approximate wording charitably: "almost all", "usually", \
"typically", "~", "roughly", "about", "tends to", and approximate rounding of a \
stated value. These permit exceptions and are never exact assertions. A hedge \
loosens the THRESHOLD of an assertion whose evidence is PRESENT; it does NOT \
substitute for absent evidence -- a hedged core assertion still needs to be visibly \
shown to be SUPPORTED.
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

Distinguish carefully: a SUBORDINATE detail that is merely unshown, paraphrased, or \
an unplottable framing/magnitude word stays SUPPORTED (do NOT refute or mark NEI); a \
CORE value, ordering, direction, or relationship the image PLOTS and CONTRADICTS is \
REFUTED; a CORE assertion this image COULD show but does NOT (neither confirmed nor \
contradicted) is NEI.

NOT_ENOUGH_INFO (NEI) reason codes -- pick exactly one when your Verdict is NEI:
- unconfirmed: the claim's CORE value/order/relation is not shown in this image (the \
element that would carry it is absent, or shows something unrelated), so there is no \
positive support and also no conflict. You MUST name the element that is missing. \
This is the PRIMARY NEI case -- the "uncontradicted but not positively confirmed" \
band.
- perception: the CORE element physically EXISTS in this image but you genuinely \
cannot resolve the value you need (overlapping/occluded curves, segments too small, \
digits too low resolution). You MUST name the specific element you cannot read.
- ambiguous: the claim admits TWO OPPOSITE verdicts because it is under-specified \
against what the image shows (a referent mapping to two rows/series; a comparison \
with no single defensible reading). You MUST state both readings.

NEI is NOT a dumping ground. It does NOT apply when: the CORE assertion IS visibly \
shown and matches (that is SUPPORTED, commit to it); the evidence conflicts (that is \
REFUTED); or the part is SUBORDINATE/external or the wording is hedged/approximate \
(that is SUPPORTED via the carve-out and Recall Principle 1; \
peripheral-uncontradicted is SUPPORTED, never NEI). Use NEI only for a CORE, \
plottable assertion that this image can neither confirm nor contradict.

Reason step by step, then conclude with EXACTLY these three lines, each on its own \
line and nothing after them:
Verdict: $VERDICT
Reason code: $CODE
Answer: $ANSWER
where $VERDICT is 'Supported', 'Refuted', or 'NEI'; $CODE is 'unconfirmed', \
'perception', or 'ambiguous' when $VERDICT is 'NEI', otherwise 'none'; and $ANSWER \
is a forced binary best guess, ALWAYS 'Supported' or 'Refuted' (when $VERDICT is \
'Supported' or 'Refuted', $ANSWER must equal it; when $VERDICT is 'NEI', lean \
'Supported' unless you can point to an actual conflict).\
"""


# ==================================================================================================
# STEP rec4_nei2 · GROUP abstention · side experiment · from rec4 (+ rec4_pos_nei) · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_nei2", parent="rec4", date="2026-07-03", status="side experiment",
    borrows="rec4_pos_nei (intro sentence and NEI answer section)",
    headline="32B greedy: acc 77.5 / pair 56.3; FP 94 (rec4: 53)",
    why="""
        Fix rec4_pos_nei, whose new errors were mostly missed contradictions: keep rec4's
        contradiction machinery intact, make the contradiction hunt explicitly come first, and
        only then split the uncontradicted case into Supported and NEI.
    """,
    changes="""
        Written out in full below. Where: identical to rec4 except the intro sentence, the
        DECISION RULE and the answer section; the plottability test, recall principles,
        guardrails A-F and the closing line are rec4's.
        - The same "SUPPORTED, REFUTED, or NEI" intro sentence and NEI answer section (reason
          codes, "NEI is NOT a dumping ground", three answer lines) as rec4_pos_nei.
        - DECISION RULE (contradiction-first, three-way): "FIRST hunt for a contradiction across
          EVERY core part of the claim ... Finding ONE detail that matches is NOT enough". Only
          once nothing contradicts does the split apply: SUPPORTED needs positive evidence for
          the core assertion, otherwise NEI ("unconfirmed"). rec4's SUPPORTED definition and its
          "Absence of proof is not proof of the opposite." are removed; "Being unable to
          positively confirm a detail is NOT grounds to refute. Only an actual conflict
          refutes." is kept.
    """,
    result="""
        full dev, 32B greedy, scored on the forced guess: acc 77.5 | pair 56.3 | FP 94 | FN 74.
        NEI on 70 claims (9.4%).
        vs rec4_pos_nei: 64 fixed / 22 broken, p = 6.5e-6.
        vs rec4: 43 fixed / 84 broken, p = 3.5e-4.
    """,
    verdict="""
        Dropped. Putting the contradiction hunt first recovered about half of rec4_pos_nei's
        loss but still fell 5.5 points short of rec4: with a live NEI channel, the model still
        misses contradictions rec4 catches. NEI and rec4's contradiction detection look
        entangled.
    """,
    prompt="REC4_NEI2_USER_PROMPT_TEMPLATE",
))

# The prompt as sent, in full. Same as rec4 except the intro sentence, the decision rule and the
# answer section; the intro sentence and the answer section are rec4_pos_nei's, the decision
# rule is new.
REC4_NEI2_USER_PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a figure or table.

Claim:
{claim}

Caption:
{caption}

Context:
{context}

Evaluate the claim against the evidence in the IMAGE (primary), using the caption \
and context as support. Decide SUPPORTED, REFUTED, or NEI (not enough info).

DECISION RULE (contradiction-first, three-way)
FIRST hunt for a contradiction across EVERY core part of the claim, applying the \
CONTRADICTION GUARDRAILS below. Finding ONE detail that matches is NOT enough to \
support -- a genuine conflict in ANY core part means REFUTED. Do not settle on \
SUPPORTED before you have checked the whole claim for conflicts.
- REFUTED: a value, ordering, direction, ranking, or relationship in the evidence \
actually CONTRADICTS the claim.
- Being unable to positively confirm a detail is NOT grounds to refute. Only an \
actual conflict refutes.
- Only once you are sure NOTHING in the image contradicts the claim, split the \
remaining (uncontradicted) case:
    * SUPPORTED: you can point to POSITIVE visible evidence for the claim's CORE \
assertion -- the value/order/relationship it asserts is actually shown and matches.
    * NEI (Not Enough Info, reason 'unconfirmed'): nothing contradicts the claim, \
but its CORE assertion is NOT positively shown here (the value/order/relation is \
absent or unresolvable), so you cannot point to evidence FOR it. A merely \
SUBORDINATE or external unshown detail (a citation, an appendix, an unprinted \
p-value, a "since / because / despite" clause) does NOT trigger NEI -- that stays \
SUPPORTED.

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

NOT_ENOUGH_INFO (NEI) reason codes -- pick exactly one when your Verdict is NEI:
- unconfirmed: the claim's CORE value/order/relation is not shown in this image (the \
element that would carry it is absent, or shows something unrelated), so there is no \
positive support and also no conflict. You MUST name the element that is missing. \
This is the PRIMARY NEI case -- the "uncontradicted but not positively confirmed" \
band.
- perception: the CORE element physically EXISTS in this image but you genuinely \
cannot resolve the value you need (overlapping/occluded curves, segments too small, \
digits too low resolution). You MUST name the specific element you cannot read.
- ambiguous: the claim admits TWO OPPOSITE verdicts because it is under-specified \
against what the image shows (a referent mapping to two rows/series; a comparison \
with no single defensible reading). You MUST state both readings.

NEI is NOT a dumping ground. It does NOT apply when: the CORE assertion IS visibly \
shown and matches (that is SUPPORTED, commit to it); the evidence conflicts (that is \
REFUTED); or the part is SUBORDINATE/external or the wording is hedged/approximate \
(that is SUPPORTED via the carve-out and Recall Principle 1; \
peripheral-uncontradicted is SUPPORTED, never NEI). Use NEI only for a CORE, \
plottable assertion that this image can neither confirm nor contradict.

Reason step by step, then conclude with EXACTLY these three lines, each on its own \
line and nothing after them:
Verdict: $VERDICT
Reason code: $CODE
Answer: $ANSWER
where $VERDICT is 'Supported', 'Refuted', or 'NEI'; $CODE is 'unconfirmed', \
'perception', or 'ambiguous' when $VERDICT is 'NEI', otherwise 'none'; and $ANSWER \
is a forced binary best guess, ALWAYS 'Supported' or 'Refuted' (when $VERDICT is \
'Supported' or 'Refuted', $ANSWER must equal it; when $VERDICT is 'NEI', lean \
'Supported' unless you can point to an actual conflict).\
"""


# --------------------------------------------------------------------------------------------------
# GROUP crop and zoom: would a high-resolution crop of the evidence help the model read it?
# Steps: localize (finds the region), rec4_crop (full image + crop), rec4_croponly (crop alone),
# localize_table (tables only), rec4_highlight (full table, region highlighted).
# Answer: no. Nothing survived a paired control and a replication.
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# STEP localize · GROUP crop and zoom · side experiment · new · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="localize", parent=None, date="2026-07-03", status="side experiment",
    headline="box usable on 121 of 140 figures and 52 of 60 tables",
    why="""
        After scale, quantization, whole-image resolution and table injection, what remained
        on the 32B included perception errors, mostly on figures. The idea: ask the model where
        the evidence is, crop that region, and verify again at high resolution.
    """,
    changes="""
        First prompt of its kind, and not a verifier prompt: a separate first call that returns
        one JSON line, {"bbox_2d": [x1, y1, x2, y2], "found": true}, or found = false when the
        whole figure is needed. Greedy, 256 tokens. The box is checked, padded and upscaled in
        code, then passed to rec4_crop or rec4_croponly.
    """,
    result="""
        enriched 200-claim set (140 figures, 60 tables): the box was usable for 121 of 140
        figures and 52 of 60 tables. Verification results are under STEP rec4_crop.
    """,
    verdict="""
        Worked mechanically. Replaced by localize_table after the 200-claim set, scored at the
        time on our corrected labels, suggested cropping tables only (STEP rec4_crop).
    """,
    prompt="LOCALIZE_PROMPT_TEMPLATE",
))

# The prompt as sent.
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


# ==================================================================================================
# STEP rec4_crop · GROUP crop and zoom · rejected · from rec4 · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_crop", parent="rec4", date="2026-07-03", status="rejected",
    headline="425 cropped tables: 24 fixed / 12 broken (p = 0.065); not replicated",
    why="""
        Spend the pixel budget where the claim's evidence is: give the model the full image plus
        a high-resolution crop of the localized region.
    """,
    changes="""
        Where: one paragraph inserted before "Evaluate the claim ...", and "IMAGE (primary)" ->
        "IMAGES (primary)"; the rest of rec4 is unchanged.
        - "You are given TWO images of the SAME figure: image 1 is the full figure (overall
          context); image 2 is a high-resolution ZOOM into the region most relevant to the
          claim ... trust the zoom for fine values."
        - The crop goes in as a second image. Without a usable box, the claim runs on plain
          rec4.
    """,
    result="""
        enriched 200-claim set: scored at the time on our corrected labels and reweighted to dev
        base rates, it looked useful on tables but not on figures, hence localize_table. On the
        released labels that set shows no table-specific advantage (tables 9 fixed / 4 broken,
        figures 17 / 10, both not significant).
        dev tables, 32B greedy, vs a same-session rec4 control, on the 425 of 482 that got a
        crop (boxes from STEP localize_table): 24 fixed / 12 broken, p = 0.065.
        replication: 24 fixed / 19 broken, p = 0.54. A header-preserving stitched crop with the
        same prompt did no better.
    """,
    verdict="""
        Rejected: not significant, did not replicate, and doubles inference. The replication's
        rec4 control matched run 1's answer for answer, so the drop came from the crop arm
        itself. Between the other table sessions, the rec4 control on the same 482 tables
        differs by 5.6-7.3%, so effects of this size can be compared only within a session.
    """,
    prompt="REC4_CROP_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with the paragraph below inserted before "Evaluate the claim", and
# "IMAGE (primary)" changed to "IMAGES (primary)"; nothing else changes. The crop is image 2.
_CROP_LINE_A = """\
You are given TWO images of the SAME figure: image 1 is the full figure (overall context); \
image 2 is a high-resolution ZOOM into the region most relevant to the claim. Read fine \
details -- overlapping curves, small bars, markers, legend entries, exact axis values -- from \
image 2; use image 1 for global structure. They show the same data; if they appear to \
disagree, trust the zoom for fine values.

"""

REC4_CROP_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _CROP_LINE_A + "Evaluate the claim against the evidence in the IMAGES (primary),", 1)


# ==================================================================================================
# STEP rec4_croponly · GROUP crop and zoom · rejected · from rec4 · 2026-07-03
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_croponly", parent="rec4", date="2026-07-03", status="rejected",
    headline="200-claim set, figures: 19 fixed / 29 broken (p = 0.19)",
    why="""
        Run beside rec4_crop on the 200-claim set: does the zoom work on its own, without the
        full image?
    """,
    changes="""
        Where: one line inserted before "Evaluate the claim ..."; the rest of rec4 is unchanged.
        "This image is a high-resolution ZOOM into the region of a scientific figure most
        relevant to the claim." The crop replaces the full image.
    """,
    result="""
        enriched 200-claim set, vs a separately run rec4 control on the same claims: figures
        19 fixed / 29 broken, p = 0.19 (19 / 26 on the 121 cropped figures).
    """,
    verdict="""
        Rejected. On figures the crop alone scored lower than the full image, though not
        conclusively: the control ran in another session, and 3 of the 10 net breaks fall on
        figures that were not cropped. Losing the axes, legend and other series was the
        suspected cause, so the crop was kept only as a second image beside the full one
        (STEP rec4_crop).
    """,
    prompt="REC4_CROPONLY_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with the line below inserted before "Evaluate the claim"; nothing else
# changes. The crop replaces the full image.
_CROP_LINE_B = """\
This image is a high-resolution ZOOM into the region of a scientific figure most relevant to \
the claim.

"""

REC4_CROPONLY_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _CROP_LINE_B + "Evaluate the claim against the evidence in the IMAGE (primary),", 1)


# ==================================================================================================
# STEP localize_table · GROUP crop and zoom · side experiment · from localize · 2026-07-04
# ==================================================================================================
HISTORY.append(dict(
    id="localize_table", parent="localize", date="2026-07-04", status="side experiment",
    headline="boxes 94.4% of tables, skips 93.2% of figures",
    why="""
        Fold the "crop tables only" policy into the single localization call. It is also meant
        to cover a table inside a multi-panel image, which a per-image modality flag cannot see
        (not measured).
    """,
    changes="""
        Box only the rows, columns or cells the claim refers to, and only if the evidence is a
        table. For a chart, plot or any non-tabular figure, "do NOT localize": found = false,
        and the caller keeps the full image.
    """,
    result="""
        full dev, 32B greedy: boxes 455 of 482 tables (94.4%) and abstains on 247 of 265
        figures (93.2%); 425 tables pass the crop checks.
    """,
    verdict="""
        An accurate gate, but the crop it gates did not pay off (STEP rec4_crop), so it shipped
        nowhere.
    """,
    prompt="LOCALIZE_TABLE_PROMPT_TEMPLATE",
))

# The prompt as sent.
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


# ==================================================================================================
# STEP rec4_highlight · GROUP crop and zoom · rejected · from rec4 · 2026-07-04
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_highlight", parent="rec4", date="2026-07-04", status="rejected",
    headline="408 highlighted tables: 20 fixed / 20 broken (p = 1.0)",
    why="""
        Keep the whole table but point at the evidence: can emphasis without cutting get what
        cropping fixes without what cropping breaks?
    """,
    changes="""
        Where: one paragraph inserted before "Evaluate the claim ..."; the rest of rec4 is
        unchanged. The image is the FULL table with a translucent highlight box (red border,
        yellow shading) over the region most relevant to the claim; read the highlighted cells,
        but use the entire table for context. The box comes from localize_table and is drawn
        onto the image in code.
    """,
    result="""
        dev tables, 32B greedy, vs a same-session rec4 control, on the 408 of 482 that got a
        highlight box: 20 fixed / 20 broken, p = 1.0.
    """,
    verdict="""
        Rejected: null. Together with a re-render of the tables at crop-equivalent resolution
        (no prompt change: 24 fixed / 24 broken, p = 1.0), it closed the presentation
        experiments: how the image was presented did not matter; the decision rule did.
    """,
    prompt="REC4_HIGHLIGHT_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with the paragraph below inserted before "Evaluate the claim"; nothing
# else changes.
_HL_LINE = """\
This image is the FULL table with a translucent highlight box (red border, yellow shading) \
marking the region most relevant to the claim. Read the highlighted cells carefully, but use \
the ENTIRE table for context -- row and column headers, and the other rows -- since the \
highlight only points at the evidence, it does not remove the rest.

"""

REC4_HIGHLIGHT_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    "Evaluate the claim against the evidence in the IMAGE (primary),",
    _HL_LINE + "Evaluate the claim against the evidence in the IMAGE (primary),", 1)


# --------------------------------------------------------------------------------------------------
# GROUP Subtask-2 tie-break: rec4 also serves Subtask 2, once per candidate image. Could a graded
# score from it break the ties between two images? Steps: rec4_conf. Answer: no, not as well as a
# joint two-image call.
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# STEP rec4_conf · GROUP Subtask-2 tie-break · side experiment · from rec4 · 2026-07-05
# ==================================================================================================
HISTORY.append(dict(
    id="rec4_conf", parent="rec4", date="2026-07-05", status="side experiment",
    headline="Subtask-2 ties (5-sample run): 45% stay tied; 92.0 vs 93.2",
    why="""
        Subtask 2 runs rec4 on each of two candidate images. In an exploratory run that judged
        each image with 5 samples at T=0.7, both images got the same majority verdict on 104 of
        the 352 dev claims. A graded per-image score might break these ties without a joint
        two-image call. Like rec4, the prompt sees one image and says nothing about the other.
    """,
    changes="""
        Where: rec4's answer sentence is replaced by a new lead-in line ("Reason step by step.
        Then conclude with these TWO lines, exactly:") and two answer lines; the rest of rec4 is
        unchanged. The same "Therefore, the final answer is: Answer: $ANSWER", plus "Support
        score: $N",
        an integer 0-100 (0 = the image clearly contradicts the claim, 50 = borderline, 100 =
        clearly and fully supports it; "Use the full range and avoid defaulting to round
        numbers").
    """,
    result="""
        those 104 ties, 32B greedy: identical scores on 47 (45%), an unparsable score on 1, and
        43 of the 56 ties it did separate were correct.
        Breaking ties by score first and sending the other 48 to the joint two-image call scored
        92.0 on Subtask-2 dev (score alone: 81.5), against 93.2 for the joint call on every tie,
        both after the 5-sample first step. The shipped greedy cascade (SIBTM_T2_R01) ties on 97
        of 352 and scores 92.9; rec4_conf was not run on its ties.
    """,
    verdict="""
        Not used: SIBTM_T2_R01 breaks ties with the joint call (see task2_standalone.py). Judged
        one image at a time, the score leaves 45% of the ties unbroken, and where it does break
        them it is right less often than the joint call (43 vs 47 of 56).
    """,
    prompt="REC4_CONF_USER_PROMPT_TEMPLATE",
))

# The prompt as sent: rec4 with its answer sentence (STEP rec4_nei) replaced by the lead-in
# line and the two answer lines below; nothing else changes.
_CONF_ANSWER_LINES = """\
Reason step by step. Then conclude with these TWO lines, exactly:
'Therefore, the final answer is: Answer: $ANSWER' (without quotes), where $ANSWER is either \
'Supported' or 'Refuted'.
'Support score: $N' (without quotes), where $N is an integer from 0 to 100 giving how strongly \
the visible evidence in the IMAGE supports the claim: 0 = the image clearly contradicts the \
claim, 50 = genuinely borderline, 100 = the image clearly and fully supports it. Use the full \
range and avoid defaulting to round numbers.\
"""

REC4_CONF_USER_PROMPT_TEMPLATE = REC4_USER_PROMPT_TEMPLATE.replace(
    _REC4_ANSWER_SENTENCE, _CONF_ANSWER_LINES, 1)


# ##################################################################################################
# THE TOOL. Nothing below is needed to read the history above: these functions only print it
# (list / show / diff) and check that every prompt is the one that was sent (check).
# ##################################################################################################

import argparse  # noqa: E402  (the imports live here so that the file opens on the history)
import ast  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import re  # noqa: E402
import textwrap  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """One HISTORY record, with its prompt text resolved."""

    id: str
    parent: str | None
    date: str
    status: str
    headline: str
    why: str
    changes: str
    result: str
    verdict: str
    prompt: str
    borrows: str | None = None


def _resolve(prompt) -> str:
    """A record names its prompt constant, or a (system, user) pair of constants."""
    if isinstance(prompt, tuple):
        system, user = (globals()[name] for name in prompt)
        return f"[system]\n{system}\n\n[user]\n{user}"
    return globals()[prompt]


STEPS = {record["id"]: Step(**{**record, "prompt": _resolve(record["prompt"])})
         for record in HISTORY}

# SHA-256 of every prompt as sent (for v0 and v1: system and user message, as `show` prints
# them), recorded from the development repository's prompt module and, for v0 and v1, its git
# history.
PROMPT_SHA256 = {
    "v0": "56fbea7d618be04a9ef6982c3d518c0594e7dcba6551c817898ab36c3e454d4a",
    "v1": "71f444be8945183f01b9b10aa9b3c78739eed8c14bad0108e4992ad842617a6d",
    "localize": "90eb1f7ca45f2c7c1f578d2b4043f38eaeda1909183e44f57bca0f8e6f899a4c",
    "localize_table": "fcf704e826b478c4bc32caaf8ed9ef058fc1eb187c3a76f42c557bfa5702161a",
    "baseline": "2ee41883c4b13390ef8e81c783d388bfe904e5c894ee56c873f2d231bf91f7bb",
    "rec1": "295712b7b080e5660a1273e24146bfa709ef9a793fb96317d956b3e7fdc222c7",
    "rec2": "ecb26008d7411de625a78b58a6d866737513ad234ccc4a1360e18f4964d8084b",
    "rec3": "54c2d1a22a8921aa69837a20912bf4c563b3945ba3108f0f9a1543e93e154c57",
    "rec4": "01bd26caad1303fb309f359dbf2a9c1040b9fa8fdbc0ee31de9488b1219435c8",
    "rec5": "a4dd20a7076d5ff0ee2e1cedc3509a6853ca3f39ec6aed629b99f7a12ff67f70",
    "rec2_hybrid": "708ca5f4f4884d6254ba7d5346f84fb66062536967a91450db670e94c544f217",
    "rec4_nei": "78e014b4b83ac42726e237245507f30fd322e0303cbc12ebb1653d4ab5eb11bd",
    "rec4_pos_nei": "7d19fdba7f4fd65d45fdb87930ae4feeb9590c88c1c72fa573736d1cf3cdc7d3",
    "rec4_nei2": "c18648d0c5a0ab16410fd04a7b0059db8d1945624af4e394d94ea54726ecfe24",
    "rec4_crop": "0ab1da107352d24306e9b01598b88482f4c1e48453930f29df668f817e8d32da",
    "rec4_croponly": "825f4a85f3db191084ff12b69ffeb5ae579e75e791448f034ae3c7fa385e557f",
    "rec4_highlight": "82800943da7b5ebc70036bed3764cb333d54ee54f72117197cb913bef7e6fae0",
    "rec4_conf": "4d1bc0868a3fb4227ca1b6ccc84e8229b4392facbf0e755a907f7c96874eb67e",
}

# Sentence boundary inside a prompt line: diffing sentence by sentence points at the edited
# sentence instead of flagging the whole paragraph. A list marker ("1.", "A.") is not a
# sentence and stays attached to the text that follows it.
_SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")
_LIST_MARKER = re.compile(r"(?:\d+|[A-Z])\.")


def _prose(text: str) -> str:
    return textwrap.indent(textwrap.dedent(text).strip(), "  ")


def lineage() -> str:
    rows = [("id", "parent", "date", "status", "result")]
    rows += [(s.id, s.parent or "-", s.date, s.status, s.headline) for s in STEPS.values()]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[4]
        for row in rows
    )


def show(step: Step) -> str:
    against = f" vs {step.parent}" if step.parent else ""
    borrows = f"; borrows from {step.borrows}" if step.borrows else ""
    return "\n\n".join([
        f"{step.id}  (parent: {step.parent or '-'}{borrows}; {step.date}; {step.status})\n"
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


def check() -> list[str]:
    """Problems found: prompts that differ from what was sent, or a rec4 that differs from the
    one task1_standalone.py runs. An empty list means all is well."""
    problems = [
        f"{step_id}: prompt text differs from the one that was sent"
        for step_id, step in STEPS.items()
        if hashlib.sha256(step.prompt.encode()).hexdigest() != PROMPT_SHA256.get(step_id)
    ]
    tree = ast.parse((ROOT / "task1_standalone.py").read_text())
    shipped = next((ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == "REC4_USER_PROMPT_TEMPLATE"
                            for t in node.targets)), None)
    if shipped != STEPS["rec4"].prompt:
        problems.append("rec4 differs from REC4_USER_PROMPT_TEMPLATE in task1_standalone.py")
    return problems


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
    sub.add_parser("check", help="verify every prompt, and that rec4 is the one the runner uses")
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
        problems = check()
        print("\n".join(problems) or f"all {len(STEPS)} prompts verified; rec4 matches "
                                       "task1_standalone.py")
        raise SystemExit(1 if problems else 0)
    else:
        print(lineage())


if __name__ == "__main__":
    main()
