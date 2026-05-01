#!/usr/bin/env python3
"""
Two-phase OpenAI runner on a questions-only file (e.g. eval_set/from_youtube_video/questions.txt).

Phase 1 — stem only: the model sees the question text but **not** the A–D lines.
  It must write reasoning first (no option letter).

Phase 2 — options: the same question with A–D shown; the model picks exactly one letter.

Does NOT read answers.txt. Default model is gpt-5.5 with Responses API reasoning;
use --reasoning-effort off for models that reject the reasoning field.

Auth:
  export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

from openai import BadRequestError, OpenAI

_SCRIPTS = Path(__file__).resolve().parent
# Module must be in sys.modules before exec_module: Python 3.9 dataclasses look up
# sys.modules[cls.__module__] while processing class bodies (importlib otherwise leaves it None).
_MOD = "_broker_run_questions_reasoned_openai"
_spec = importlib.util.spec_from_file_location(_MOD, _SCRIPTS / "run_questions_reasoned_openai.py")
assert _spec and _spec.loader
_rq = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _rq
_spec.loader.exec_module(_rq)

parse_questions = _rq.parse_questions
MCQ = _rq.MCQ
parse_model_response = _rq.parse_model_response

PHASE1_SYSTEM = """You are answering California P&C–style licensing exam questions.

You will see ONLY the question stem. The multiple-choice options A–D are intentionally hidden.
Explain what the question is asking, which concepts or rules apply, and how you would narrow
the answer — in 4–8 sentences.

Rules:
- Do NOT output option letters A, B, C, or D anywhere in your reply.
- Do not claim you already know the correct letter (you have not seen the choices).

Output format (exactly one line starting with this prefix, then your text):
REASON: <your reasoning>"""

PHASE2_SYSTEM = """You are continuing the same exam question. You now see the full stem plus options.

You previously reasoned without seeing choices; use that if helpful, but you may revise.

Output EXACTLY two lines and nothing else:
OPTION: A|B|C|D
REASON: one or two sentences (why this option fits the stem)"""


def collect_output_text(resp: object) -> str:
    t = (getattr(resp, "output_text", None) or "").strip()
    if t:
        return t
    parts: list[str] = []
    for item in resp.output:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                parts.append(getattr(c, "text", "") or "")
    return "".join(parts).strip()


def _responses_create(client: OpenAI, req: dict) -> object:
    """Some models (e.g. gpt-5.5) reject ``temperature`` on the Responses API."""
    try:
        return client.responses.create(**req)
    except BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg and "not supported" in msg and "temperature" in req:
            req2 = {k: v for k, v in req.items() if k != "temperature"}
            return client.responses.create(**req2)
        raise


def call_responses(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> str:
    req: dict = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort and str(reasoning_effort).lower() != "off":
        req["reasoning"] = {"effort": reasoning_effort}
    resp = _responses_create(client, req)
    return collect_output_text(resp)


def build_phase1_user(q: MCQ) -> str:
    return f"Question {q.number} (stem only — no options yet):\n\n{q.stem}"


def build_phase2_user(q: MCQ, prior_reason: str) -> str:
    lines = [
        "Your earlier reasoning (written before you saw any options):",
        "",
        prior_reason.strip(),
        "",
        "---",
        "",
        f"Question {q.number} (full, with options):",
        f"{q.stem}",
        "",
    ]
    for k in ("A", "B", "C", "D"):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    return "\n".join(lines)


def parse_phase1_reason(text: str) -> str:
    t = (text or "").replace("\x00", "")
    lines = [ln.rstrip() for ln in t.splitlines()]
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.upper().startswith("REASON:"):
            parts = [s.split(":", 1)[1].strip()]
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                if nxt.upper().startswith("OPTION:"):
                    break
                parts.append(nxt)
            return re.sub(r"\s+", " ", " ".join(parts)).strip()
    return re.sub(r"\s+", " ", t.strip()).strip() or "PARSE_FAILED: empty phase-1 response"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, type=Path, help="Path to questions.txt")
    ap.add_argument("--model", default="gpt-5.5", help="OpenAI model, e.g. gpt-5.5")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--reasoning-effort",
        default="medium",
        help='Reasoning effort for Responses API (e.g. low, medium, high). Use "off" to omit.',
    )
    ap.add_argument(
        "--max-output-tokens-phase1",
        type=int,
        default=4096,
        help="Max output tokens for phase 1 (reasoning models need headroom)",
    )
    ap.add_argument("--max-output-tokens-phase2", type=int, default=1024)
    args = ap.parse_args()

    if args.questions.name.lower().endswith("answers.txt"):
        raise SystemExit("Refusing to run: --questions points at answers.txt")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    re_eff: str | None = args.reasoning_effort
    if re_eff and re_eff.lower() == "off":
        re_eff = None

    qs = parse_questions(args.questions)
    if not qs:
        print(f"No questions parsed from {args.questions}", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        qs = qs[: args.limit]

    client = OpenAI()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_number", "reason_before_options", "option", "reason_after_options"])
        f.flush()

        for q in qs:
            present = [k for k in ("A", "B", "C", "D") if q.choices.get(k)]
            if len(present) < 2:
                w.writerow(
                    [
                        q.number,
                        "",
                        "",
                        f"SKIPPED: malformed question (choices present: {present})",
                    ]
                )
                f.flush()
                continue

            u1 = build_phase1_user(q)
            t0 = time.time()
            raw1 = call_responses(
                client,
                model=args.model,
                system=PHASE1_SYSTEM,
                user=u1,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens_phase1,
                reasoning_effort=re_eff,
            )
            r_before = parse_phase1_reason(raw1)

            u2 = build_phase2_user(q, r_before)
            raw2 = call_responses(
                client,
                model=args.model,
                system=PHASE2_SYSTEM,
                user=u2,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens_phase2,
                reasoning_effort=re_eff,
            )
            opt, r_after = parse_model_response(raw2)

            if not opt:
                fix = (
                    "FORMAT FIX. Output EXACTLY these two lines and nothing else:\n"
                    "OPTION: A|B|C|D\n"
                    "REASON: one or two sentences\n\n"
                    + u2
                )
                raw2b = call_responses(
                    client,
                    model=args.model,
                    system=PHASE2_SYSTEM,
                    user=fix,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens_phase2,
                    reasoning_effort=re_eff,
                )
                opt, r_after = parse_model_response(raw2b)

            r_before = re.sub(r"\s+", " ", r_before).strip()
            r_after = re.sub(r"\s+", " ", r_after).strip()
            if len(r_before) > 4000:
                r_before = r_before[:3997] + "..."
            if len(r_after) > 800:
                r_after = r_after[:797] + "..."

            dt = time.time() - t0
            w.writerow([q.number, r_before, opt, r_after])
            f.flush()
            print(f"Q{q.number} -> {opt or '?'} ({dt:.2f}s)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
