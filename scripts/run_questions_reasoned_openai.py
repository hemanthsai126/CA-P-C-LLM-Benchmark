#!/usr/bin/env python3
"""
OpenAI version of the questions-only runner.

Reads ONLY a questions file (e.g. eval_set/from_youtube_video/questions.txt) and writes a CSV with:
  question_number, option, reason

The script does NOT read answers.txt.

Auth:
  export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import BadRequestError, OpenAI

Option = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]


_Q_START = re.compile(r"^(\d+)\.\s*(.*)$")
_CHOICE = re.compile(r"^([A-D])\.\s*(.*)$")


def _split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def parse_questions(path: Path) -> list[MCQ]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _split_blocks(text)
    out: list[MCQ] = []

    for b in blocks:
        m = _Q_START.match(b[0])
        if not m:
            continue
        n = int(m.group(1))
        stem = m.group(2).strip()
        choices: dict[str, str] = {}
        for line in b[1:]:
            cm = _CHOICE.match(line)
            if not cm:
                if not choices:
                    stem = (stem + " " + line).strip()
                continue
            choices[cm.group(1)] = cm.group(2).strip()
        out.append(MCQ(number=n, stem=stem, choices=choices))

    out.sort(key=lambda q: q.number)
    return out


def build_prompt(q: MCQ) -> str:
    lines = [
        "You are a California P&C licensing exam tutor.",
        "Pick the best option letter and give a clear explanation.",
        "Do NOT write step-by-step reasoning. Give a straight explanation in 2-3 sentences.",
        "You MUST choose exactly one option (A, B, C, or D). You may NOT refuse and you may NOT leave it blank.",
        "Do NOT write anything except the required 2 lines.",
        "",
        "Output format (EXACTLY 2 lines, no extra text):",
        "OPTION: A|B|C|D",
        "REASON: 2-3 sentences explaining why that option is correct",
        "",
        f"Question {q.number}: {q.stem}",
    ]
    for k in ("A", "B", "C", "D"):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    return "\n".join(lines)


def parse_model_response(text: str) -> tuple[str, str]:
    t = text.replace("\x00", "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    opt = ""
    reason = ""

    for ln in lines[:12]:
        up = ln.upper()
        if up.startswith("OPTION:") or up.startswith("ANSWER:"):
            m = re.search(r"\b([A-D])\b", up)
            if m:
                opt = m.group(1)
        if up.startswith("REASON:"):
            reason = ln.split(":", 1)[-1].strip()

    if not reason:
        reason = t or "PARSE_FAILED: empty response"
    if opt not in ("A", "B", "C", "D"):
        opt = ""
    return opt, reason


def call_openai(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    # Use Responses API (preferred in openai>=1.x). Some models reject ``temperature``.
    req = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    try:
        resp = client.responses.create(**req)
    except BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg and "not supported" in msg:
            req.pop("temperature", None)
            resp = client.responses.create(**req)
        else:
            raise
    # Collect any text output segments.
    out_parts: list[str] = []
    for item in resp.output:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) == "output_text":
                out_parts.append(getattr(c, "text", "") or "")
    return "".join(out_parts).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, type=Path, help="Path to questions.txt (MCQs only)")
    ap.add_argument("--model", required=True, help="OpenAI model name, e.g. gpt-4.1-mini")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of questions (0 = all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=220)
    args = ap.parse_args()

    if args.questions.name.lower().endswith("answers.txt"):
        raise SystemExit("Refusing to run: --questions points at answers.txt")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

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
        w.writerow(["question_number", "option", "reason"])
        f.flush()

        for q in qs:
            present = [k for k in ("A", "B", "C", "D") if q.choices.get(k)]
            if len(present) < 2:
                w.writerow([q.number, "", f"SKIPPED: malformed question (found choices: {present})"])
                f.flush()
                continue

            prompt = build_prompt(q)
            t0 = time.time()
            raw = call_openai(
                client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            dt = time.time() - t0

            opt, reason = parse_model_response(raw)

            # If it didn't follow the format, force a second attempt with a strict reminder.
            if not opt:
                fix = (
                    "FORMAT FIX. Output EXACTLY 2 lines and nothing else:\n"
                    "OPTION: A|B|C|D\n"
                    "REASON: 2-3 sentences\n\n"
                    + prompt
                )
                raw2 = call_openai(
                    client,
                    model=args.model,
                    prompt=fix,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
                opt, reason = parse_model_response(raw2)

            reason = re.sub(r"\s+", " ", reason).strip()
            if len(reason) > 500:
                reason = reason[:497] + "..."

            w.writerow([q.number, opt, reason])
            f.flush()
            print(f"Q{q.number} -> {opt or '?'} ({dt:.2f}s)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

