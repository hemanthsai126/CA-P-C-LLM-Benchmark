#!/usr/bin/env python3
"""
Run multiple-choice questions (from a plain-text file) through a local Ollama model
and write a 3-column CSV:

  question_number, option(A/B/C/D), reason

Important:
- This script reads ONLY the provided questions file.
- It does NOT read any answer key files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Option = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]


_Q_START = re.compile(r"^(\d+)\.\s*(.*)$")
_CHOICE = re.compile(r"^([A-D])\.\s*(.*)$")
_FIRST_LETTER = re.compile(r"\b([A-D])\b")


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
        "CRITICAL: Start your output with the text 'OPTION:' on the first line.",
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

    # Prefer explicit OPTION/ANSWER lines (avoid grabbing 'A' from words like 'And/Also/At').
    for ln in lines[:12]:
        up = ln.upper()
        if up.startswith("OPTION:") or up.startswith("ANSWER:"):
            m = re.search(r"\b([A-D])\b", up)
            if m:
                opt = m.group(1)
        if up.startswith("REASON:"):
            reason = ln.split(":", 1)[-1].strip()

    # If the model didn't follow the format, do NOT guess the option letter from the body,
    # because the prompt itself contains "A. ...", which creates false positives.
    if not reason:
        # Best-effort: use the whole response as the reason so you can inspect it.
        reason = t or "PARSE_FAILED: empty response"
    return opt, reason


def build_format_fix_prompt(q: MCQ, prior_response: str) -> str:
    """
    Second attempt prompt: force strict output (2 lines) and explicitly disallow blank answers.
    """
    prior = prior_response.replace("\x00", "").strip()
    lines = [
        "FORMAT FIX. You MUST follow the output format exactly.",
        "You MUST choose one letter A/B/C/D. Do NOT leave it blank.",
        "Output EXACTLY 2 lines and nothing else:",
        "OPTION: A|B|C|D",
        "REASON: 5-6 sentences",
        "",
        f"Question {q.number}: {q.stem}",
    ]
    for k in ("A", "B", "C", "D"):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    if prior:
        lines += ["", "Your previous response (for context):", prior[:1500]]
    return "\n".join(lines)


def force_pick_letter_from_text(text: str) -> str:
    """
    Last-resort fallback: if the model still won't format, pick a letter from the response text.
    This is only used to ensure the output CSV never has an empty option.
    """
    t = text.replace("\x00", "").upper()
    m = re.search(r"\b([A-D])\b", t)
    return m.group(1) if m else "A"


def _ollama_generate_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
) -> dict:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": int(num_predict), "num_ctx": int(num_ctx)},
    }
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def ollama_generate_with_retry(*, base_url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Normal models return text in `response`.
    DeepSeek-R1 may return empty `response` early; it may put interim text in `thinking`.
    """
    if "deepseek-r1" in model:
        first_num_predict, retry_num_predict, num_ctx = 48, 128, 256
    elif model.startswith("qwen3:"):
        # qwen3:14b can be extremely slow; keep responses tightly bounded.
        first_num_predict, retry_num_predict, num_ctx = 64, 96, 128
    else:
        first_num_predict, retry_num_predict, num_ctx = 96, 192, 512

    try:
        data1 = _ollama_generate_once(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=first_num_predict,
            num_ctx=num_ctx,
        )
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"Ollama HTTP error {e.code}: {detail}".strip()) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Ollama at {base_url}: {e.reason}") from e

    resp1 = str(data1.get("response", "")).replace("\x00", "").strip()
    think1 = str(data1.get("thinking", "")).replace("\x00", "").strip()
    # For qwen3 models on Ollama, the output may appear only in `thinking`.
    if resp1:
        return resp1
    if think1:
        return think1

    done_reason = str(data1.get("done_reason", "")).lower()
    if done_reason in ("length", "max_tokens", "max_length") or data1.get("thinking"):
        data2 = _ollama_generate_once(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=retry_num_predict,
            num_ctx=num_ctx,
        )
        resp2 = str(data2.get("response", "")).replace("\x00", "").strip()
        if resp2:
            return resp2
        think2 = str(data2.get("thinking", "")).replace("\x00", "").strip()
        if think2:
            return think2
        return resp2

    return resp1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, type=Path, help="Path to questions.txt (MCQs only)")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. mistral:7b")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of questions (0 = all)")
    args = ap.parse_args()

    if args.questions.name.lower().endswith("answers.txt"):
        raise SystemExit("Refusing to run: --questions points at answers.txt")

    qs = parse_questions(args.questions)
    if not qs:
        print(f"No questions parsed from {args.questions}", file=sys.stderr)
        return 2

    if args.limit and args.limit > 0:
        qs = qs[: args.limit]

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
            resp = ollama_generate_with_retry(base_url=args.ollama, model=args.model, prompt=prompt, temperature=0.0)
            dt = time.time() - t0

            opt, reason = parse_model_response(resp)
            if opt not in ("A", "B", "C", "D"):
                # Retry once with a strict format-fix prompt.
                fix_prompt = build_format_fix_prompt(q, resp)
                t1 = time.time()
                resp2 = ollama_generate_with_retry(
                    base_url=args.ollama, model=args.model, prompt=fix_prompt, temperature=0.0
                )
                dt += time.time() - t1
                opt, reason = parse_model_response(resp2)
                # If still missing, force a letter so nothing is blank.
                if opt not in ("A", "B", "C", "D"):
                    opt = force_pick_letter_from_text(resp2 or resp or "")
                    if reason:
                        reason = f"[FORCED_OPTION] {reason}"
                    else:
                        reason = "[FORCED_OPTION] Model did not provide OPTION line."

            reason = reason.replace("\x00", "")
            reason = re.sub(r"\s+", " ", reason).strip()
            if len(reason) > 500:
                reason = reason[:497] + "..."

            w.writerow([q.number, opt, reason])
            f.flush()
            print(f"Q{q.number} -> {opt or '?'} ({dt:.2f}s)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
