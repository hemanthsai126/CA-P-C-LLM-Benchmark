#!/usr/bin/env python3
"""
Repair a formatted MCQ file using two source PDFs + OpenAI (default: gpt-5.5).

Fixes broken options (e.g. OCR splits one choice across B/C), wrong answer keys,
and incomplete phrases while **preserving intact wording** where it is already
correct—merge fragments into single A/B/C/D lines instead of inventing new text.

Overwrites:
  - results/from_quizlet_pdfs/questions_formatted_165.txt
  - results/from_quizlet_pdfs/answers.txt

Default model is **gpt-5.5** (Responses API + ``reasoning``). Use ``--reasoning-effort off`` for models
that reject the reasoning field (e.g. gpt-4.1).

Example:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/repair_quizlet_mcqs_from_pdfs_openai.py

Answers only (rewrite ``answers.txt`` from current questions + PDFs; leave questions file unchanged):
  .venv/bin/python3 scripts/repair_quizlet_mcqs_from_pdfs_openai.py --answers-only

If a batch returns duplicate A–D text, a follow-up GPT call replaces duplicate distractors.

Auth: OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI
from pypdf import PdfReader

_Q_START = re.compile(r"^(\d+)\.\s*(.*)$")
_CHOICE = re.compile(r"^([A-D])\.\s*(.*)$")


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]


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


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_handbook_context(pdf: Path, max_chars: int) -> str:
    reader = PdfReader(str(pdf))
    n = len(reader.pages)
    if n == 0:
        return ""
    head = list(range(0, min(55, n)))
    mid_start = max(0, n // 3)
    mid = list(range(mid_start, min(mid_start + 45, n)))
    tail = list(range(max(0, n - 45), n))
    idxs = sorted(set(head + mid + tail))
    parts: list[str] = []
    for i in idxs:
        t = reader.pages[i].extract_text() or ""
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < 40:
            continue
        parts.append(f"--- PDF page {i + 1} ---\n{t}")
    blob = "\n\n".join(parts)
    return blob[:max_chars]


def combined_reference_context(pdfs: list[Path], max_chars: int) -> str:
    pdfs = [p for p in pdfs if p.is_file()]
    if not pdfs:
        return ""
    per = max(8_000, max_chars // len(pdfs))
    chunks: list[str] = []
    for i, p in enumerate(pdfs):
        body = extract_handbook_context(p, per)
        chunks.append(f"=== Reference PDF {i + 1}: {p.name} ===\n{body}")
    out = "\n\n".join(chunks)
    return out[:max_chars]


def _response_text(resp: Any) -> str:
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        parts: list[str] = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
        text = "".join(parts).strip()
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON object in model output") from None
        return json.loads(m.group(0))


def _responses_create_safe(client: OpenAI, req: dict[str, Any]) -> Any:
    """gpt-5.x etc. may reject ``temperature`` on the Responses API."""
    try:
        return client.responses.create(**req)
    except BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg and "not supported" in msg and "temperature" in req:
            req2 = {k: v for k, v in req.items() if k != "temperature"}
            return client.responses.create(**req2)
        raise


def _openai_json(
    client: OpenAI,
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        req["reasoning"] = {"effort": reasoning_effort}
    resp = _responses_create_safe(client, req)
    return _parse_json_object(_response_text(resp))


def fix_duplicate_options_openai(
    client: OpenAI,
    *,
    model: str,
    reference: str,
    item: dict[str, Any],
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    """
    When A–D are not four distinct strings, ask the model to replace duplicate(s)
    with new plausible wrong answers while preserving the keyed correct answer when possible.
    """
    ref = reference[:16_000]
    sysm = (
        "California P&C licensing MCQ repair. Two or more options have identical or empty text. "
        "Return JSON ONLY: {\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\",\"answer\":\"A\"|\"B\"|\"C\"|\"D\"} "
        "with FOUR distinct, non-empty strings (trimmed). "
        "Keep the best existing wording for the letter given in `answer` unless it is factually wrong. "
        "For every other letter whose text duplicates another, replace that text with a NEW plausible "
        "wrong answer consistent with the stem (do not copy another option's text). "
        "Output JSON only."
    )
    blob = json.dumps(
        {"number": item["number"], "stem": item["stem"], "A": item["A"], "B": item["B"], "C": item["C"], "D": item["D"], "answer": item["answer"]},
        ensure_ascii=False,
        indent=2,
    )
    user = f"REFERENCE EXCERPTS (optional):\n{ref}\n\n---\n\nITEM JSON:\n{blob}\n"
    for attempt in range(3):
        data = _openai_json(
            client,
            model=model,
            system=sysm,
            user=user + (f"\n(attempt {attempt + 2}: still need 4 distinct options.)\n" if attempt else ""),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        a = _squish(str(data.get("A", "")))
        b = _squish(str(data.get("B", "")))
        c = _squish(str(data.get("C", "")))
        d = _squish(str(data.get("D", "")))
        ans = str(data.get("answer", "")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            continue
        if min(len(a), len(b), len(c), len(d)) < 2:
            continue
        if len({a, b, c, d}) == 4:
            out = dict(item)
            out["A"], out["B"], out["C"], out["D"], out["answer"] = a, b, c, d, ans
            return out
    raise ValueError(f"Q{item['number']}: could not resolve duplicate options after retries")


def call_repair_batch(
    client: OpenAI,
    *,
    model: str,
    reference: str,
    batch: list[MCQ],
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> list[dict[str, Any]]:
    payload = []
    for q in batch:
        payload.append(
            {
                "number": q.number,
                "stem": q.stem,
                "A": q.choices.get("A", ""),
                "B": q.choices.get("B", ""),
                "C": q.choices.get("C", ""),
                "D": q.choices.get("D", ""),
            }
        )
    sysm = (
        "You repair California property & casualty licensing-style multiple-choice questions. "
        "The items were extracted from PDFs; some choices are OCR/split errors (e.g. one idea broken across "
        "B and C, or a phrase split mid-line). "
        "Rules:\n"
        "1) Return EXACTLY one JSON object: {\"items\":[...]} with the SAME count and `number` values as the input batch, same order.\n"
        "2) Each item: {\"number\":int,\"stem\":str,\"A\":str,\"B\":str,\"C\":str,\"D\":str,\"answer\":\"A\"|\"B\"|\"C\"|\"D\"}.\n"
        "3) Preserve original wording whenever it is already correct and complete—do not paraphrase for style. "
        "When a choice was split across letters, **merge into a single contiguous line** for the correct letter; "
        "do not leave dangling fragments; do not split one intact clause into two options.\n"
        "4) All four options must be substantive, parallel, and mutually distinct (no empty duplicates).\n"
        "5) Choose `answer` using the reference PDF excerpts + standard California P&C exam logic. "
        "If the PDFs are silent, use best professional judgment consistent with the stem.\n"
        "6) Output JSON only—no markdown fences, no commentary."
    )
    user = (
        "REFERENCE PDF EXCERPTS (may be partial; use for fact-checking and wording):\n\n"
        + reference
        + "\n\n---\n\nINPUT BATCH (repair each):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    data = _openai_json(
        client,
        model=model,
        system=sysm,
        user=user,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("missing items array")
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        n = int(it["number"])
        stem = _squish(str(it.get("stem", "")))
        a = _squish(str(it.get("A", "")))
        b = _squish(str(it.get("B", "")))
        c = _squish(str(it.get("C", "")))
        d = _squish(str(it.get("D", "")))
        ans = str(it.get("answer", "")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            raise ValueError(f"Q{n} bad answer {ans!r}")
        if min(len(stem), len(a), len(b), len(c), len(d)) < 3:
            raise ValueError(f"Q{n} too short fields")
        row = {"number": n, "stem": stem, "A": a, "B": b, "C": c, "D": d, "answer": ans}
        if len({a, b, c, d}) < 4:
            print(f"note: Q{n} duplicate options — follow-up fix", file=sys.stderr)
            row = fix_duplicate_options_openai(
                client,
                model=model,
                reference=reference,
                item=row,
                temperature=temperature,
                max_output_tokens=min(4096, max_output_tokens),
                reasoning_effort=reasoning_effort,
            )
        out.append(row)
    nums = [x["number"] for x in out]
    if sorted(nums) != sorted(q.number for q in batch):
        raise ValueError(f"number mismatch: expected {[q.number for q in batch]} got {nums}")
    return out


def call_answer_key_batch(
    client: OpenAI,
    *,
    model: str,
    reference: str,
    batch: list[MCQ],
    temperature: float,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> list[dict[str, Any]]:
    """Return [{number, answer}, ...] for each question in batch (GPT picks best letter)."""
    payload = []
    for q in batch:
        payload.append(
            {
                "number": q.number,
                "stem": q.stem,
                "A": q.choices.get("A", ""),
                "B": q.choices.get("B", ""),
                "C": q.choices.get("C", ""),
                "D": q.choices.get("D", ""),
            }
        )
    sysm = (
        "You are a California property & casualty licensing exam expert. "
        "For each multiple-choice item, pick the single best letter A, B, C, or D using the stem, "
        "the four options, and the reference excerpts when helpful. "
        "Return JSON ONLY: {\"items\":[{\"number\":int,\"answer\":\"A\"|\"B\"|\"C\"|\"D\"}, ...]} "
        "with the SAME count and `number` values as the input, same order. No other keys."
    )
    user = (
        "REFERENCE EXCERPTS:\n\n"
        + reference[: min(len(reference), 48_000)]
        + "\n\n---\n\nQUESTIONS (pick one letter each):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    data = _openai_json(
        client,
        model=model,
        system=sysm,
        user=user,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("missing items array")
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        n = int(it["number"])
        ans = str(it.get("answer", "")).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            raise ValueError(f"Q{n} bad answer {ans!r}")
        out.append({"number": n, "answer": ans})
    nums = [x["number"] for x in out]
    if sorted(nums) != sorted(q.number for q in batch):
        raise ValueError("number mismatch in answer-key batch")
    return out


def write_outputs(items: list[dict[str, Any]], q_path: Path, a_path: Path, *, header_note: str) -> None:
    q_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        fa.write(header_note.rstrip() + "\n")
        for it in items:
            n = it["number"]
            fq.write(f"{n}. {_squish(it['stem'])}\n")
            for letter in ("A", "B", "C", "D"):
                fq.write(f"{letter}. {_squish(it[letter])}\n")
            fq.write("\n")
            fa.write(f"{n} {it['answer']}\n")


def write_answers_only(path: Path, ordered: list[tuple[int, str]], *, header_note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(header_note.rstrip() + "\n")
        for n, letter in ordered:
            f.write(f"{n} {letter}\n")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--questions-in",
        type=Path,
        default=root / "results/from_quizlet_pdfs/questions_formatted_165.txt",
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        action="append",
        default=None,
        help="Reference PDF (pass twice for two files). Defaults to CA INS QA.pdf and INS QA.pdf under data/eval_set/from_quizlet_pdfs/.",
    )
    ap.add_argument(
        "--out-questions",
        type=Path,
        default=root / "results/from_quizlet_pdfs/questions_formatted_165.txt",
    )
    ap.add_argument(
        "--out-answers",
        type=Path,
        default=root / "results/from_quizlet_pdfs/answers.txt",
    )
    ap.add_argument("--model", type=str, default="gpt-5.5", help="OpenAI model (default: gpt-5.5).")
    ap.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        help='Responses API reasoning effort for gpt-5.x (low, medium, high). Use "off" to omit.',
    )
    ap.add_argument("--batch-size", type=int, default=7)
    ap.add_argument("--ref-chars", type=int, default=72_000, help="Max chars of PDF excerpts per API call.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=16_384,
        help="Per-call max output tokens (reasoning models need headroom).",
    )
    ap.add_argument("--sleep-ms", type=int, default=400)
    ap.add_argument("--no-backup", action="store_true", help="Do not copy *.bak_<timestamp> before overwriting outputs.")
    ap.add_argument(
        "--answers-only",
        action="store_true",
        help="Do not rewrite questions; only write answers.txt using GPT from current --questions-in.",
    )
    args = ap.parse_args()

    re_eff: str | None = args.reasoning_effort.strip()
    if re_eff.lower() == "off":
        re_eff = None

    if not os.environ.get("OPENAI_API_KEY"):
        print("error: set OPENAI_API_KEY", file=sys.stderr)
        return 2

    pdfs: list[Path]
    if args.pdf:
        pdfs = list(args.pdf)
    else:
        base = root / "data/eval_set/from_quizlet_pdfs"
        pdfs = [
            base / "CA INS QA.pdf",
            base / "INS QA.pdf",
        ]

    for p in pdfs:
        if not p.is_file():
            print(f"error: PDF not found: {p}", file=sys.stderr)
            return 2

    qs = parse_questions(args.questions_in)
    if not qs:
        print("error: no questions parsed", file=sys.stderr)
        return 2

    client = OpenAI()
    reference = combined_reference_context(pdfs, args.ref_chars)
    if len(reference) < 500:
        print("error: very little text extracted from PDFs; check files/encryption", file=sys.stderr)
        return 2

    pdf_names = ", ".join(p.name for p in pdfs)
    batch_size = max(1, args.batch_size)

    if args.answers_only:
        merged_ans: dict[int, str] = {}
        for i in range(0, len(qs), batch_size):
            batch = qs[i : i + batch_size]
            print(
                f"answers batch {i // batch_size + 1}: Q{batch[0].number}–Q{batch[-1].number} ({len(batch)} items)",
                file=sys.stderr,
            )
            rows = call_answer_key_batch(
                client,
                model=args.model,
                reference=reference,
                batch=batch,
                temperature=args.temperature,
                max_output_tokens=min(8192, args.max_output_tokens),
                reasoning_effort=re_eff,
            )
            for row in rows:
                merged_ans[row["number"]] = row["answer"]
            time.sleep(args.sleep_ms / 1000.0)
        pairs = [(q.number, merged_ans[q.number]) for q in qs]
        header = (
            f"# Answers for question IDs 1–{len(pairs)} (aligned with questions_formatted_165.txt). "
            f"Answer key from OpenAI {args.model} using PDF excerpts: {pdf_names}. Questions file unchanged."
        )
        if not args.no_backup and args.out_answers.is_file():
            ts = time.strftime("%Y%m%d_%H%M%S")
            bak = args.out_answers.with_suffix(args.out_answers.suffix + f".bak_{ts}")
            shutil.copy2(args.out_answers, bak)
            print(f"backup: {bak}", file=sys.stderr)
        write_answers_only(args.out_answers, pairs, header_note=header)
        print(f"Wrote {args.out_answers}", file=sys.stderr)
        return 0

    merged: dict[int, dict[str, Any]] = {}
    for i in range(0, len(qs), batch_size):
        batch = qs[i : i + batch_size]
        print(f"batch {i // batch_size + 1}: Q{batch[0].number}–Q{batch[-1].number} ({len(batch)} items)", file=sys.stderr)
        items = call_repair_batch(
            client,
            model=args.model,
            reference=reference,
            batch=batch,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=re_eff,
        )
        for it in items:
            merged[it["number"]] = it
        time.sleep(args.sleep_ms / 1000.0)

    ordered = [merged[q.number] for q in qs]
    header = (
        f"# Answers for question IDs 1–{len(ordered)} (aligned with questions_formatted_165.txt). "
        f"Repaired with OpenAI {args.model} using PDF excerpts: {pdf_names}."
    )
    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (args.out_questions, args.out_answers):
            if p.is_file():
                bak = p.with_suffix(p.suffix + f".bak_{ts}")
                shutil.copy2(p, bak)
                print(f"backup: {bak}", file=sys.stderr)
    write_outputs(ordered, args.out_questions, args.out_answers, header_note=header)
    print(f"Wrote {args.out_questions}", file=sys.stderr)
    print(f"Wrote {args.out_answers}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
