#!/usr/bin/env python3
"""
Generate synthetic California P&C-style MCQs (medium difficulty) using OpenAI,
balanced so each curriculum bucket has the same count (default 60 × 8 = 480).

Uses excerpts from ``Insurance_Handbook_20103.pdf`` as style/topic reference
(do not copy long verbatim passages).

Writes:
  - questions file: same block format as questions_formatted.txt
  - answers file: ``N LETTER`` lines (optional ``#`` header)

Example:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/generate_synthetic_handbook_mcqs.py \\
    --handbook \"source_material/Cali Data/Insurance_Handbook_20103.pdf\" \\
    --out-questions eval_set/from_quizlet_pdfs/questions_formatted.txt \\
    --out-answers eval_set/from_quizlet_pdfs/answers.txt \\
    --model gpt-4.1 \\
    --per-bucket 60 \\
    --batch-size 10

Backups of existing outputs are created by default; pass ``--no-backup`` to skip.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from pypdf import PdfReader

try:
    from openai import BadRequestError, OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


BUCKETS: list[tuple[int, str]] = [
    (1, "Basic concepts"),
    (2, "Contract law"),
    (3, "CA laws & ethics"),
    (4, "Tort & property fundamentals"),
    (5, "Residential property"),
    (6, "Valuation & government programs"),
    (7, "Personal liability & inland marine"),
    (8, "Commercial insurance"),
]


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())[:160]


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


def _parse_items_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON object in model output")
        data = json.loads(m.group(0))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("JSON missing 'items' array")
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        stem = str(it.get("stem", "")).strip()
        a = str(it.get("A", "")).strip()
        b = str(it.get("B", "")).strip()
        c = str(it.get("C", "")).strip()
        d = str(it.get("D", "")).strip()
        ans = str(it.get("answer", "")).strip().upper()
        if len(stem) < 25 or min(len(a), len(b), len(c), len(d)) < 8:
            continue
        if ans not in ("A", "B", "C", "D"):
            continue
        if len({a, b, c, d}) < 4:
            continue
        out.append({"stem": stem, "A": a, "B": b, "C": c, "D": d, "answer": ans})
    return out


def call_openai_batch(
    client: Any,
    *,
    model: str,
    handbook: str,
    bucket_id: int,
    bucket_name: str,
    need: int,
    avoid: list[str],
    temperature: float,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    avoid_blob = "\n".join(f"- {s[:200]}" for s in avoid[:40])
    sysm = (
        "You write original California property & casualty licensing style multiple-choice questions. "
        "Difficulty: medium — requires reasoning, not trivial definitions, but not obscure trick questions. "
        "Each item must be clearly answerable from general P&C knowledge; use the handbook excerpts only as "
        "topic/style reference, not for long verbatim copying. "
        "Return JSON ONLY: {\"items\":[{\"stem\":\"...\",\"A\":\"...\",\"B\":\"...\",\"C\":\"...\",\"D\":\"...\",\"answer\":\"A|B|C|D\"}]} "
        f"Include exactly {need} items. All items must belong to this ONE topic bucket: {bucket_id} = {bucket_name}. "
        "Four distinct options; one correct letter."
    )
    user = (
        f"Topic bucket (must match every item): {bucket_id} — {bucket_name}\n\n"
        f"Handbook excerpts (reference only):\n{handbook}\n\n"
        "Avoid near-duplicates of these stems (paraphrase differently):\n"
        f"{avoid_blob if avoid_blob.strip() else '(none yet)'}\n"
    )
    req: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": sysm},
            {"role": "user", "content": user},
        ],
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
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        parts: list[str] = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
        text = "".join(parts).strip()
    return _parse_items_json(text)


def write_outputs(items: list[dict[str, Any]], q_path: Path, a_path: Path) -> None:
    q_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        fa.write(
            "# Synthetic balanced set (60 per bucket × 8); medium difficulty; "
            "reference: Insurance Handbook excerpts.\n"
        )
        for n, it in enumerate(items, start=1):
            fq.write(f"{n}. {_squish(it['stem'])}\n")
            fq.write(f"A. {_squish(it['A'])}\n")
            fq.write(f"B. {_squish(it['B'])}\n")
            fq.write(f"C. {_squish(it['C'])}\n")
            fq.write(f"D. {_squish(it['D'])}\n")
            fq.write("\n")
            fa.write(f"{n} {it['answer']}\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--handbook",
        type=Path,
        default=Path("source_material/Cali Data/Insurance_Handbook_20103.pdf"),
    )
    ap.add_argument(
        "--out-questions",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/questions_formatted.txt"),
    )
    ap.add_argument(
        "--out-answers",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/answers.txt"),
    )
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--per-bucket", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--handbook-chars", type=int, default=110_000)
    ap.add_argument("--max-output-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--sleep-ms", type=int, default=250)
    ap.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not copy existing outputs to *.bak_TIMESTAMP before overwrite",
    )
    args = ap.parse_args()

    if OpenAI is None:
        raise SystemExit("openai package not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    if not args.handbook.is_file():
        raise SystemExit(f"Handbook PDF not found: {args.handbook}")

    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (args.out_questions, args.out_answers):
            if p.is_file():
                bak = p.with_suffix(p.suffix + f".bak_{ts}")
                shutil.copy2(p, bak)
                print(f"Backed up {p} -> {bak}", file=sys.stderr)

    hb = extract_handbook_context(args.handbook, args.handbook_chars)
    if len(hb) < 2000:
        raise SystemExit("Very little text extracted from handbook PDF; check file.")

    client = OpenAI()
    all_items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for bucket_id, bucket_name in BUCKETS:
        need_total = args.per_bucket
        got: list[dict[str, Any]] = []
        avoid: list[str] = []
        while len(got) < need_total:
            batch = min(args.batch_size, need_total - len(got))
            try:
                new = call_openai_batch(
                    client,
                    model=args.model,
                    handbook=hb,
                    bucket_id=bucket_id,
                    bucket_name=bucket_name,
                    need=batch,
                    avoid=avoid,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                )
            except Exception as e:
                print(f"ERROR bucket {bucket_id} batch: {e}", file=sys.stderr)
                raise
            added = 0
            for it in new:
                k = _norm_key(it["stem"])
                if k in seen:
                    continue
                seen.add(k)
                it["bucket"] = bucket_id
                got.append(it)
                avoid.append(it["stem"])
                added += 1
            print(
                f"bucket {bucket_id} ({bucket_name}): +{added} (total {len(got)}/{need_total})",
                file=sys.stderr,
            )
            if added == 0:
                raise SystemExit(f"No new items accepted in bucket {bucket_id}; aborting.")
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)
        all_items.extend(got)

    write_outputs(all_items, args.out_questions, args.out_answers)
    print(f"Wrote {len(all_items)} questions -> {args.out_questions}", file=sys.stderr)
    print(f"Wrote {len(all_items)} answers -> {args.out_answers}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
