#!/usr/bin/env python3
"""
Remove duplicate MCQs from a questions.txt (block format) using:
  1) Exact / near-exact stem dedupe (normalized text).
  2) OpenAI **small** model (default: gpt-4o-mini) to propose semantic duplicate groups.

Writes:
  - Updated --questions (with backup *.bak_<timestamp> unless --no-backup)
  - Optional --answers rewritten with same kept items, renumbered 1..N
  - --issues-out: human review list (incomplete / nonsensical / broken MCQs) — NOT auto-deleted

Auth: export OPENAI_API_KEY=...

Example:
  .venv/bin/python3 scripts/dedupe_questions_openai.py \\
    --questions eval_set/from_youtube_video/questions.txt \\
    --answers eval_set/from_youtube_video/answers.txt \\
    --issues-out eval_set/from_youtube_video/questions_issues_report.txt

  .venv/bin/python3 scripts/dedupe_questions_openai.py \\
    --questions eval_set/from_quizlet_pdfs/questions.txt \\
    --issues-out eval_set/from_quizlet_pdfs/questions_issues_report.txt
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

_Q_START = re.compile(r"^(\d+)\.\s*(.*)$")
_CHOICE = re.compile(r"^([A-D])\.\s*(.*)$")


@dataclass
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


def _norm_stem(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def exact_stem_dupes(mcqs: list[MCQ]) -> set[int]:
    """Later duplicate IDs (same normalized stem) are dropped; first occurrence kept."""
    seen: dict[str, int] = {}
    drop: set[int] = set()
    for q in mcqs:
        k = _norm_stem(q.stem)
        if not k:
            continue
        if k in seen:
            drop.add(q.number)
        else:
            seen[k] = q.number
    return drop


def _merge_duplicate_groups(groups: list[list[int]]) -> set[int]:
    """For each merged component, keep min ID; return IDs to drop."""
    sets = [set(g) for g in groups if len(g) >= 2]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(sets):
            j = i + 1
            while j < len(sets):
                if sets[i] & sets[j]:
                    sets[i] |= sets[j]
                    sets.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1
    drop: set[int] = set()
    for s in sets:
        if len(s) < 2:
            continue
        keep = min(s)
        for x in s:
            if x != keep:
                drop.add(x)
    return drop


def call_openai_review(
    client: OpenAI,
    *,
    model: str,
    mcqs: list[MCQ],
    temperature: float,
    max_output_tokens: int,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    lines = []
    for q in mcqs:
        opts = " | ".join(f"{k}:{(q.choices.get(k) or '')[:120]}" for k in ("A", "B", "C", "D"))
        stem = re.sub(r"\s+", " ", q.stem)[:280]
        lines.append(f"ID {q.number} | {stem} | {opts}")
    blob = "\n".join(lines)

    sysm = (
        "You review California P&C licensing-style multiple-choice items. "
        "Return JSON ONLY with keys duplicate_groups (array of arrays of integer IDs) and "
        "problematic (array of objects {question_number, reason}). "
        "duplicate_groups: each inner array lists IDs that are the SAME or near-duplicate question "
        "(rephrased stem, same fact tested). Merge overlapping groups mentally; each ID at most once across groups. "
        "problematic: incomplete stems, nonsense options, missing/blank options, garbled OCR-only garbage, or not a real MCQ. "
        "Do NOT mark items as problematic only for typos if the question is still answerable."
    )
    user = f"MCQ list (ID | stem | A|B|C|D previews):\n\n{blob}\n"

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

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise ValueError("No JSON in model output")
        data = json.loads(m.group(0))

    groups_raw = data.get("duplicate_groups") or []
    prob_raw = data.get("problematic") or []

    groups: list[list[int]] = []
    for g in groups_raw:
        if not isinstance(g, list):
            continue
        ids = [int(x) for x in g if str(x).isdigit() or isinstance(x, int)]
        ids = sorted(set(ids))
        if len(ids) >= 2:
            groups.append(ids)

    problematic: list[dict[str, Any]] = []
    for p in prob_raw:
        if not isinstance(p, dict):
            continue
        try:
            n = int(p.get("question_number"))
            r = str(p.get("reason", "")).strip()
            if r:
                problematic.append({"question_number": n, "reason": r})
        except Exception:
            continue

    return groups, problematic


def write_mcqs(path: Path, mcqs: list[MCQ]) -> None:
    def squish(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(mcqs, start=1):
            f.write(f"{i}. {squish(q.stem)}\n")
            for k in ("A", "B", "C", "D"):
                f.write(f"{k}. {squish(q.choices.get(k, ''))}\n")
            f.write("\n")


def parse_answers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s+([ABCD])\s*$", line, re.I)
        if m:
            out[int(m.group(1))] = m.group(2).upper()
    return out


def write_answers(path: Path, pairs: list[tuple[int, str]], header: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write(header.rstrip() + "\n")
        for n, letter in pairs:
            f.write(f"{n} {letter}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--answers", type=Path, default=None, help="Optional answers.txt to rewrite in lockstep")
    ap.add_argument("--issues-out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="Small model; not the gpt-4.1 synthetic generator.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-output-tokens", type=int, default=8000)
    ap.add_argument("--sleep-ms", type=int, default=200)
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; do not call API or write files")
    args = ap.parse_args()

    mcqs = parse_questions(args.questions)
    if not mcqs:
        raise SystemExit(f"No questions parsed from {args.questions}")

    ids_all = {q.number for q in mcqs}
    drop_exact = exact_stem_dupes(mcqs)

    if args.dry_run:
        print(f"Parsed {len(mcqs)} blocks; exact-stem drops: {sorted(drop_exact)}", file=sys.stderr)
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    client = OpenAI()
    survivors = [q for q in mcqs if q.number not in drop_exact]
    groups, problematic = call_openai_review(
        client,
        model=args.model,
        mcqs=survivors,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    if args.sleep_ms:
        time.sleep(args.sleep_ms / 1000.0)

    drop_semantic = _merge_duplicate_groups(groups)
    drop_semantic = {d for d in drop_semantic if d in ids_all and d not in drop_exact}
    drop_all = set(drop_exact) | set(drop_semantic)

    kept = [q for q in mcqs if q.number not in drop_all]
    old_to_new = {q.number: i for i, q in enumerate(kept, start=1)}

    report_lines = [
        f"# Question review report",
        f"# Source: {args.questions}",
        f"# Model: {args.model}",
        f"# Parsed: {len(mcqs)} | Removed as duplicate: {len(drop_all)} | Kept: {len(kept)}",
        "",
        "## Removed duplicate IDs (not listed in issues — safe auto-removal)",
        "",
    ]
    if drop_exact:
        report_lines.append("### Exact normalized-stem duplicates (dropped later ID)")
        report_lines.append(", ".join(str(x) for x in sorted(drop_exact)))
        report_lines.append("")
    if drop_semantic:
        report_lines.append("### Semantic duplicate groups (dropped non-min ID per group)")
        report_lines.append(", ".join(str(x) for x in sorted(drop_semantic)))
        report_lines.append("")
    report_lines.append("## Flagged for manual review (still kept in questions.txt until you delete)")
    report_lines.append("")
    if not problematic:
        report_lines.append("(none)")
    else:
        for p in sorted(problematic, key=lambda x: x["question_number"]):
            oid = p["question_number"]
            nid = old_to_new.get(oid, "(dropped as duplicate)")
            report_lines.append(f"- Old ID {oid} → New ID {nid}: {p['reason']}")
    report_lines.append("")
    report_lines.append("## Model duplicate_groups (raw)")
    report_lines.append(json.dumps(groups, indent=2))

    args.issues_out.parent.mkdir(parents=True, exist_ok=True)
    args.issues_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = args.questions.with_suffix(args.questions.suffix + f".bak_{ts}")
        shutil.copy2(args.questions, bak)
        print(f"Backed up questions -> {bak}", file=sys.stderr)
        if args.answers and args.answers.is_file():
            bak_a = args.answers.with_suffix(args.answers.suffix + f".bak_{ts}")
            shutil.copy2(args.answers, bak_a)
            print(f"Backed up answers -> {bak_a}", file=sys.stderr)

    write_mcqs(args.questions, kept)

    if args.answers and args.answers.is_file():
        ans = parse_answers(args.answers)
        header_lines = []
        body_lines = []
        for raw in args.answers.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.strip().startswith("#"):
                header_lines.append(raw.rstrip())
            else:
                body_lines.append(raw)
        header = "\n".join(header_lines) if header_lines else None
        pairs: list[tuple[int, str]] = []
        for q in kept:
            old = q.number
            if old not in ans:
                print(f"WARN: no answer for kept question old_id={old}", file=sys.stderr)
                continue
            pairs.append((old_to_new[old], ans[old]))
        write_answers(args.answers, pairs, header or "# Renumbered after dedupe\n")

    print(f"Wrote {len(kept)} questions -> {args.questions}", file=sys.stderr)
    print(f"Wrote issues -> {args.issues_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
