#!/usr/bin/env python3
"""
Shrink an ordered-bucket MCQ set (e.g. 100×8 → 50×8) while keeping answers aligned.

Assumes global question IDs are contiguous blocks: bucket *b* uses IDs
``(b-1)*from_per_bucket + 1`` through ``b*from_per_bucket``.

Strategies:
  openai  — per bucket, ask the model for exactly ``to_per_bucket`` local indices (1..from_per_bucket)
            to keep for topic diversity (default when OPENAI_API_KEY is set).
  first50 — keep the first ``to_per_bucket`` questions in each bucket (no API).

Also rewrites ``question_buckets_generator_layout.csv`` when ``--layout-out`` is set.

Examples:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/trim_synthetic_per_bucket_openai.py \\
    --questions results/synthetic_data/questions_formatted.txt \\
    --answers results/synthetic_data/answers.txt \\
    --model gpt-4.1

  .venv/bin/python3 scripts/trim_synthetic_per_bucket_openai.py --strategy first50
"""

from __future__ import annotations

import argparse
import csv
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


def parse_answers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            out[int(parts[0])] = parts[1].strip().upper()
    return out


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


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
            raise ValueError("No JSON in model output") from None
        return json.loads(m.group(0))


def openai_pick_indices(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    bucket_id: int,
    bucket_name: str,
    mcqs: list[MCQ],
    from_n: int,
    to_n: int,
) -> list[int]:
    """Return sorted local indices (1..from_n) to keep, length to_n."""
    lines: list[str] = []
    for i, q in enumerate(mcqs, start=1):
        stem = _squish(q.stem)[:220]
        opts = " | ".join(
            f"{k}:{_squish(q.choices.get(k) or '')[:80]}" for k in ("A", "B", "C", "D")
        )
        lines.append(f"{i}\t{stem}\t{opts}")
    blob = "\n".join(lines)

    sysm = (
        "You curate California P&C licensing-style multiple-choice exams. "
        f"Bucket {bucket_id} ({bucket_name}): there are exactly {from_n} items numbered 1–{from_n} (first column). "
        f"Choose exactly {to_n} distinct integers from 1 to {from_n} to KEEP. "
        "Prefer a diverse mix of subtopics and difficulty; avoid keeping near-duplicate stems when you can. "
        'Return JSON ONLY: {"keep": [int, ...]} with exactly '
        f"{to_n} unique integers, sorted ascending."
    )

    user = f"Items (local_index TAB stem TAB A|B|C|D previews):\n\n{blob}\n"

    req: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": sysm},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    def _call() -> dict[str, Any]:
        try:
            resp = client.responses.create(**req)
        except BadRequestError as e:
            msg = str(e).lower()
            if "temperature" in msg and "not supported" in msg:
                req.pop("temperature", None)
                resp = client.responses.create(**req)
            else:
                raise
        return _parse_json_object(_response_text(resp))

    data = _call()
    raw = data.get("keep") or data.get("indices") or data.get("selected")
    if not isinstance(raw, list):
        raw = []

    picked: list[int] = []
    for x in raw:
        try:
            v = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= v <= from_n:
            picked.append(v)

    picked = sorted(set(picked))
    if len(picked) < to_n:
        for v in range(1, from_n + 1):
            if v not in picked:
                picked.append(v)
            if len(picked) >= to_n:
                break
    if len(picked) > to_n:
        picked = sorted(picked)[:to_n]
    return sorted(picked[:to_n])


def normalize_keep_indices(picked: list[int], *, from_n: int, to_n: int) -> list[int]:
    picked = sorted({i for i in picked if 1 <= i <= from_n})
    if len(picked) < to_n:
        for v in range(1, from_n + 1):
            if v not in picked:
                picked.append(v)
            if len(picked) >= to_n:
                break
    return sorted(picked)[:to_n]


def write_questions_and_answers(
    items: list[MCQ],
    answers_by_old: dict[int, str],
    q_path: Path,
    a_path: Path,
    *,
    per_bucket: int,
    header_note: str,
) -> None:
    q_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        fa.write(
            f"# Synthetic balanced set ({per_bucket} per bucket × 8 = {per_bucket * 8}); "
            f"{header_note}\n"
        )
        for n, q in enumerate(items, start=1):
            fq.write(f"{n}. {_squish(q.stem)}\n")
            for letter in ("A", "B", "C", "D"):
                fq.write(f"{letter}. {_squish(q.choices.get(letter) or '')}\n")
            fq.write("\n")
            letter = answers_by_old[q.number]
            fa.write(f"{n} {letter}\n")


def write_layout_csv(path: Path, *, per_bucket: int, model_label: str, notes_suffix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_number", "bucket", "bucket_name", "model", "notes"])
        for qn in range(1, per_bucket * 8 + 1):
            b = (qn - 1) // per_bucket + 1
            name = dict(BUCKETS).get(b, str(b))
            w.writerow(
                [
                    qn,
                    b,
                    name,
                    model_label,
                    f"ordered_block_per_bucket={per_bucket}{notes_suffix}",
                ]
            )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--questions",
        type=Path,
        default=root / "results/synthetic_data/questions_formatted.txt",
    )
    ap.add_argument(
        "--answers",
        type=Path,
        default=root / "results/synthetic_data/answers.txt",
    )
    ap.add_argument(
        "--layout-out",
        type=Path,
        default=root / "results/synthetic_data/question_buckets_generator_layout.csv",
        help="Write bucket layout CSV.",
    )
    ap.add_argument("--no-layout", action="store_true", help="Skip writing the layout CSV.")
    ap.add_argument("--from-per-bucket", type=int, default=100)
    ap.add_argument("--to-per-bucket", type=int, default=50)
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=4096)
    ap.add_argument("--sleep-ms", type=int, default=300)
    ap.add_argument(
        "--strategy",
        choices=("openai", "first50"),
        default=None,
        help="Default: openai if OPENAI_API_KEY is set, else first50.",
    )
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    from_n = args.from_per_bucket
    to_n = args.to_per_bucket
    if from_n < 1 or to_n < 1 or to_n > from_n:
        print("error: need 1 <= to-per-bucket <= from-per-bucket", file=sys.stderr)
        return 2

    q_path = args.questions
    a_path = args.answers
    layout_out = None if args.no_layout else args.layout_out

    strategy = args.strategy
    if strategy is None:
        strategy = "openai" if os.environ.get("OPENAI_API_KEY") else "first50"
        if strategy == "first50":
            print("note: OPENAI_API_KEY unset; using strategy first50", file=sys.stderr)

    mcqs = parse_questions(q_path)
    ans = parse_answers(a_path)
    expect = from_n * 8
    if len(mcqs) != expect:
        print(f"error: expected {expect} questions, got {len(mcqs)}", file=sys.stderr)
        return 2
    for q in mcqs:
        if q.number not in ans:
            print(f"error: missing answer for question {q.number}", file=sys.stderr)
            return 2

    client: OpenAI | None = None
    if strategy == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("error: strategy openai requires OPENAI_API_KEY", file=sys.stderr)
            return 2
        client = OpenAI()

    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (q_path, a_path):
            if p.is_file():
                bak = p.with_suffix(p.suffix + f".bak_{ts}")
                shutil.copy2(p, bak)
                print(f"backup: {bak}")

    kept_mcqs: list[MCQ] = []
    per_bucket_picks: list[list[int]] = []

    for bi, (bid, bname) in enumerate(BUCKETS):
        lo = bi * from_n
        hi = lo + from_n
        bucket_mcqs = mcqs[lo:hi]
        if len(bucket_mcqs) != from_n:
            print(f"error: bucket {bid} size {len(bucket_mcqs)} != {from_n}", file=sys.stderr)
            return 2

        if strategy == "first50":
            local = list(range(1, to_n + 1))
        else:
            assert client is not None
            local = openai_pick_indices(
                client,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                bucket_id=bid,
                bucket_name=bname,
                mcqs=bucket_mcqs,
                from_n=from_n,
                to_n=to_n,
            )
            local = normalize_keep_indices(local, from_n=from_n, to_n=to_n)
            time.sleep(args.sleep_ms / 1000.0)

        per_bucket_picks.append(local)
        for idx in local:
            kept_mcqs.append(bucket_mcqs[idx - 1])

    if len(kept_mcqs) != to_n * 8:
        print(f"error: internal size {len(kept_mcqs)}", file=sys.stderr)
        return 2

    renumbered: list[MCQ] = []
    new_answers: dict[int, str] = {}
    for new_n, q in enumerate(kept_mcqs, start=1):
        renumbered.append(MCQ(number=new_n, stem=q.stem, choices=dict(q.choices)))
        new_answers[new_n] = ans[q.number]

    header_note = "medium difficulty; reference: PDF excerpts (not verbatim copies)"
    if strategy == "openai":
        header_note += f"; trimmed with OpenAI model={args.model}"
    else:
        header_note += "; trimmed first N per bucket (no API)"

    write_questions_and_answers(
        renumbered,
        new_answers,
        q_path,
        a_path,
        per_bucket=to_n,
        header_note=header_note,
    )

    if layout_out is not None:
        notes = f";trim_strategy={strategy}"
        if strategy == "openai":
            notes += f";model={args.model}"
        write_layout_csv(
            layout_out,
            per_bucket=to_n,
            model_label="generator_layout",
            notes_suffix=notes,
        )

    print(f"Wrote {len(renumbered)} questions → {q_path}")
    print(f"Wrote {len(new_answers)} answers → {a_path}")
    if layout_out:
        print(f"Wrote layout → {layout_out}")
    for bid, loc in zip([b[0] for b in BUCKETS], per_bucket_picks):
        print(f"bucket {bid} kept local indices: {loc[:8]}{'...' if len(loc) > 8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
