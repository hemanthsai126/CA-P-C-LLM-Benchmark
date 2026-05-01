#!/usr/bin/env python3
"""
Balance MCQs so each curriculum bucket has exactly ``--target`` questions (default 100),
for 8 buckets → ``8 * target`` total (default 800).

- Buckets with **more** than ``target``: randomly keep ``target`` items (reproducible seed).
- Buckets with **fewer** than ``target``: generate fill-in MCQs via OpenAI + handbook excerpts
  (same pipeline as ``generate_synthetic_handbook_mcqs.py``).

Initial bucket assignment comes from ``--assignments-csv`` (e.g. heuristic CSV). Any question
missing from that CSV is classified with the same keyword rules as ``bucket_questions_heuristic.py``.

Output order: bucket 1 block, then bucket 2, …, bucket 8. Global numbering 1 … 8*target.

Writes questions block file, answers file, bucket CSV (layout truth), and PNG plot.

  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/balance_eval_buckets_layout.py --dry-run
  .venv/bin/python3 scripts/balance_eval_buckets_layout.py --target 100
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent

_spec_bh = importlib.util.spec_from_file_location("_bh", _SCRIPTS / "bucket_questions_heuristic.py")
assert _spec_bh and _spec_bh.loader
_bh = importlib.util.module_from_spec(_spec_bh)
_spec_bh.loader.exec_module(_bh)
bucket_for_text = _bh.bucket_for_text
parse_questions = _bh.parse_questions

_spec_bo = importlib.util.spec_from_file_location("_bo", _SCRIPTS / "bucket_questions_openai.py")
assert _spec_bo and _spec_bo.loader
_bo = importlib.util.module_from_spec(_spec_bo)
_spec_bo.loader.exec_module(_bo)
BUCKET_INDEX = _bo.BUCKET_INDEX
write_plot = _bo.write_plot

_spec_gen = importlib.util.spec_from_file_location("_gen", _SCRIPTS / "generate_synthetic_handbook_mcqs.py")
assert _spec_gen and _spec_gen.loader
_gen = importlib.util.module_from_spec(_spec_gen)
_spec_gen.loader.exec_module(_gen)
call_openai_batch = _gen.call_openai_batch
extract_handbook_context = _gen.extract_handbook_context
_norm_key = _gen._norm_key
_squish = _gen._squish

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

_ANS_LINE = re.compile(r"^(\d+)\s+([ABCD])\s*$", re.I)


def parse_answers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ANS_LINE.match(line)
        if not m:
            continue
        out[int(m.group(1))] = m.group(2).upper()
    return out


def load_bucket_map_from_csv(path: Path) -> dict[int, int]:
    m: dict[int, int] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m[int(row["question_number"])] = int(row["bucket"])
    return m


def mcq_to_item(q: Any, answer: str, orig_num: int) -> dict[str, Any]:
    ch = q.choices
    return {
        "stem": q.stem,
        "A": ch.get("A", "").strip(),
        "B": ch.get("B", "").strip(),
        "C": ch.get("C", "").strip(),
        "D": ch.get("D", "").strip(),
        "answer": answer,
        "notes": f"kept_from_q{orig_num}",
    }


def write_questions_answers(
    items: list[dict[str, Any]],
    q_path: Path,
    a_path: Path,
    header: str,
) -> None:
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        fa.write(header.rstrip() + "\n")
        for n, it in enumerate(items, start=1):
            fq.write(f"{n}. {_squish(it['stem'])}\n")
            fq.write(f"A. {_squish(it['A'])}\n")
            fq.write(f"B. {_squish(it['B'])}\n")
            fq.write(f"C. {_squish(it['C'])}\n")
            fq.write(f"D. {_squish(it['D'])}\n")
            fq.write("\n")
            fa.write(f"{n} {it['answer']}\n")


def write_layout_csv(items: list[dict[str, Any]], out_csv: Path, target: int) -> None:
    rows: list[dict[str, str]] = []
    for n, it in enumerate(items, start=1):
        b = (n - 1) // target + 1
        rows.append(
            {
                "question_number": str(n),
                "bucket": str(b),
                "bucket_name": BUCKET_INDEX[b],
                "model": "layout_balanced",
                "notes": it.get("notes", ""),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_number", "bucket", "bucket_name", "model", "notes"])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--questions", type=Path, default=Path("eval_set/from_quizlet_pdfs/questions_formatted.txt"))
    ap.add_argument("--answers", type=Path, default=Path("eval_set/from_quizlet_pdfs/answers.txt"))
    ap.add_argument(
        "--assignments-csv",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/question_buckets_heuristic.csv"),
        help="CSV with question_number,bucket,... used to group existing items before trim/fill",
    )
    ap.add_argument("--out-questions", type=Path, default=Path("eval_set/from_quizlet_pdfs/questions_formatted.txt"))
    ap.add_argument("--out-answers", type=Path, default=Path("eval_set/from_quizlet_pdfs/answers.txt"))
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/question_buckets_heuristic.csv"),
        help="Rewritten: layout truth (bucket by block position after balance).",
    )
    ap.add_argument(
        "--out-plot",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/question_buckets_heuristic.png"),
    )
    ap.add_argument("--handbook", type=Path, default=Path("source_material/Cali Data/Insurance_Handbook_20103.pdf"))
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--target", type=int, default=100, help="Exact count per bucket (8 buckets).")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--handbook-chars", type=int, default=110_000)
    ap.add_argument("--max-output-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--sleep-ms", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42, help="RNG seed when trimming overfull buckets.")
    ap.add_argument("--dry-run", action="store_true", help="Print per-bucket plan and exit (no API, no writes).")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    if args.target < 1 or args.target > 500:
        raise SystemExit("--target must be in a sane range (e.g. 1–500).")

    qs = parse_questions(args.questions)
    if not qs:
        raise SystemExit(f"No questions parsed from {args.questions}")
    answers = parse_answers(args.answers)
    missing_ans = [q.number for q in qs if q.number not in answers]
    if missing_ans:
        raise SystemExit(f"Missing answers for question numbers: {missing_ans[:20]}{'...' if len(missing_ans) > 20 else ''}")

    csv_map = {}
    if args.assignments_csv.is_file():
        csv_map = load_bucket_map_from_csv(args.assignments_csv)

    by_bucket: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, 9)}
    for q in qs:
        b = csv_map.get(q.number)
        if b is None or b < 1 or b > 8:
            opts_blob = " ".join(q.choices.get(k, "") for k in ("A", "B", "C", "D"))
            b, _tag = bucket_for_text(q.stem, opts_blob)
        item = mcq_to_item(q, answers[q.number], q.number)
        by_bucket[b].append(item)

    rng = random.Random(args.seed)
    trimmed: dict[int, list[dict[str, Any]]] = {}
    deficits: dict[int, int] = {}
    for b in range(1, 9):
        items = by_bucket[b][:]
        if len(items) > args.target:
            trimmed[b] = rng.sample(items, args.target)
            deficits[b] = 0
        else:
            trimmed[b] = items
            deficits[b] = args.target - len(items)

    need_api = sum(deficits.values()) > 0
    print("Per-bucket (before fill): kept / need_synthetic / deficit_after_trim", file=sys.stderr)
    for b in range(1, 9):
        print(
            f"  bucket {b} ({BUCKET_INDEX[b]}): have {len(by_bucket[b])}, "
            f"keep {len(trimmed[b])}, generate {deficits[b]}",
            file=sys.stderr,
        )
    print(f"Total synthetic needed: {sum(deficits.values())}", file=sys.stderr)

    if args.dry_run:
        return 0

    if need_api:
        if OpenAI is None:
            raise SystemExit("openai package not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY required to generate fill-in questions.")
        if not args.handbook.is_file():
            raise SystemExit(f"Handbook PDF not found: {args.handbook}")

    if not args.no_backup:
        ts = time.strftime("%Y%m%d_%H%M%S")
        for p in (args.out_questions, args.out_answers, args.out_csv, args.out_plot):
            if p.is_file():
                if p.suffix.lower() == ".png":
                    bak = p.with_name(p.stem + f".bak_{ts}" + p.suffix)
                else:
                    bak = p.with_suffix(p.suffix + f".bak_{ts}")
                shutil.copy2(p, bak)
                print(f"Backed up {p} -> {bak}", file=sys.stderr)

    handbook = ""
    client = None
    if need_api:
        handbook = extract_handbook_context(args.handbook, args.handbook_chars)
        if len(handbook) < 2000:
            raise SystemExit("Very little text extracted from handbook PDF.")
        client = OpenAI()

    seen: set[str] = set()
    for b in range(1, 9):
        for it in trimmed[b]:
            seen.add(_norm_key(it["stem"]))

    filled: dict[int, list[dict[str, Any]]] = {b: trimmed[b][:] for b in range(1, 9)}

    for b in range(1, 9):
        need = deficits[b]
        if need <= 0:
            continue
        assert client is not None
        bucket_name = BUCKET_INDEX[b]
        avoid: list[str] = [it["stem"] for it in filled[b]]
        while len(filled[b]) < args.target:
            batch = min(args.batch_size, args.target - len(filled[b]))
            new = call_openai_batch(
                client,
                model=args.model,
                handbook=handbook,
                bucket_id=b,
                bucket_name=bucket_name,
                need=batch,
                avoid=avoid,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            added = 0
            for it in new:
                k = _norm_key(it["stem"])
                if k in seen:
                    continue
                seen.add(k)
                filled[b].append(
                    {
                        "stem": it["stem"],
                        "A": it["A"],
                        "B": it["B"],
                        "C": it["C"],
                        "D": it["D"],
                        "answer": it["answer"],
                        "notes": f"synthetic_bucket{b}",
                    }
                )
                avoid.append(it["stem"])
                added += 1
            print(f"bucket {b}: +{added} synthetic (now {len(filled[b])}/{args.target})", file=sys.stderr)
            if added == 0:
                raise SystemExit(f"No new synthetic items accepted for bucket {b}; aborting.")
            if args.sleep_ms:
                time.sleep(args.sleep_ms / 1000.0)

    final: list[dict[str, Any]] = []
    for b in range(1, 9):
        if len(filled[b]) != args.target:
            raise SystemExit(f"Internal error: bucket {b} has {len(filled[b])} != {args.target}")
        final.extend(filled[b])

    total = 8 * args.target
    header = (
        f"# Balanced layout: exactly {args.target} questions per bucket × 8 = {total}; "
        f"trimmed overfull buckets, synthetic fill from handbook + {args.model} where needed.\n"
    )
    write_questions_answers(final, args.out_questions, args.out_answers, header)
    write_layout_csv(final, args.out_csv, args.target)
    title = f"Topic buckets (layout {args.target}/bucket, n={len(final)})"
    with args.out_csv.open(encoding="utf-8", newline="") as f:
        plot_rows = list(csv.DictReader(f))
    write_plot(plot_rows, args.out_plot, title)

    print(f"Wrote {total} questions -> {args.out_questions}", file=sys.stderr)
    print(f"Wrote {total} answers -> {args.out_answers}", file=sys.stderr)
    print(f"Wrote CSV -> {args.out_csv}", file=sys.stderr)
    print(f"Wrote plot -> {args.out_plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
