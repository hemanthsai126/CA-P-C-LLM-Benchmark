#!/usr/bin/env python3
"""
Write bucket CSV + PNG assuming **fixed-size blocks in question number order**:
  questions 1..per_bucket → bucket 1,
  per_bucket+1 .. 2*per_bucket → bucket 2,
  … (no API).

Use this for sets produced by ``generate_synthetic_handbook_mcqs.py`` (default 100×8),
where ``question_buckets_gpt-4.1.*`` only shows how a **model would re-tag** topics,
not the curriculum layout used during generation.

Example:
  .venv/bin/python3 scripts/write_ordered_layout_bucket_plot.py \\
    --questions results/synthetic_data/questions_formatted.txt \\
    --per-bucket 100 \\
    --out-csv results/synthetic_data/question_buckets_generator_layout.csv \\
    --out-plot results/synthetic_data/question_buckets_generator_layout.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_bo", _SCRIPTS / "bucket_questions_openai.py")
assert _spec and _spec.loader
_bo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bo)

parse_questions = _bo.parse_questions
BUCKET_INDEX = _bo.BUCKET_INDEX
write_plot = _bo.write_plot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-plot", type=Path, required=True)
    args = ap.parse_args()

    qs = parse_questions(args.questions)
    if not qs:
        print(f"No questions parsed from {args.questions}", file=sys.stderr)
        return 2

    per = args.per_bucket
    rows: list[dict] = []
    for q in qs:
        b = (q.number - 1) // per + 1
        if b < 1:
            b = 1
        if b > 8:
            b = 8
        rows.append(
            {
                "question_number": str(q.number),
                "bucket": str(b),
                "bucket_name": BUCKET_INDEX[b],
                "model": "generator_layout",
                "notes": f"ordered_block_per_bucket={per}",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_number", "bucket", "bucket_name", "model", "notes"])
        w.writeheader()
        w.writerows(rows)

    title = f"Topic buckets (generator layout, {per}/bucket, n={len(rows)})"
    write_plot(rows, args.out_plot, title)
    print(f"Wrote {args.out_csv}", file=sys.stderr)
    print(f"Wrote {args.out_plot}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
