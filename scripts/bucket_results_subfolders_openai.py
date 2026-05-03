#!/usr/bin/env python3
"""
Run GPT-4.1 topic-bucket classification for every questions file under ``results/``,
and save ``question_buckets_gpt-4.1.csv`` + ``question_buckets_gpt-4.1.png`` in the same folder.

Discovers:
  - ``results/**/questions.txt``
  - ``results/**/questions_formatted*.txt`` (e.g. ``questions_formatted.txt``, ``questions_formatted_165.txt``)

Requires ``OPENAI_API_KEY``. Delegates to ``scripts/bucket_questions_openai.py``.

Example:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/bucket_results_subfolders_openai.py --model gpt-4.1 --sleep-ms 40
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def discover_question_files(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        raise SystemExit(f"Not a directory: {results_root}")
    found: set[Path] = set()
    for pattern in ("**/questions.txt", "**/questions_formatted*.txt"):
        for p in results_root.glob(pattern):
            if p.is_file():
                found.add(p.resolve())
    return sorted(found, key=lambda p: str(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results"), help="Root to scan (default: ./results)")
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--sleep-ms", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="List files only; do not call OpenAI")
    args = ap.parse_args()

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required (or pass --dry-run).")

    results_root = args.results_root.resolve()
    paths = discover_question_files(results_root)
    if not paths:
        print(f"No questions*.txt under {results_root}", file=sys.stderr)
        return 1

    repo = Path(__file__).resolve().parents[1]
    bucket_script = repo / "scripts" / "bucket_questions_openai.py"
    if not bucket_script.is_file():
        raise SystemExit(f"Missing {bucket_script}")

    print(f"Found {len(paths)} question file(s) under {results_root}:", file=sys.stderr)
    for p in paths:
        print(f"  {p.relative_to(repo)}", file=sys.stderr)

    if args.dry_run:
        return 0

    for qpath in paths:
        out_dir = qpath.parent
        csv_path = out_dir / "question_buckets_gpt-4.1.csv"
        png_path = out_dir / "question_buckets_gpt-4.1.png"
        cmd = [
            sys.executable,
            str(bucket_script),
            "--questions",
            str(qpath),
            "--out-csv",
            str(csv_path),
            "--out-plot",
            str(png_path),
            "--model",
            args.model,
            "--sleep-ms",
            str(args.sleep_ms),
            "--temperature",
            str(args.temperature),
            "--max-output-tokens",
            str(args.max_output_tokens),
        ]
        print(f"\n=== {qpath.relative_to(repo)} -> {csv_path.name}, {png_path.name} ===", file=sys.stderr)
        r = subprocess.run(cmd, cwd=str(repo))
        if r.returncode != 0:
            print(f"FAILED (exit {r.returncode}): {' '.join(cmd)}", file=sys.stderr)
            return r.returncode

    print("\nAll bucket runs finished.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
