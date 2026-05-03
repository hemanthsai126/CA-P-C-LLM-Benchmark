#!/usr/bin/env python3
"""
Run ``scripts/run_questions_reasoned.py`` (Ollama, questions-only, no answer key) for each
benchmark snapshot under ``results/``.

By default, writes ``results/<source>/option/<model>.csv`` with columns
``question_number``, ``answer``, ``reason``.

With ``--reasoned-csv``, writes ``results/<source>/option/<model>_reasoned.csv`` with the same
three columns (same schema, different filename).

Default sources (relative to repo root):
  - results/from_quizlet_pdfs/questions.txt
  - results/from_youtube_video/questions.txt
  - results/synthetic_data/questions_formatted.txt

Example:
  .venv/bin/python3 scripts/run_results_option_ollama_batch.py --model tinyllama

Reasoned CSV (columns ``question_number``, ``answer``, ``reason``) next to each questions file:
  .venv/bin/python3 scripts/run_results_option_ollama_batch.py --model tinyllama --reasoned-csv

Requires a running Ollama server and the model pulled (e.g. ``ollama pull tinyllama``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def safe_out_stem(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="tinyllama", help="Ollama model tag (default: tinyllama)")
    ap.add_argument("--ollama", type=str, default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="Pass through to run_questions_reasoned (0 = all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent Ollama requests per source (pass-through). Default 6 for throughput.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated subdirs to run (default: all). E.g. from_youtube_video,synthetic_data",
    )
    ap.add_argument(
        "--reasoned-csv",
        action="store_true",
        help=(
            "Write results/<source>/option/<model>_reasoned.csv (same schema as default: question_number,answer,reason)."
        ),
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts" / "run_questions_reasoned.py"
    if not runner.is_file():
        raise SystemExit(f"Missing {runner}")

    stem = safe_out_stem(args.model)
    if args.reasoned_csv:
        jobs: list[tuple[str, Path, Path]] = [
            (
                "from_quizlet_pdfs",
                repo / "results" / "from_quizlet_pdfs" / "questions.txt",
                repo / "results" / "from_quizlet_pdfs" / "option" / f"{stem}_reasoned.csv",
            ),
            (
                "from_youtube_video",
                repo / "results" / "from_youtube_video" / "questions.txt",
                repo / "results" / "from_youtube_video" / "option" / f"{stem}_reasoned.csv",
            ),
            (
                "synthetic_data",
                repo / "results" / "synthetic_data" / "questions_formatted.txt",
                repo / "results" / "synthetic_data" / "option" / f"{stem}_reasoned.csv",
            ),
        ]
    else:
        jobs = [
            (
                "from_quizlet_pdfs",
                repo / "results" / "from_quizlet_pdfs" / "questions.txt",
                repo / "results" / "from_quizlet_pdfs" / "option" / f"{stem}.csv",
            ),
            (
                "from_youtube_video",
                repo / "results" / "from_youtube_video" / "questions.txt",
                repo / "results" / "from_youtube_video" / "option" / f"{stem}.csv",
            ),
            (
                "synthetic_data",
                repo / "results" / "synthetic_data" / "questions_formatted.txt",
                repo / "results" / "synthetic_data" / "option" / f"{stem}.csv",
            ),
        ]

    only_set = {s.strip() for s in args.only.split(",") if s.strip()}
    if only_set:
        jobs = [j for j in jobs if j[0] in only_set]

    for name, qpath, outpath in jobs:
        if not qpath.is_file():
            print(f"SKIP {name}: missing {qpath}", file=sys.stderr)
            continue
        outpath.parent.mkdir(parents=True, exist_ok=True)
        # -u + PYTHONUNBUFFERED: child stderr is line-flushed so progress prints show immediately.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        cmd = [
            sys.executable,
            "-u",
            str(runner),
            "--questions",
            str(qpath),
            "--model",
            args.model,
            "--ollama",
            args.ollama,
            "--out",
            str(outpath),
        ]
        if args.limit and args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        cmd += ["--workers", str(args.workers)]
        print(f"\n=== {name} -> {outpath.relative_to(repo)} ===", file=sys.stderr)
        r = subprocess.run(cmd, cwd=str(repo), env=env)
        if r.returncode != 0:
            print(f"FAILED {name} (exit {r.returncode})", file=sys.stderr)
            return r.returncode

    if args.reasoned_csv:
        print(f"\nDone. Outputs: results/*/option/{stem}_reasoned.csv", file=sys.stderr)
    else:
        print(f"\nDone. Outputs: results/*/option/{stem}.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
