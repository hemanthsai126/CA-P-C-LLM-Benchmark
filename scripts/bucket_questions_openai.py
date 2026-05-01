#!/usr/bin/env python3
"""
Classify each MCQ in questions.txt into one of eight curriculum buckets using OpenAI
(default: gpt-4.1), then write a CSV and a scatter (+ bar) plot.

Buckets (exactly one integer 1–8 per question):
  1 Basic concepts
  2 Contract law
  3 CA laws & ethics
  4 Tort & property fundamentals
  5 Residential property
  6 Valuation & government programs
  7 Personal liability & inland marine
  8 Commercial insurance

Auth:
  export OPENAI_API_KEY=...

Examples:
  python3 scripts/bucket_questions_openai.py --questions eval_set/from_youtube_video/questions.txt \\
    --out-csv eval_set/from_youtube_video/question_buckets_gpt-4.1.csv --out-plot eval_set/from_youtube_video/question_buckets_gpt-4.1.png

  python3 scripts/bucket_questions_openai.py --plot-only eval_set/from_youtube_video/question_buckets_gpt-4.1.csv \\
    --out-plot eval_set/from_youtube_video/question_buckets_gpt-4.1.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import BadRequestError, OpenAI

_SCRIPTS = Path(__file__).resolve().parent
_MOD = "_broker_run_questions_reasoned_openai"
_spec = importlib.util.spec_from_file_location(_MOD, _SCRIPTS / "run_questions_reasoned_openai.py")
assert _spec and _spec.loader
_rq = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _rq
_spec.loader.exec_module(_rq)

parse_questions = _rq.parse_questions
MCQ = _rq.MCQ

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

BUCKET_INDEX = {n: name for n, name in BUCKETS}

SYSTEM = """You label California property & casualty licensing style multiple-choice questions
for a study outline. Pick exactly ONE topic bucket (integer 1–8) that best matches the question.

Buckets:
1 = Basic concepts
2 = Contract law
3 = CA laws & ethics
4 = Tort & property fundamentals
5 = Residential property
6 = Valuation & government programs
7 = Personal liability & inland marine
8 = Commercial insurance

Reply with JSON ONLY, no markdown fences:
{"bucket": <integer 1-8>}"""


def build_user_block(q: MCQ) -> str:
    lines = [f"Question {q.number}:", "", q.stem, "", "Options:"]
    for k in ("A", "B", "C", "D"):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    return "\n".join(lines)


def call_bucket_json(
    client: OpenAI,
    *,
    model: str,
    user: str,
    temperature: float,
    max_output_tokens: int,
) -> dict:
    req: dict = {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM},
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
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def clamp_bucket(x: object) -> tuple[int, str]:
    try:
        v = int(x)
    except Exception:
        return 1, "parse_error"
    if v < 1 or v > 8:
        return max(1, min(8, v)), "clamped"
    return v, ""


def write_plot(rows: list[dict], out_plot: Path, title: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    qns = [int(r["question_number"]) for r in rows]
    buckets = [int(r["bucket"]) for r in rows]
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.28, 0.28, size=len(qns))
    y = np.array(buckets, dtype=float) + jitter

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.35, 0.65]})
    ax_s, ax_b = axes

    cmap = mpl.colormaps["tab10"].resampled(8)
    sc = ax_s.scatter(qns, y, c=np.array(buckets), cmap=cmap, vmin=1, vmax=8, s=52, alpha=0.9, edgecolors="0.3", linewidths=0.35)
    ax_s.set_xlabel("Question number")
    ax_s.set_ylabel("Topic bucket (jittered for visibility)")
    ax_s.set_yticks(range(1, 9))
    ax_s.set_yticklabels([f"{n} {name[:22]}…" if len(name) > 22 else f"{n} {name}" for n, name in BUCKETS], fontsize=8)
    ax_s.set_ylim(0.4, 8.6)
    ax_s.grid(True, axis="y", alpha=0.25)
    ax_s.set_title(title)

    counts = [sum(1 for b in buckets if b == n) for n in range(1, 9)]
    colors = [cmap((n - 1) / 7.0) for n in range(1, 9)]
    ax_b.barh(list(range(1, 9)), counts, color=colors, edgecolor="0.3", linewidth=0.35)
    ax_b.set_xlabel("Count")
    ax_b.set_yticks(range(1, 9))
    ax_b.set_yticklabels([f"{n}" for n in range(1, 9)])
    ax_b.set_title("Questions per bucket")
    ax_b.grid(True, axis="x", alpha=0.25)
    ax_b.invert_yaxis()

    cbar = fig.colorbar(sc, ax=ax_s, fraction=0.03, pad=0.02, ticks=range(1, 9))
    cbar.set_ticklabels([str(i) for i in range(1, 9)])

    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_rows_from_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=None, help="questions.txt (required unless --plot-only)")
    ap.add_argument("--plot-only", type=Path, default=None, help="Existing bucket CSV; only draw plot")
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-plot", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=120)
    ap.add_argument("--sleep-ms", type=int, default=0)
    args = ap.parse_args()

    if args.plot_only:
        rows = load_rows_from_csv(args.plot_only)
        if not rows:
            raise SystemExit("No rows in CSV")
        title = f"Topic buckets (from {args.plot_only.name})"
        write_plot(rows, args.out_plot, title)
        print(f"Wrote plot -> {args.out_plot}")
        return 0

    if not args.questions or not args.out_csv:
        raise SystemExit("Need --questions and --out-csv unless using --plot-only")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY environment variable.")

    qs = parse_questions(args.questions)
    if not qs:
        print(f"No questions parsed from {args.questions}", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        qs = qs[: args.limit]

    client = OpenAI()
    rows: list[dict] = []

    for q in qs:
        present = [k for k in ("A", "B", "C", "D") if q.choices.get(k)]
        if len(present) < 2:
            rows.append(
                {
                    "question_number": q.number,
                    "bucket": "1",
                    "bucket_name": BUCKET_INDEX[1],
                    "model": args.model,
                    "notes": "skipped_malformed_choices",
                }
            )
            continue

        user = build_user_block(q)
        t0 = time.time()
        try:
            data = call_bucket_json(
                client,
                model=args.model,
                user=user,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            b, flag = clamp_bucket(data.get("bucket"))
            notes = flag or ""
        except Exception as e:
            b = 1
            notes = f"api_error: {e!s}"[:200]

        rows.append(
            {
                "question_number": q.number,
                "bucket": str(b),
                "bucket_name": BUCKET_INDEX[b],
                "model": args.model,
                "notes": notes,
            }
        )
        dt = time.time() - t0
        print(f"Q{q.number} -> bucket {b} ({dt:.2f}s)", file=sys.stderr)
        if args.sleep_ms and args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_number", "bucket", "bucket_name", "model", "notes"])
        w.writeheader()
        w.writerows(rows)

    title = f"Topic buckets ({args.model}, n={len(rows)})"
    write_plot(rows, args.out_plot, title)
    print(f"Wrote CSV -> {args.out_csv}")
    print(f"Wrote plot -> {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
