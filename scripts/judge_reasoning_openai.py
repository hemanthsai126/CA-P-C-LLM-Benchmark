#!/usr/bin/env python3
"""
Judge explanation quality for each model’s ``*_reasoned.csv`` under the YouTube benchmark
using a single OpenAI **judge** model.

The judge compares the model’s ``reason`` text to the ground-truth line in
``results/from_youtube_video/explanations.txt`` (same ``question_number``), with the MCQ
stem and options for context.

Score scale (0–3):
  0 worst, 1 bad, 2 ok, 3 better

Defaults (paths relative to repo root = parent of ``scripts/``):

  --results-dir   results/from_youtube_video/option   (glob ``*_reasoned.csv``)
  --questions     results/from_youtube_video/questions.txt
  --answers       results/from_youtube_video/answers.txt
  --explanations  results/from_youtube_video/explanations.txt

Outputs:

  judge_runs_openai/<judge_model>/<run_stem>.jsonl
  judge_runs_openai/<judge_model>/summary.csv

Auth:
  export OPENAI_API_KEY=...

For reasoning models (e.g. gpt-5.2), pass ``--reasoning-effort medium`` (or low/high).
Use ``--reasoning-effort off`` to omit the ``reasoning`` field (e.g. gpt-4.1-mini).

Legacy layout (``data/eval_set/from_youtube_video/``) is still supported by passing explicit
``--questions``, ``--answers``, ``--explanations``, and ``--results-dir`` paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import BadRequestError, OpenAI


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]  # A-D


def parse_questions_txt(path: Path) -> dict[int, MCQ]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    out: dict[int, MCQ] = {}
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", lines[0])
        if not m:
            continue
        n = int(m.group(1))
        stem = m.group(2).strip()
        choices: dict[str, str] = {}
        for ln in lines[1:]:
            cm = re.match(r"^([A-D])\.\s*(.*)$", ln)
            if cm:
                choices[cm.group(1)] = cm.group(2).strip()
        out[n] = MCQ(number=n, stem=stem, choices=choices)
    return out


def parse_answers_txt(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        n_s, L = ln.split()
        out[int(n_s)] = L.strip().upper()
    return out


def parse_explanations_txt(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s+(.*)$", ln)
        if not m:
            continue
        body = m.group(2).strip()
        if body:
            out[int(m.group(1))] = body
    return out


def list_result_csvs(results_dir: Path) -> list[Path]:
    """Ollama-style ``*_reasoned.csv`` and OpenAI-style ``*_reasoned_from_<source>.csv``."""
    paths: set[Path] = set()
    for p in results_dir.glob("*_reasoned.csv"):
        if p.is_file():
            paths.add(p)
    for p in results_dir.glob("*_reasoned_from_*.csv"):
        if p.is_file():
            paths.add(p)
    return sorted(paths)


RUBRIC = """You are grading how well a model's explanation matches the ground-truth explanation.

Use this 0–3 scale:
0 (worst): unrelated, nonsensical, or clearly contradicts the reference explanation.
1 (bad): somewhat related but misses the key idea(s) or adds major incorrect claims.
2 (ok): mostly matches the key idea(s), with minor omissions or mild imprecision.
3 (better): clearly matches the key idea(s) and is accurate and well-grounded.

You MUST output JSON ONLY with these keys:
  alignment_score: integer 0..3
  notes: short string (1-2 sentences)
"""


def build_user_content(*, q: MCQ, chosen_option: str, model_reason: str, reference_expl: str) -> str:
    return "\n".join(
        [
            "### Question",
            f"{q.number}. {q.stem}",
            "",
            "### Options",
            f"A. {q.choices.get('A','')}",
            f"B. {q.choices.get('B','')}",
            f"C. {q.choices.get('C','')}",
            f"D. {q.choices.get('D','')}",
            "",
            "### Model output",
            f"Chosen option: {chosen_option or '(blank)'}",
            "Explanation:",
            (model_reason or "(blank)").strip(),
            "",
            "### Ground-truth explanation",
            reference_expl.strip(),
        ]
    )


def call_openai_json(
    client: OpenAI,
    *,
    judge_model: str,
    user_content: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "model": judge_model,
        "input": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort and str(reasoning_effort).lower() != "off":
        req["reasoning"] = {"effort": reasoning_effort}

    try:
        resp = client.responses.create(**req)
    except BadRequestError as e:
        msg = str(e).lower()
        if "temperature" in msg and "not supported" in msg and "temperature" in req:
            req.pop("temperature", None)
            resp = client.responses.create(**req)
        else:
            raise
    text = resp.output_text.strip() if getattr(resp, "output_text", None) else ""
    if not text:
        # Fallback: join text parts
        out_parts: list[str] = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_parts.append(getattr(c, "text", "") or "")
        text = "".join(out_parts).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise


def clamp_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _resolve(repo: Path, p: Path) -> Path:
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def load_repo_dotenv(repo: Path) -> None:
    """Load ``repo/.env`` into the process environment (keys not already set). No extra deps."""
    p = repo / ".env"
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    yt = repo / "results" / "from_youtube_video"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-model", default="gpt-4.1-mini", help="OpenAI model id for the judge (e.g. gpt-4.1-mini, gpt-4o)")
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        help='For reasoning models: low, medium, high. Use "off" to omit the reasoning field.',
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=yt / "option",
        help="Directory containing *_reasoned.csv (default: results/from_youtube_video/option)",
    )
    ap.add_argument("--questions", type=Path, default=yt / "questions.txt")
    ap.add_argument("--answers", type=Path, default=yt / "answers.txt")
    ap.add_argument("--explanations", type=Path, default=yt / "explanations.txt")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated CSV stems to judge (default: all *_reasoned.csv). E.g. mistral_7b_reasoned,gpt-4o_reasoned_from_youtube_video",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit rows per model CSV (0=all)")
    ap.add_argument("--max-output-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep-ms", type=int, default=0)
    args = ap.parse_args()

    load_repo_dotenv(repo)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Export it in the shell or add it to a repo-root `.env` file "
            "(see `.gitignore`; never commit secrets)."
        )

    questions_p = _resolve(repo, args.questions)
    answers_p = _resolve(repo, args.answers)
    expl_p = _resolve(repo, args.explanations)
    results_dir = _resolve(repo, args.results_dir)

    qmap = parse_questions_txt(questions_p)
    answers = parse_answers_txt(answers_p)
    ref = parse_explanations_txt(expl_p)

    csvs = list_result_csvs(results_dir)
    if args.only.strip():
        allow = {s.strip() for s in args.only.split(",") if s.strip()}
        csvs = [p for p in csvs if p.stem in allow]
        missing = allow - {p.stem for p in csvs}
        if missing:
            raise SystemExit(f"--only stems not found as *_reasoned.csv: {sorted(missing)}")
    if not csvs:
        raise SystemExit(f"No *_reasoned.csv files found in {results_dir}")

    client = OpenAI()
    out_dir = (repo / "judge_runs_openai" / args.judge_model).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for csv_path in csvs:
        run_name = csv_path.stem
        out_jsonl = out_dir / f"{run_name}.jsonl"

        model_rows: list[dict[str, str]] = []
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
            r = csv.DictReader(f)
            for row in r:
                model_rows.append(row)

        if args.limit and args.limit > 0:
            model_rows = model_rows[: args.limit]

        n_scored = 0
        sum_align = 0
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        n_correct = 0

        with out_jsonl.open("w", encoding="utf-8") as jf:
            for row in model_rows:
                try:
                    qn = int((row.get("question_number") or "").strip())
                except Exception:
                    continue
                if qn not in qmap or qn not in ref:
                    continue

                chosen = (row.get("answer") or row.get("option") or "").strip().upper()
                reason = (row.get("reason") or "").strip()
                user_content = build_user_content(
                    q=qmap[qn],
                    chosen_option=chosen,
                    model_reason=reason,
                    reference_expl=ref[qn],
                )

                t0 = time.time()
                judged = call_openai_json(
                    client,
                    judge_model=args.judge_model,
                    user_content=user_content,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    reasoning_effort=args.reasoning_effort,
                )
                dt = time.time() - t0

                ascore = clamp_int(judged.get("alignment_score", 0), 0, 3)
                notes = str(judged.get("notes", "")).strip()

                is_correct = bool(chosen and (qn in answers) and (chosen == answers[qn]))
                if is_correct:
                    n_correct += 1

                rec = {
                    "question_number": qn,
                    "model_option": chosen,
                    "correct_answer": answers.get(qn, ""),
                    "is_correct": is_correct,
                    "alignment_score": ascore,
                    "notes": notes,
                    "latency_s": round(dt, 3),
                }
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                n_scored += 1
                sum_align += ascore
                counts[ascore] += 1

                if args.sleep_ms and args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

        avg_align = (sum_align / n_scored) if n_scored else 0.0
        summary_rows.append(
            {
                "run": run_name,
                "file": str(csv_path.relative_to(repo)) if csv_path.is_relative_to(repo) else csv_path.as_posix(),
                "judge_model": args.judge_model,
                "n_scored": n_scored,
                "accuracy": round((n_correct / n_scored) if n_scored else 0.0, 4),
                "avg_alignment_score": round(avg_align, 4),
                "pct_score_0": round(counts[0] / n_scored, 4) if n_scored else 0.0,
                "pct_score_1": round(counts[1] / n_scored, 4) if n_scored else 0.0,
                "pct_score_2": round(counts[2] / n_scored, 4) if n_scored else 0.0,
                "pct_score_3": round(counts[3] / n_scored, 4) if n_scored else 0.0,
            }
        )

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"Wrote per-item JSONL -> {out_dir}/<run>.jsonl")
    print(f"Wrote summary CSV -> {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

