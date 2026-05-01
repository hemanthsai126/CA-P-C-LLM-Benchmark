#!/usr/bin/env python3
"""
Judge reasoning quality for each model output in results/*_reasoned.csv using Anthropic (Claude).

Inputs:
  - eval_set/from_youtube_video/questions.txt        (question + A-D choices)
  - eval_set/from_youtube_video/answers.txt          (ground truth letters; used only for reporting correctness)
  - eval_set/from_youtube_video/explanations.txt     (optional reference explanations; used only for alignment scoring)
  - results/*_reasoned.csv        (question_number, option, reason)

Outputs:
  - judge_runs/<judge_model>/<run_name>.jsonl   (per-item judge results)
  - judge_runs/<judge_model>/summary.csv        (per-model aggregates)

Auth:
  export ANTHROPIC_API_KEY=...
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

from anthropic import Anthropic
from anthropic import NotFoundError


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]  # A-D


def parse_questions_txt(path: Path) -> dict[int, MCQ]:
    """
    Parse eval_set/from_youtube_video/questions.txt format:
      N. stem
      A. ...
      B. ...
      C. ...
      D. ...
      <blank line>
    """
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
        if not ln:
            continue
        m = re.match(r"^(\d+)\s+(.*)$", ln)
        if not m:
            continue
        out[int(m.group(1))] = m.group(2).strip()
    return out


def list_result_csvs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*_reasoned.csv"))


RUBRIC = """You are grading the QUALITY of the model's explanation for a multiple-choice question.
You must use this 0–3 scale:

0 (worst): empty/nonsense/unrelated; clearly contradictory; or obviously hallucinated; no justification.
1 (bad): somewhat relevant but mostly generic; key concept wrong; doesn't engage options; weak justification.
2 (ok): mostly correct high-level reasoning; ties to question; may mention why choice fits OR why one alternative doesn't.
3 (better): clear and specific; grounded in the question/options; distinguishes the best option and rules out 1–2 wrong options; no obvious hallucinations.

You MUST output JSON ONLY with these keys:
  reasoning_score: integer 0..3
  notes: short string (1-2 sentences)
If a reference explanation is provided, also include:
  alignment_score: integer 0..3  (how well the model explanation matches the key idea(s) in the reference)
"""


def build_judge_user_content(
    *,
    q: MCQ,
    model_option: str,
    model_reason: str,
    reference_explanation: str | None,
) -> str:
    lines = [
        "### Question",
        f"{q.number}. {q.stem}",
        "",
        "### Options",
        f"A. {q.choices.get('A','')}",
        f"B. {q.choices.get('B','')}",
        f"C. {q.choices.get('C','')}",
        f"D. {q.choices.get('D','')}",
        "",
        "### Model answer",
        f"Chosen option: {model_option}",
        "Explanation:",
        model_reason.strip(),
    ]
    if reference_explanation is not None:
        lines += [
            "",
            "### Reference explanation (for alignment scoring only)",
            reference_explanation.strip(),
        ]
    return "\n".join(lines)


def call_claude_json(
    client: Anthropic,
    *,
    judge_model: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    msg = client.messages.create(
        model=judge_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=RUBRIC,
        messages=[{"role": "user", "content": user_content}],
    )
    text = ""
    for part in msg.content:
        if getattr(part, "type", None) == "text":
            text += part.text
    text = text.strip()
    # Best-effort JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object if the model wrapped it
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # Prefer an explicit, commonly-available model id (avoid "*-latest" aliases which may not exist).
    # Claude 3.5 IDs may not be enabled on all accounts; Claude 3 Haiku is widely available.
    ap.add_argument("--judge-model", default="claude-3-haiku-20240307")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--questions", type=Path, default=Path("eval_set/from_youtube_video/questions.txt"))
    ap.add_argument("--answers", type=Path, default=Path("eval_set/from_youtube_video/answers.txt"))
    ap.add_argument("--explanations", type=Path, default=Path("eval_set/from_youtube_video/explanations.txt"))
    ap.add_argument("--use-reference-explanations", action="store_true", help="Include alignment_score vs explanations.txt")
    ap.add_argument("--limit", type=int, default=0, help="Limit questions per model (0=all)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between requests")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Missing ANTHROPIC_API_KEY environment variable.")

    qmap = parse_questions_txt(args.questions)
    answers = parse_answers_txt(args.answers)
    ref_expl = parse_explanations_txt(args.explanations) if args.use_reference_explanations else {}

    csvs = list_result_csvs(args.results_dir)
    if not csvs:
        raise SystemExit(f"No *_reasoned.csv files found in {args.results_dir}")

    client = Anthropic()

    out_dir = Path("judge_runs") / args.judge_model
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for csv_path in csvs:
        run_name = csv_path.stem
        out_jsonl = out_dir / f"{run_name}.jsonl"

        # Load model outputs
        model_rows: list[dict[str, str]] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                model_rows.append(row)

        # Optional limit
        if args.limit and args.limit > 0:
            model_rows = model_rows[: args.limit]

        n_scored = 0
        sum_score = 0
        sum_score_correct = 0
        n_correct = 0
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        align_sum = 0
        align_n = 0

        with out_jsonl.open("w", encoding="utf-8") as jf:
            for row in model_rows:
                try:
                    qn = int(row.get("question_number", "").strip())
                except Exception:
                    continue
                if qn not in qmap:
                    continue
                opt = (row.get("option") or "").strip().upper()
                reason = (row.get("reason") or "").strip()

                # Build judge input
                ref = ref_expl.get(qn) if args.use_reference_explanations else None
                user_content = build_judge_user_content(
                    q=qmap[qn],
                    model_option=opt or "(blank)",
                    model_reason=reason or "(blank)",
                    reference_explanation=ref,
                )

                # Call judge
                t0 = time.time()
                try:
                    judged = call_claude_json(
                        client,
                        judge_model=args.judge_model,
                        user_content=user_content,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                except NotFoundError as e:
                    common = [
                        "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307",
                        "claude-3-5-sonnet-20240620",
                        "claude-3-5-haiku-20241022",
                    ]
                    raise SystemExit(
                        "Anthropic returned 404 for the judge model name.\n"
                        f"Requested: {args.judge_model}\n\n"
                        "Try rerunning with one of these common model ids:\n"
                        + "\n".join(f"  - {m}" for m in common)
                        + "\n\nExample:\n"
                        "  python3 scripts/judge_reasoning_anthropic.py --judge-model claude-3-haiku-20240307\n"
                    ) from e
                dt = time.time() - t0

                rscore = clamp_int(judged.get("reasoning_score", 0), 0, 3)
                notes = str(judged.get("notes", "")).strip()
                aline = judged.get("alignment_score", None) if args.use_reference_explanations else None
                ascore = clamp_int(aline, 0, 3) if aline is not None else None

                correct = bool(opt and (qn in answers) and (opt == answers[qn]))

                rec = {
                    "question_number": qn,
                    "model_option": opt,
                    "correct_answer": answers.get(qn, ""),
                    "is_correct": correct,
                    "reasoning_score": rscore,
                    "notes": notes,
                    "latency_s": round(dt, 3),
                }
                if args.use_reference_explanations:
                    rec["alignment_score"] = ascore

                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Aggregate
                n_scored += 1
                sum_score += rscore
                counts[rscore] += 1
                if correct:
                    n_correct += 1
                    sum_score_correct += rscore
                if ascore is not None:
                    align_sum += ascore
                    align_n += 1

                if args.sleep_ms and args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)

        avg = (sum_score / n_scored) if n_scored else 0.0
        avg_correct = (sum_score_correct / n_correct) if n_correct else 0.0
        summary = {
            "run": run_name,
            "file": csv_path.as_posix(),
            "judge_model": args.judge_model,
            "n_scored": n_scored,
            "accuracy": round((n_correct / n_scored) if n_scored else 0.0, 4),
            "avg_reasoning_score": round(avg, 4),
            "avg_reasoning_score_correct_only": round(avg_correct, 4),
            "pct_score_0": round(counts[0] / n_scored, 4) if n_scored else 0.0,
            "pct_score_1": round(counts[1] / n_scored, 4) if n_scored else 0.0,
            "pct_score_2": round(counts[2] / n_scored, 4) if n_scored else 0.0,
            "pct_score_3": round(counts[3] / n_scored, 4) if n_scored else 0.0,
        }
        if args.use_reference_explanations:
            summary["avg_alignment_score"] = round((align_sum / align_n) if align_n else 0.0, 4)
        summary_rows.append(summary)

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

