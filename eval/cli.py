"""Run MCQ eval against local Ollama. Labels file is never sent to the model."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.ollama_runner import evaluate_split, load_jsonl, load_labels


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def append_csv(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_experiment_log_md(
    *,
    log_path: Path,
    ts: str,
    model: str,
    method: str,
    correct: int,
    total: int,
    accuracy: float,
    notes: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            "# Experiment log\n\n"
            "| UTC timestamp | Model | Method | Correct | Total | Accuracy | Notes |\n"
            "|---|---|---|---:|---:|---:|---|\n",
            encoding="utf-8",
        )
    line = (
        f"| {ts} | {model} | {method} | {correct} | {total} | {accuracy:.4f} | {notes} |\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    root = repo_root()
    default_csv = root / "results" / "runs.csv"
    default_log = root / "EXPERIMENT_LOG.md"

    p = argparse.ArgumentParser(description="Run CA P&C broker MCQ eval (Ollama).")
    p.add_argument(
        "--public",
        type=Path,
        required=True,
        help="JSONL: id, question, choices[] (no answers).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="JSON with labels[id].correct_index (eval only).",
    )
    p.add_argument("--model", default="llama3.2", help="Ollama model name")
    p.add_argument(
        "--method",
        default="zero_shot",
        choices=("zero_shot", "cot"),
        help="First local sweep: zero_shot; cot adds reasoning instruction.",
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--csv", type=Path, default=default_csv)
    p.add_argument("--experiment-log", type=Path, default=default_log)
    p.add_argument("--notes", default="", help="Free-text note stored in CSV/log")
    p.add_argument("--print-details", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Score with placeholder predictions (always A). No Ollama; verifies CSV/log only.",
    )
    args = p.parse_args()

    if not args.public.is_file():
        raise SystemExit(f"Missing --public file: {args.public}")
    if not args.labels.is_file():
        raise SystemExit(f"Missing --labels file: {args.labels}")

    questions = load_jsonl(args.public)
    labels = load_labels(args.labels)
    if not questions and not args.dry_run:
        raise SystemExit(f"No questions loaded from {args.public}")

    if args.dry_run:
        correct, total, details = 0, 0, []
        for q in questions:
            qid = q["id"]
            if qid not in labels:
                continue
            gold = labels[qid]
            pred = 0
            ok = pred == gold
            if ok:
                correct += 1
            total += 1
            details.append(
                {
                    "id": qid,
                    "gold_index": gold,
                    "pred_index": pred,
                    "raw_tail": "[dry_run: forced A]",
                    "correct": ok,
                }
            )
    else:

        async def _run() -> tuple[int, int, list]:
            return await evaluate_split(
                questions=questions,
                labels=labels,
                base_url=args.ollama_url,
                model=args.model,
                method=args.method,
            )

        correct, total, details = asyncio.run(_run())
    acc = (correct / total) if total else 0.0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    notes = args.notes
    if args.dry_run:
        notes = (notes + " dry_run_forced_A").strip()

    row = {
        "utc_timestamp": ts,
        "model": args.model,
        "method": args.method,
        "ollama_url": args.ollama_url,
        "public_path": str(args.public),
        "labels_path": str(args.labels),
        "n_scored": total,
        "n_correct": correct,
        "accuracy": f"{acc:.6f}",
        "notes": notes,
    }
    append_csv(row, args.csv)
    append_experiment_log_md(
        log_path=args.experiment_log,
        ts=ts,
        model=args.model,
        method=args.method,
        correct=correct,
        total=total,
        accuracy=acc,
        notes=(notes or "").replace("|", "/"),
    )

    summary = {
        "utc_timestamp": ts,
        "model": args.model,
        "method": args.method,
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "csv": str(args.csv),
        "experiment_log": str(args.experiment_log),
    }
    print(json.dumps(summary, indent=2))
    if args.print_details:
        print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
