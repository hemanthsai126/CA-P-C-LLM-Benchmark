#!/usr/bin/env python3
"""
Compare ``results/<source>/option/*.csv`` (columns question_number, answer) to the
matching ``answers.txt`` in the same folder, then write ``results.md`` with accuracy.

Scans (by default):
  - results/from_quizlet_pdfs/
  - results/from_youtube_video/
  - results/synthetic_data/

Example:
  .venv/bin/python3 scripts/write_option_accuracy_results_md.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ANS = re.compile(r"^(\d+)\s+([ABCD])\s*$", re.I)


def load_answers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ANS.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).upper()
    return out


def load_predictions(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                n = int(row.get("question_number", "").strip())
            except ValueError:
                continue
            opt = (row.get("option") or row.get("answer") or "").strip().upper()
            if opt in ("A", "B", "C", "D"):
                out[n] = opt
    return out


def compare(gold: dict[int, str], pred: dict[int, str]) -> tuple[int, int, int]:
    """Returns (compared, correct, missing_pred_or_gold)."""
    keys = sorted(set(gold) & set(pred))
    correct = sum(1 for k in keys if gold[k] == pred[k])
    missing = len(set(gold) | set(pred)) - len(keys)
    return len(keys), correct, missing


def write_md(
    *,
    out_path: Path,
    source_name: str,
    sections: list[tuple[str, Path, int, int, int, int]],
) -> None:
    lines = [
        f"# Option run vs answer key — `{source_name}`",
        "",
        "Local model outputs live under `option/*.csv` (columns `question_number`, `answer`, `reason`). "
        "Ground truth is `answers.txt` in this folder. **Accuracy** = fraction of question IDs where "
        "both files have a letter and the model’s `answer` matches the key.",
        "",
    ]
    for model_label, csv_path, compared, correct, n_gold, n_pred in sections:
        pct = (100.0 * correct / compared) if compared else 0.0
        lines += [
            f"## `{csv_path.name}`",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Answer-key rows | {n_gold} |",
            f"| Prediction rows (valid A–D) | {n_pred} |",
            f"| IDs compared (overlap) | {compared} |",
            f"| Correct | {correct} |",
            f"| Wrong | {compared - correct} |",
            f"| **Accuracy** | **{pct:.2f}%** |",
            "",
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_source(repo: Path, rel: str) -> Path | None:
    base = repo / "results" / rel
    ans = base / "answers.txt"
    opt_dir = base / "option"
    if not ans.is_file():
        print(f"SKIP {rel}: no answers.txt", file=sys.stderr)
        return None
    gold = load_answers(ans)
    if not gold:
        print(f"SKIP {rel}: empty answers", file=sys.stderr)
        return None
    if not opt_dir.is_dir():
        print(f"SKIP {rel}: no option/", file=sys.stderr)
        return None

    csvs = sorted(p for p in opt_dir.glob("*.csv") if p.is_file())
    if not csvs:
        print(f"SKIP {rel}: no option/*.csv", file=sys.stderr)
        return None

    sections: list[tuple[str, Path, int, int, int, int]] = []
    for csv_path in csvs:
        pred = load_predictions(csv_path)
        compared, correct, _ = compare(gold, pred)
        sections.append((csv_path.stem, csv_path, compared, correct, len(gold), len(pred)))

    out_md = base / "results.md"
    write_md(out_path=out_md, source_name=rel, sections=sections)
    print(f"Wrote {out_md.relative_to(repo)}", file=sys.stderr)
    return out_md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        type=str,
        default="from_quizlet_pdfs,from_youtube_video,synthetic_data",
        help="Comma-separated results subfolders",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    for rel in [s.strip() for s in args.sources.split(",") if s.strip()]:
        process_source(repo, rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
