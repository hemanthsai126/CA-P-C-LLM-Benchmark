#!/usr/bin/env python3
"""
Rebuild ``results/<source>/results.md`` as a single sorted table (CSV | Parameters |
Correct | Wrong | Accuracy) from ``answers.txt`` and every ``option/*.csv``.

Example:
  .venv/bin/python3 scripts/update_option_results_tables.py --sources from_quizlet_pdfs,synthetic_data
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ANS = re.compile(r"^(\d+)\s+([ABCD?])\s*$", re.I)


def load_answers(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ANS.match(line)
        if m and m.group(2).upper() in ("A", "B", "C", "D"):
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


def compare(gold: dict[int, str], pred: dict[int, str]) -> tuple[int, int]:
    keys = sorted(set(gold) & set(pred))
    correct = sum(1 for k in keys if gold[k] == pred[k])
    return len(keys), correct


def parameters_for_csv(name: str) -> str:
    """Published sizes for open models; estimates / not disclosed for OpenAI API CSVs."""
    n = name.lower()
    if "tinyllama" in n:
        return "1.1B"
    if "gemma2_9b" in n:
        return "9B"
    if "gemma_2b" in n:
        return "2B"
    if "phi3" in n:
        return "3.8B"
    if "qwen2.5" in n or "qwen2_5" in n:
        return "7B"
    if "mistral" in n:
        return "7B"
    if "llama3.1" in n or "llama3_1" in n:
        return "8B"
    if "gpt-4o_reasoned" in n and "mini" not in n:
        return "Not disclosed"
    if "gpt-4o-mini" in n:
        return "~8B (est.)"
    if "gpt-4_1-mini" in n:
        return "~7B (est.)"
    if "gpt-4_1-nano" in n:
        return "~4B (est.)"
    if "gpt-3_5" in n or "gpt-3.5" in n:
        return "Not disclosed"
    return "—"


def intro_for_source(rel: str) -> str:
    if rel == "from_quizlet_pdfs":
        return (
            "Predictions live under `option/*.csv` (`question_number`, `answer` or `option`, `reason`). "
            "Ground truth is `answers.txt`. Overlap excludes rows where the key is not A–D. "
            "**Accuracy** = correct / overlapped IDs."
        )
    if rel == "from_youtube_video":
        return (
            "Local outputs under `option/*.csv`: Ollama uses `answer`; OpenAI CSVs use **`option`**. "
            "Ground truth is `answers.txt` (**150** IDs). **Accuracy** = correct / overlap."
        )
    if rel == "synthetic_data":
        return (
            "Outputs under `option/*.csv`: Ollama uses `answer`; OpenAI runs use **`option`**. "
            "Ground truth is `answers.txt` (**400** IDs). **Accuracy** = correct / overlap."
        )
    return ""


def footnote() -> str:
    return (
        "Rows sorted by accuracy (highest first). **Parameters:** Ollama weights use published sizes "
        "where listed (TinyLlama 1.1B, Gemma 2 2B, Gemma 2 9B, Phi-3 Mini 3.8B, Qwen2.5-7B, Mistral 7B, "
        "Llama 3.1 8B). OpenAI API models: “Not disclosed” / “est.” as in `synthetic_data/results.md`."
    )


def write_table_md(*, out_path: Path, rel: str, rows: list[tuple[str, str, int, int, int, float]]) -> None:
    """rows: (csv_name, params, compared, correct, wrong, accuracy_pct)"""
    lines = [
        f"# Option run vs answer key — `{rel}`",
        "",
        intro_for_source(rel),
        "",
        "| CSV | Parameters | Correct | Wrong | Accuracy |",
        "|-----|------------|--------:|------:|----------:|",
    ]
    for name, params, comp, ok, wrong, acc in rows:
        lines.append(f"| `{name}` | {params} | {ok} | {wrong} | **{acc:.2f}%** |")
    lines.append("")
    lines.append(footnote())
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def process_source(repo: Path, rel: str) -> None:
    base = repo / "results" / rel
    ans_path = base / "answers.txt"
    opt_dir = base / "option"
    if not ans_path.is_file():
        print(f"SKIP {rel}: no answers.txt", file=sys.stderr)
        return
    if not opt_dir.is_dir():
        print(f"SKIP {rel}: no option/", file=sys.stderr)
        return
    gold = load_answers(ans_path)
    if not gold:
        print(f"SKIP {rel}: empty answers", file=sys.stderr)
        return

    scored: list[tuple[str, str, int, int, int, float]] = []
    for csv_path in sorted(opt_dir.glob("*.csv")):
        if not csv_path.is_file():
            continue
        pred = load_predictions(csv_path)
        compared, correct = compare(gold, pred)
        if compared == 0:
            continue
        wrong = compared - correct
        acc = 100.0 * correct / compared
        params = parameters_for_csv(csv_path.name)
        scored.append((csv_path.name, params, compared, correct, wrong, acc))

    scored.sort(key=lambda r: r[5], reverse=True)
    write_table_md(out_path=base / "results.md", rel=rel, rows=scored)
    print(f"Wrote {base / 'results.md'} ({len(scored)} models)", file=sys.stderr)


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
