#!/usr/bin/env python3
"""
Create simple charts from judge_runs_openai/<judge_model>/summary.csv.

Outputs PNGs next to the CSV by default:
  - avg_alignment_score.png
  - score_distribution_stacked.png
  - accuracy_vs_alignment.png
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short_run(name: str) -> str:
    name = name.replace("_reasoned", "")
    name = name.replace("openai_", "openai:")
    return name


def read_summary(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not (row.get("run") or "").strip():
                continue
            rows.append(row)
    return rows


def plot_avg_alignment(rows: list[dict[str, str]], out: Path) -> None:
    xs = [_short_run(r["run"]) for r in rows]
    ys = [float(r["avg_alignment_score"]) for r in rows]

    order = sorted(range(len(rows)), key=lambda i: ys[i], reverse=True)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    plt.figure(figsize=(10, max(3.2, 0.35 * len(xs))))
    plt.barh(xs[::-1], ys[::-1], color="#4C78A8")
    plt.xlim(0, 3)
    plt.xlabel("Average alignment score (0 = worst, 3 = better)")
    plt.title("Model explanations vs ground truth (higher is better)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_score_distribution_stacked(rows: list[dict[str, str]], out: Path) -> None:
    labels = [_short_run(r["run"]) for r in rows]
    p0 = [float(r["pct_score_0"]) for r in rows]
    p1 = [float(r["pct_score_1"]) for r in rows]
    p2 = [float(r["pct_score_2"]) for r in rows]
    p3 = [float(r["pct_score_3"]) for r in rows]

    order = sorted(range(len(rows)), key=lambda i: p3[i] + p2[i], reverse=True)
    labels = [labels[i] for i in order]
    p0 = [p0[i] for i in order]
    p1 = [p1[i] for i in order]
    p2 = [p2[i] for i in order]
    p3 = [p3[i] for i in order]

    plt.figure(figsize=(10, max(3.2, 0.35 * len(labels))))
    y = range(len(labels))
    left0 = [0.0] * len(labels)
    plt.barh(y, p0, left=left0, color="#E45756", label="0 worst")
    left1 = [p0[i] for i in y]
    plt.barh(y, p1, left=left1, color="#F58518", label="1 bad")
    left2 = [p0[i] + p1[i] for i in y]
    plt.barh(y, p2, left=left2, color="#72B7B2", label="2 ok")
    left3 = [p0[i] + p1[i] + p2[i] for i in y]
    plt.barh(y, p3, left=left3, color="#54A24B", label="3 better")

    plt.yticks(list(y), labels)
    plt.xlim(0, 1)
    plt.xlabel("Fraction of questions")
    plt.title("Score distribution (stacked)")
    plt.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_alignment(rows: list[dict[str, str]], out: Path) -> None:
    xs = [float(r["avg_alignment_score"]) for r in rows]
    ys = [float(r["accuracy"]) for r in rows]
    labels = [_short_run(r["run"]) for r in rows]

    plt.figure(figsize=(7.5, 6))
    plt.scatter(xs, ys, s=70, alpha=0.85, color="#4C78A8")
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlim(0, 3)
    plt.ylim(0, 1)
    plt.xlabel("Average alignment score (0–3)")
    plt.ylabel("MCQ accuracy (0–1)")
    plt.title("Accuracy vs explanation alignment (each dot is one model)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True, help="Path to summary.csv")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write PNGs (default: same folder as summary.csv)",
    )
    ap.add_argument(
        "--also-copy-to",
        type=Path,
        default=None,
        help="Optional extra directory to copy the three PNGs into (e.g. Results_plots for eval_set/from_youtube_video/Results.md previews)",
    )
    args = ap.parse_args()

    rows = read_summary(args.summary)
    if not rows:
        raise SystemExit("No rows found in summary.csv")

    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    names = (
        "avg_alignment_score.png",
        "score_distribution_stacked.png",
        "accuracy_vs_alignment.png",
    )
    plot_avg_alignment(rows, out_dir / "avg_alignment_score.png")
    plot_score_distribution_stacked(rows, out_dir / "score_distribution_stacked.png")
    plot_accuracy_vs_alignment(rows, out_dir / "accuracy_vs_alignment.png")

    print("Wrote:")
    for name in names:
        print(f"  {out_dir / name}")

    if args.also_copy_to is not None:
        args.also_copy_to.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(out_dir / name, args.also_copy_to / name)
        print("Also copied to:")
        for name in names:
            print(f"  {args.also_copy_to / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
