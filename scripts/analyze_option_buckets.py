#!/usr/bin/env python3
"""
Per-bucket MCQ accuracy for each ``results/<source>/option/*.csv``.

Uses ``question_buckets_gpt-4.1.csv`` next to ``option/`` when present; for
``synthetic_data`` (no bucket file), uses the generator layout **50 questions × 8 buckets**
(same names as ``bucket_questions_openai.py``).

Writes under each ``option/analysis/``:

  - ``<model_stem>_by_bucket.png`` — bar chart of accuracy per bucket 1–8
  - ``heatmap_models_x_buckets.png`` — models × buckets (accuracy)
  - ``mean_accuracy_by_bucket.png`` — mean accuracy across models per bucket
  - ``bucket_accuracy_long.csv`` — long table: model, bucket, n, correct, accuracy
  - ``README.md`` — short text + best/worst bucket per model

Example::

  .venv/bin/python scripts/analyze_option_buckets.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import update_option_results_tables as ort

# Same curriculum names as scripts/bucket_questions_openai.py
BUCKET_INDEX: dict[int, str] = {
    1: "Basic concepts",
    2: "Contract law",
    3: "CA laws & ethics",
    4: "Tort & property fundamentals",
    5: "Residential property",
    6: "Valuation & government programs",
    7: "Personal liability & inland marine",
    8: "Commercial insurance",
}

SYNTHETIC_PER_BUCKET = 50


def load_bucket_map(source_dir: Path) -> dict[int, tuple[int, str]]:
    """question_number -> (bucket int 1..8, name)."""
    out: dict[int, tuple[int, str]] = {}
    csv_path = source_dir / "question_buckets_gpt-4.1.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    qn = int(row.get("question_number", "").strip())
                    b = int(row.get("bucket", "").strip())
                except ValueError:
                    continue
                if b < 1 or b > 8:
                    continue
                name = (row.get("bucket_name") or "").strip() or BUCKET_INDEX.get(b, str(b))
                out[qn] = (b, name)
        return out

    if source_dir.name == "synthetic_data":
        for b in range(1, 9):
            lo = (b - 1) * SYNTHETIC_PER_BUCKET + 1
            hi = b * SYNTHETIC_PER_BUCKET
            name = BUCKET_INDEX[b]
            for qn in range(lo, hi + 1):
                out[qn] = (b, name)
        return out

    raise FileNotFoundError(
        f"Missing {csv_path} and not synthetic_data; cannot assign buckets for {source_dir}"
    )


def short_model_label(stem: str) -> str:
    s = stem
    for suf in (
        "_reasoned_from_youtube_video",
        "_reasoned_from_quizlet_pdfs",
        "_reasoned_synthetic_data",
        "_reasoned",
    ):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s or stem


def bucket_stats_for_pred(
    gold: dict[int, str],
    pred: dict[int, str],
    qn_bucket: dict[int, tuple[int, str]],
) -> dict[int, tuple[int, int]]:
    """bucket_id -> (n_correct, n_total) on overlap qn in gold, pred, and bucket map."""
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # correct, total
    keys = set(gold) & set(pred) & set(qn_bucket)
    for qn in keys:
        b, _ = qn_bucket[qn]
        acc[b][1] += 1
        if gold[qn] == pred[qn]:
            acc[b][0] += 1
    return {b: (acc[b][0], acc[b][1]) for b in range(1, 9) if acc[b][1] > 0}


def list_option_csvs(option_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(option_dir.glob("*.csv")):
        if not p.is_file():
            continue
        if p.parent.name != "option":
            continue
        name = p.name.lower()
        if not (name.endswith("_reasoned.csv") or "_reasoned_from_" in name or "_reasoned_synthetic_data" in name):
            continue
        out.append(p)
    return out


def plot_model_by_bucket(stem: str, bucket_to_acc: dict[int, float], out: Path) -> None:
    xs = list(range(1, 9))
    ys = [bucket_to_acc.get(b, float("nan")) for b in xs]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = ["#4C78A8" if not np.isnan(y) else "#DDDDDD" for y in ys]
    bars = ax.bar(xs, [0 if np.isnan(y) else y for y in ys], color=colors, edgecolor="#333", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([BUCKET_INDEX[b] for b in xs], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (correct / overlap in bucket)")
    ax.set_xlabel("Bucket")
    ax.set_title(f"{short_model_label(stem)} — accuracy by bucket")
    ax.grid(axis="y", alpha=0.25)
    for bar, y in zip(bars, ys):
        if not np.isnan(y) and y > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(y + 0.03, 1.0),
                f"{y:.0%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_heatmap(mat: np.ndarray, model_labels: list[str], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.32 * len(model_labels))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{i+1}\n{BUCKET_INDEX[i+1][:14]}" for i in range(8)], fontsize=7)
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="0.1" if v > 0.45 else "0.95", fontsize=6)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def plot_mean_by_bucket(means: list[float], out: Path, title: str) -> None:
    xs = list(range(1, 9))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(xs, means, color="#72B7B2", edgecolor="#333", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([BUCKET_INDEX[b] for b in xs], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean accuracy across models")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def process_source(repo: Path, rel: str) -> None:
    base = repo / "results" / rel
    opt = base / "option"
    if not opt.is_dir():
        print(f"SKIP {rel}: no option/", file=sys.stderr)
        return
    ans_path = base / "answers.txt"
    if not ans_path.is_file():
        print(f"SKIP {rel}: no answers.txt", file=sys.stderr)
        return
    try:
        qn_bucket = load_bucket_map(base)
    except FileNotFoundError as e:
        print(f"SKIP {rel}: {e}", file=sys.stderr)
        return

    analysis = opt / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    gold = ort.load_answers(ans_path)
    csvs = list_option_csvs(opt)
    if not csvs:
        print(f"SKIP {rel}: no matching CSVs in option/", file=sys.stderr)
        return

    long_rows: list[dict[str, object]] = []
    matrix_rows: list[tuple[str, np.ndarray]] = []

    for csv_path in csvs:
        stem = csv_path.stem
        pred = ort.load_predictions(csv_path)
        stats = bucket_stats_for_pred(gold, pred, qn_bucket)
        bucket_to_acc: dict[int, float] = {}
        for b in range(1, 9):
            if b not in stats:
                bucket_to_acc[b] = float("nan")
                long_rows.append(
                    {
                        "model": stem,
                        "bucket": b,
                        "bucket_name": BUCKET_INDEX[b],
                        "n": 0,
                        "correct": 0,
                        "accuracy": "",
                    }
                )
                continue
            c, t = stats[b]
            acc = c / t if t else float("nan")
            bucket_to_acc[b] = acc
            long_rows.append(
                {
                    "model": stem,
                    "bucket": b,
                    "bucket_name": BUCKET_INDEX[b],
                    "n": t,
                    "correct": c,
                    "accuracy": round(acc, 5) if t else "",
                }
            )

        vec = np.array([bucket_to_acc.get(b, float("nan")) for b in range(1, 9)], dtype=float)
        matrix_rows.append((stem, vec))
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("_") or "model"
        plot_model_by_bucket(stem, bucket_to_acc, analysis / f"{safe}_by_bucket.png")

    # Heatmap: sort models by mean accuracy (non-nan)
    mat_list: list[np.ndarray] = []
    labels: list[str] = []
    for stem, vec in sorted(matrix_rows, key=lambda x: float(np.nanmean(x[1])), reverse=True):
        mat_list.append(vec)
        labels.append(short_model_label(stem))
    mat = np.vstack(mat_list) if mat_list else np.zeros((0, 8))
    plot_heatmap(
        mat,
        labels,
        analysis / "heatmap_models_x_buckets.png",
        f"{rel} — accuracy heatmap (models × buckets)",
    )

    mean_per_bucket = [float(np.nanmean(mat[:, j])) for j in range(8)]
    plot_mean_by_bucket(
        mean_per_bucket,
        analysis / "mean_accuracy_by_bucket.png",
        f"{rel} — mean accuracy across models per bucket",
    )

    long_path = analysis / "bucket_accuracy_long.csv"
    with long_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "bucket", "bucket_name", "n", "correct", "accuracy"],
        )
        w.writeheader()
        for row in long_rows:
            w.writerow(row)

    # README with best / worst bucket per model
    lines = [
        f"# Bucket ablation — `{rel}`",
        "",
        f"Ground truth: `../answers.txt` (relative to this folder). Eight curriculum buckets (1 = Basic concepts … 8 = Commercial insurance).",
        "",
        "## Per-model strongest / weakest bucket (by accuracy)",
        "",
        "| Model | Best bucket | Acc | Worst bucket | Acc |",
        "|-------|-------------|-----|--------------|-----|",
    ]
    stem_to_vec = {s: v for s, v in matrix_rows}
    for stem in sorted(stem_to_vec.keys(), key=lambda s: float(np.nanmean(stem_to_vec[s])), reverse=True):
        v = stem_to_vec[stem]
        valid = [(b + 1, v[b]) for b in range(8) if not np.isnan(v[b])]
        if not valid:
            continue
        best_b, best_a = max(valid, key=lambda t: t[1])
        worst_b, worst_a = min(valid, key=lambda t: t[1])
        bn = BUCKET_INDEX[best_b]
        wn = BUCKET_INDEX[worst_b]
        lines.append(
            f"| `{short_model_label(stem)}` | {best_b} {bn} | **{best_a:.1%}** | {worst_b} {wn} | **{worst_a:.1%}** |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `bucket_accuracy_long.csv` — all models × buckets",
            "- `heatmap_models_x_buckets.png` — overview",
            "- `mean_accuracy_by_bucket.png` — which buckets are hardest on average",
            "- `*_by_bucket.png` — one bar chart per model CSV",
            "",
        ]
    )
    (analysis / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {analysis}/ ({len(csvs)} models)", file=sys.stderr)


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
