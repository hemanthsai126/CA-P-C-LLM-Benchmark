#!/usr/bin/env python3
"""
Bar + heatmap of option-run accuracy vs answer key for each ``results/<source>/``.

Reads ``answers.txt`` and ``option/*.csv`` the same way as ``update_option_results_tables.py``.
Writes PNGs under ``results/charts/`` by default.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse scoring logic from sibling module (``python3 scripts/this.py`` → scripts/ on path).
import update_option_results_tables as ort


def model_key(csv_name: str) -> str:
    """Stable id across folders (OpenAI files embed source in the name)."""
    base = Path(csv_name).stem
    if "_reasoned_from_" in base:
        return base.split("_reasoned_from_", 1)[0]
    # Synthetic uses ``*_reasoned_synthetic_data``; quizlet/youtube use ``*_reasoned_from_*``.
    if base.endswith("_reasoned_synthetic_data"):
        return base[: -len("_reasoned_synthetic_data")]
    if base.endswith("_reasoned"):
        return base[: -len("_reasoned")]
    return base


def display_label(key: str, params: str) -> str:
    if params and params != "—":
        return f"{key} ({params})"
    return key


def collect_accuracies(repo: Path, sources: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """
    Returns:
      acc[key][source] = accuracy 0..100
      params[key] = parameters string (from first seen csv)
    """
    acc: dict[str, dict[str, float]] = {}
    params_map: dict[str, str] = {}

    for rel in sources:
        base = repo / "results" / rel
        ans_path = base / "answers.txt"
        opt_dir = base / "option"
        if not ans_path.is_file() or not opt_dir.is_dir():
            continue
        gold = ort.load_answers(ans_path)
        if not gold:
            continue
        for csv_path in sorted(opt_dir.glob("*.csv")):
            if not csv_path.is_file():
                continue
            pred = ort.load_predictions(csv_path)
            compared, correct = ort.compare(gold, pred)
            if compared == 0:
                continue
            key = model_key(csv_path.name)
            a = 100.0 * correct / compared
            acc.setdefault(key, {})[rel] = a
            if key not in params_map:
                params_map[key] = ort.parameters_for_csv(csv_path.name)

    return acc, params_map


def ordered_keys(acc: dict[str, dict[str, float]], sources: list[str]) -> list[str]:
    def mean_score(k: str) -> float:
        vals = [acc[k][s] for s in sources if s in acc[k]]
        return sum(vals) / len(vals) if vals else 0.0

    keys = list(acc.keys())
    keys.sort(key=lambda k: (-mean_score(k), k.lower()))
    return keys


def plot_grouped_barh(
    *,
    acc: dict[str, dict[str, float]],
    params_map: dict[str, str],
    sources: list[str],
    source_labels: list[str],
    out: Path,
) -> None:
    keys = ordered_keys(acc, sources)
    y = np.arange(len(keys), dtype=float)
    n = len(sources)
    width = 0.8 / max(n, 1)
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n) if n else [0.0]
    colors = ("#4C78A8", "#F58518", "#54A24B")

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.38 * len(keys))))
    for i, (src, off, lab) in enumerate(zip(sources, offsets, source_labels)):
        xs = [acc[k].get(src, float("nan")) for k in keys]
        ax.barh(y + off, xs, width, label=lab, color=colors[i % len(colors)], alpha=0.92)

    ax.set_yticks(y)
    ax.set_yticklabels([display_label(k, params_map.get(k, "")) for k in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy vs answer key (%)")
    ax.set_title("Option runs — accuracy by benchmark folder")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def plot_heatmap(
    *,
    acc: dict[str, dict[str, float]],
    params_map: dict[str, str],
    sources: list[str],
    source_labels: list[str],
    out: Path,
) -> None:
    keys = ordered_keys(acc, sources)
    mat = np.array([[acc[k].get(s, float("nan")) for s in sources] for k in keys], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, max(5.0, 0.35 * len(keys))))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(source_labels, rotation=15, ha="right")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([display_label(k, params_map.get(k, "")) for k in keys], fontsize=8)
    ax.set_title("Option runs — accuracy heatmap (%)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)")

    for i in range(len(keys)):
        for j in range(len(sources)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="0.15" if v > 55 else "0.95", fontsize=7)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sources",
        type=str,
        default="from_quizlet_pdfs,from_youtube_video,synthetic_data",
        help="Comma-separated results subfolders",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for PNGs (default: <repo>/results/charts)",
    )
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo / "results" / "charts"
    out_dir = out_dir.resolve()

    acc, params_map = collect_accuracies(repo, sources)
    if not acc:
        print("No data: check results/*/answers.txt and option/*.csv", file=sys.stderr)
        return 1

    labels_short = {
        "from_quizlet_pdfs": "Quizlet PDFs",
        "from_youtube_video": "YouTube",
        "synthetic_data": "Synthetic",
    }
    source_labels = [labels_short.get(s, s) for s in sources]

    plot_grouped_barh(
        acc=acc,
        params_map=params_map,
        sources=sources,
        source_labels=source_labels,
        out=out_dir / "option_accuracy_by_source_barh.png",
    )
    plot_heatmap(
        acc=acc,
        params_map=params_map,
        sources=sources,
        source_labels=source_labels,
        out=out_dir / "option_accuracy_by_source_heatmap.png",
    )
    print(f"Wrote {out_dir / 'option_accuracy_by_source_barh.png'}", file=sys.stderr)
    print(f"Wrote {out_dir / 'option_accuracy_by_source_heatmap.png'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
