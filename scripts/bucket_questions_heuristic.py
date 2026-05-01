#!/usr/bin/env python3
"""
Heuristic bucketing (no API) for MCQs into the same 1–8 buckets as bucket_questions_openai.py.

This is a fallback when you don't have OPENAI_API_KEY set. It uses keyword rules, so it
won't match the LLM buckets perfectly, but it produces the same CSV + plot format.

Usage:
  .venv/bin/python3 scripts/bucket_questions_heuristic.py \
    --questions eval_set/from_quizlet_pdfs/questions_formatted.txt \
    --out-csv eval_set/from_quizlet_pdfs/question_buckets_heuristic.csv \
    --out-plot eval_set/from_quizlet_pdfs/question_buckets_heuristic.png
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent

# Reuse parser + plot writer from the OpenAI bucketer.
_spec = importlib.util.spec_from_file_location("_bucket_questions_openai", _SCRIPTS / "bucket_questions_openai.py")
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

parse_questions = _mod.parse_questions
BUCKET_INDEX = _mod.BUCKET_INDEX
write_plot = _mod.write_plot


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def bucket_for_text(stem: str, opts_blob: str) -> tuple[int, str]:
    t = _norm(stem + " " + opts_blob)

    # 3 CA laws & ethics
    if any(k in t for k in ["california", "commissioner", "department of insurance", "insurance code", "cdi", "cease", "desist", "license", "appointment", "fiduciary", "rebate", "unfair", "misrepresentation", "fraud", "surplus lines", "nonadmitted", "admitted"]):
        return 3, "keyword_ca_law"

    # 2 Contract law
    if any(k in t for k in ["contract", "offer", "acceptance", "consideration", "warranty", "representation", "concealment", "void", "voidable", "rescission", "endorsement", "policy provisions", "cancellation", "nonrenewal", "premium return", "loss payable", "mortgage clause", "standard fire policy"]):
        return 2, "keyword_contract"

    # 6 Valuation & government programs
    if any(k in t for k in ["coinsurance", "replacement cost", "actual cash value", "acv", "valued policy", "fair plan", "ca earthquake authority", "cea", "caarp", "assigned risk", "guaranty association", "valuation", "limit offered", "coverage limit"]):
        return 6, "keyword_valuation_programs"

    # 5 Residential property
    if any(k in t for k in ["homeowners", "dwelling", "ho ", "dp ", "residence", "condominium", "fire policy", "property disclosure", "earthquake", "wildfire", "personal property", "coverage a", "coverage b", "coverage c", "coverage d", "coverage e", "coverage f"]):
        return 5, "keyword_residential"

    # 7 Personal liability & inland marine
    if any(k in t for k in ["personal umbrella", "liability", "medical payments", "bodily injury", "property damage", "defense", "suit", "negligence", "defamation", "slander", "libel", "inland marine", "bailee", "floater", "accounts receivable", "valuable papers"]):
        return 7, "keyword_liability_inland"

    # 8 Commercial insurance
    if any(k in t for k in ["commercial", "cgl", "businessowners", "bop", "workers compensation", "workers' compensation", "employers liability", "equipment breakdown", "builders risk", "crime", "fidelity", "surety", "cargo", "ocean marine", "truck", "motor carrier", "umbrella"]):
        return 8, "keyword_commercial"

    # 4 Tort & property fundamentals
    if any(k in t for k in ["tort", "damages", "negligence", "proximate", "strict liability", "defendant", "plaintiff", "loss exposure", "hazard", "peril", "risk management", "avoidance", "retention", "transfer", "sharing"]):
        return 4, "keyword_tort_fundamentals"

    # 1 Basic concepts (default)
    return 1, "default"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-plot", type=Path, required=True)
    args = ap.parse_args()

    qs = parse_questions(args.questions)
    if not qs:
        raise SystemExit(f"No questions parsed from {args.questions}")

    rows: list[dict] = []
    for q in qs:
        opts_blob = " ".join(q.choices.get(k, "") for k in ("A", "B", "C", "D"))
        b, notes = bucket_for_text(q.stem, opts_blob)
        rows.append(
            {
                "question_number": q.number,
                "bucket": str(b),
                "bucket_name": BUCKET_INDEX[b],
                "model": "heuristic",
                "notes": notes,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_number", "bucket", "bucket_name", "model", "notes"])
        w.writeheader()
        w.writerows(rows)

    title = f"Topic buckets (heuristic, n={len(rows)})"
    write_plot(rows, args.out_plot, title)
    print(f"Wrote CSV -> {args.out_csv}")
    print(f"Wrote plot -> {args.out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

