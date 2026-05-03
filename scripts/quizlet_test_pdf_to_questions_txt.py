#!/usr/bin/env python3
"""
Convert a Quizlet **Test** print/save PDF (Term/Definition + four choices + ``N of M`` footer)
into ``questions.txt`` block format (same style as ``results/from_youtube_video/questions.txt``):

  N. stem text
  A. ...
  B. ...
  C. ...
  D. ...

``Term`` blocks: stem first, then four answer lines.
``Definition`` blocks: four answer lines first, then stem (Quizlet flips the card).

Example:
  .venv/bin/python3 scripts/quizlet_test_pdf_to_questions_txt.py \\
    --pdf results/from_quizlet_pdfs/CA555.pdf \\
    --out results/from_quizlet_pdfs/questions.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader


_JUNK_LINE = re.compile(
    r"^(https?://|.*quizlet\.com|Page \d+ of \d+|\d+/\d+/\d+.*\b(AM|PM)\b|Name:\s*Score:|\d+ Multiple choice questions)$",
    re.I,
)
_COUNTER = re.compile(r"^(\d+) of (\d+)$")


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_lines(pdf: Path) -> list[str]:
    reader = PdfReader(str(pdf))
    lines: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        for raw in t.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            if _JUNK_LINE.match(ln):
                continue
            if "quizlet.com" in ln.lower():
                continue
            lines.append(ln)
    return lines


def parse_cards(lines: list[str]) -> list[tuple[str, list[str]]]:
    """
    Return list of (label, body_lines) where label is 'Term' or 'Definition',
    body_lines excludes the trailing ``N of M`` counter line.
    """
    cards: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln in ("Term", "Definition"):
            label = ln
            i += 1
            body: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt in ("Term", "Definition"):
                    break
                m = _COUNTER.match(nxt)
                if m:
                    i += 1
                    if body:
                        cards.append((label, body))
                    break
                body.append(nxt)
                i += 1
            continue
        i += 1
    return cards


def card_to_mcq(label: str, body: list[str]) -> tuple[str, str, str, str, str] | None:
    if len(body) < 5:
        return None
    if label == "Term":
        stem_lines, opts = body[:-4], body[-4:]
    else:
        opts, stem_lines = body[:4], body[4:]
    stem = _squish(" ".join(stem_lines))
    a, b, c, d = (_squish(x) for x in opts)
    if len(stem) < 10 or min(len(a), len(b), len(c), len(d)) < 2:
        return None
    if len({a, b, c, d}) < 4:
        return None
    return stem, a, b, c, d


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", type=Path, default=root / "results/from_quizlet_pdfs/CA555.pdf")
    ap.add_argument("--out", type=Path, default=root / "results/from_quizlet_pdfs/questions.txt")
    args = ap.parse_args()

    if not args.pdf.is_file():
        print(f"error: PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    lines = extract_lines(args.pdf)
    cards = parse_cards(lines)
    mcqs: list[tuple[str, str, str, str, str]] = []
    for label, body in cards:
        row = card_to_mcq(label, body)
        if row:
            mcqs.append(row)

    if not mcqs:
        print("error: no MCQs parsed; PDF layout may differ from Quizlet test export", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for n, (stem, a, b, c, d) in enumerate(mcqs, start=1):
            f.write(f"{n}. {stem}\n")
            f.write(f"A. {a}\n")
            f.write(f"B. {b}\n")
            f.write(f"C. {c}\n")
            f.write(f"D. {d}\n")
            f.write("\n")

    print(f"Wrote {len(mcqs)} questions → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
