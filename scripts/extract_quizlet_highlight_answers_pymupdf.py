#!/usr/bin/env python3
"""
Read a Quizlet Test PDF where the chosen answer is shown either as:
  - a green stroke around a 2x2 / list option cell, or
  - the darkest gray fill among four option tiles, or
  - (Definition cards) black text in the first answer slot above the grid.

Maps highlighted text to A–D using results/from_quizlet_pdfs/questions.txt and
writes ``N LETTER`` lines (552 questions).
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


def norm(s: str) -> str:
    s = s.lower()
    return re.sub(r"\s+", " ", s).strip()


def parse_questions_blocks(path: Path) -> list[tuple[str, dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=\d+\.\s)", text.strip())
    out: list[tuple[str, dict[str, str]]] = []
    for b in blocks:
        if not b.strip():
            continue
        lines = b.strip().split("\n")
        current = "stem"
        stem: list[str] = []
        opts = {k: "" for k in "ABCD"}
        for line in lines:
            m = re.match(r"^([A-D])\.\s*(.*)$", line)
            if m:
                current = m.group(1)
                opts[current] = m.group(2).strip()
            else:
                if current == "stem":
                    stem.append(line)
                else:
                    opts[current] = (opts[current] + " " + line).strip()
        stem_s = re.sub(r"^\d+\.\s*", "", " ".join(stem))
        stem_s = re.sub(r"\s+", " ", stem_s).strip()
        for k in "ABCD":
            opts[k] = re.sub(r"\s+", " ", opts[k]).strip()
        out.append((stem_s, opts))
    return out


def lum(rgb: tuple) -> float:
    r, g, b = rgb[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def green_stroke_rect(page) -> "fitz.Rect | None":
    import fitz

    for d in page.get_drawings():
        if d.get("type") != "s" or not d.get("color"):
            continue
        c = d["color"]
        r, g, b = c[:3]
        if g > 0.45 and r < 0.2 and b < 0.45:
            return fitz.Rect(d["rect"])
    return None


def candidate_option_fills(page) -> list[tuple[tuple, "fitz.Rect"]]:
    import fitz

    out: list[tuple[tuple, fitz.Rect]] = []
    for d in page.get_drawings():
        if d.get("type") != "f" or not d.get("fill") or d.get("fill") == (1, 1, 1):
            continue
        r = fitz.Rect(d["rect"])
        if r.width < 150 or r.height < 30 or r.height > 130:
            continue
        if r.y0 < 200 or r.y0 > 580:
            continue
        out.append((d["fill"], r))
    return out


def pool_fills(page) -> list[tuple[tuple, "fitz.Rect"]]:
    fills = candidate_option_fills(page)
    wide = [(f, r) for f, r in fills if r.width > 350]
    narrow = [(f, r) for f, r in fills if r.width <= 350]
    if len(wide) >= 2:
        return wide
    if len(narrow) >= 2:
        return narrow
    return fills


def cell_text(page, rect) -> str:
    import fitz

    rect = rect + (-3, -3, 3, 3)
    parts: list[str] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                br = fitz.Rect(s["bbox"])
                if br.intersects(rect):
                    parts.append(s["text"])
    return norm(" ".join(parts))


def definition_black_first_slot(page) -> str | None:
    """Join black (color 0) spans in the first answer band on Definition cards."""
    if "Definition" not in (page.get_text() or ""):
        return None
    parts: list[tuple[float, str]] = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for s in line.get("spans", []):
                bb = s["bbox"]
                if bb[1] < 85 or bb[1] > 210:
                    continue
                if s.get("color", 0) != 0:
                    continue
                t = (s.get("text") or "").strip()
                if len(t) < 2 or t == "Definition":
                    continue
                parts.append((bb[1], t))
    if not parts:
        return None
    parts.sort(key=lambda x: x[0])
    return norm(" ".join(t for _, t in parts))


def best_letter(opts: dict[str, str], extracted: str) -> tuple[str | None, float]:
    ex = norm(extracted)
    best: str | None = None
    best_sc = -1.0
    for L in "ABCD":
        t = norm(opts[L])
        if not t:
            continue
        if ex == t:
            return L, 1.0
        sc = SequenceMatcher(None, ex, t).ratio()
        if len(t) > 10 and (ex.startswith(t) or t.startswith(ex)):
            sc = max(sc, 0.92)
        if len(ex) > 10 and ex in t:
            sc = max(sc, 0.88)
        if len(t) > 10 and t in ex:
            sc = max(sc, 0.88)
        if sc > best_sc:
            best_sc, best = sc, L
    return best, best_sc


def answer_from_page(page, stem: str, opts: dict[str, str]) -> tuple[str | None, float, str]:
    import fitz

    stem_n = norm(stem)

    gr = green_stroke_rect(page)
    if gr is not None:
        txt = cell_text(page, gr)
        L, sc = best_letter(opts, txt)
        if L and sc >= 0.45:
            return L, sc, txt

    band = definition_black_first_slot(page)
    if band and len(band) >= 8:
        L, sc = best_letter(opts, band)
        if L and sc >= 0.72:
            return L, sc, band

    pool = pool_fills(page)
    pool_sorted = sorted(pool, key=lambda fr: lum(fr[0]))
    for _f, r in pool_sorted:
        txt = cell_text(page, r)
        stem_sim = SequenceMatcher(None, txt, stem_n).ratio()
        L, sc = best_letter(opts, txt)
        if stem_sim > 0.72 and sc < 0.55:
            continue
        if L and sc >= 0.48:
            return L, sc, txt

    for _f, r in pool_sorted:
        txt = cell_text(page, r)
        L, sc = best_letter(opts, txt)
        if L and sc >= 0.35:
            return L, sc, txt

    if band:
        L, sc = best_letter(opts, band)
        if L:
            return L, sc, band

    return None, 0.0, band or ""


def main() -> int:
    import fitz

    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdf",
        type=Path,
        default=root / "results/from_quizlet_pdfs/CAv 555 w answ.pdf",
    )
    ap.add_argument(
        "--questions",
        type=Path,
        default=root / "results/from_quizlet_pdfs/questions.txt",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "results/from_quizlet_pdfs/answers.txt",
    )
    args = ap.parse_args()
    if not args.pdf.is_file():
        print(f"error: PDF not found: {args.pdf}", file=sys.stderr)
        return 2
    if not args.questions.is_file():
        print(f"error: questions not found: {args.questions}", file=sys.stderr)
        return 2

    rows = parse_questions_blocks(args.questions)
    doc = fitz.open(str(args.pdf))
    if len(doc) < 553:
        print(f"error: expected >=553 pages, got {len(doc)}", file=sys.stderr)
        return 2

    lines_out: list[str] = []
    low: list[tuple[int, float, str]] = []
    for qi in range(1, 553):
        page = doc[qi]
        stem, opts = rows[qi - 1]
        L, sc, _snippet = answer_from_page(page, stem, opts)
        if not L or sc < 0.4:
            low.append((qi, sc, _snippet[:120]))
            L = L or "?"
        lines_out.append(f"{qi} {L}")

    hdr = (
        "# Answer key from CAv 555 w answ.pdf (Quizlet highlight: green stroke / "
        "darkest option tile / Definition first-slot black text), matched to "
        "results/from_quizlet_pdfs/questions.txt via PyMuPDF.\n"
        "# Rows marked ? (e.g. 443, 538): PDF has merged/garbled cards with no reliable highlight vs A–D.\n"
    )
    args.out.write_text(hdr + "\n".join(lines_out) + "\n", encoding="utf-8")
    if low:
        print(f"warning: {len(low)} questions had weak match (written as best-effort or ?):", file=sys.stderr)
        for t in low[:30]:
            print(" ", t, file=sys.stderr)
        if len(low) > 30:
            print(f"  ... and {len(low) - 30} more", file=sys.stderr)
    print(f"wrote {args.out} ({len(lines_out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
