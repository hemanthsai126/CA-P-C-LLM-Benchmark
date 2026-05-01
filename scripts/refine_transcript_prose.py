#!/usr/bin/env python3
"""Strip tactiq-style timestamps and merge caption lines into plain English paragraphs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_TS_LINE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*")
# Do not strip "like" — it appears in legitimate phrases ("coverage like", "looks like").
_FILLER = re.compile(r"\b(uh|um|er|hm)\b\s*", re.I)
_SPACE = re.compile(r"\s+")

_POST_NORMALIZE = (
    ("flashc cards", "flash cards"),
    ("multiplechoice", "multiple-choice"),
    ("coins insurance", "coinsurance"),
    ("topics insurance regulation", "topics, including insurance regulation"),
)


def extract_body_lines(text: str) -> tuple[list[str], str]:
    """Return (comment_header_lines joined, body_fragment_lines)."""
    header: list[str] = []
    fragments: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# Refined:"):
                continue
            header.append(line)
            continue
        m = _TS_LINE.match(line)
        if m:
            fragments.append(line[m.end() :].strip())
        else:
            fragments.append(line)
    return "\n".join(header), " ".join(fragments)


def _word_wrap_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split at spaces so each block is at most max_chars (podcast ASR often has no sentence periods)."""
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] == " ":
            i += 1
        if i >= n:
            break
        if i + max_chars >= n:
            out.append(text[i:n].strip())
            break
        cut = text.rfind(" ", i + 1, i + max_chars + 1)
        if cut <= i:
            cut = i + max_chars
        out.append(text[i:cut].strip())
        i = cut + 1
    return [p for p in out if p]


def _join_paragraphs_keep_word_gaps(paragraphs: list[str]) -> str:
    """Double-newline between blocks; keep a word space when a break falls between letters/digits."""
    paras = [p.strip() for p in paragraphs if p.strip()]
    for j in range(len(paras) - 1):
        a, b = paras[j], paras[j + 1]
        if a[-1].isalnum() and b[0].isalnum():
            paras[j] = a + " "
    return "\n\n".join(paras)


def to_prose(blob: str, *, chars_per_paragraph: int = 550) -> str:
    blob = _SPACE.sub(" ", blob).strip()
    blob = _FILLER.sub("", blob)
    blob = _SPACE.sub(" ", blob).strip()
    # Light punctuation spacing
    blob = re.sub(r"\s+([.,;!?])", r"\1", blob)
    blob = re.sub(r",\s*", ", ", blob)
    for a, b in _POST_NORMALIZE:
        blob = blob.replace(a, b)

    if not blob:
        return blob
    paragraphs = _word_wrap_paragraphs(blob, chars_per_paragraph)
    return _join_paragraphs_keep_word_gaps(paragraphs)


def refine_file(path: Path, *, in_place: bool) -> None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    header, body = extract_body_lines(raw)
    prose = to_prose(body)
    note = "# Refined: timestamps removed; wrapped into paragraphs for readability.\n"
    out = (header + "\n\n" + note + "\n" + prose + "\n") if header else (note + "\n" + prose + "\n")
    target = path if in_place else path.with_name(path.stem + ".refined" + path.suffix)
    target.write_text(out, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--in-place", action="store_true", help="Overwrite each input file")
    args = ap.parse_args()
    for p in args.paths:
        refine_file(p, in_place=args.in_place)
        print("Wrote", p.resolve() if args.in_place else p.with_name(p.stem + ".refined" + p.suffix).resolve())


if __name__ == "__main__":
    main()
