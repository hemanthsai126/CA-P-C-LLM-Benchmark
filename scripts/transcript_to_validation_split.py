#!/usr/bin/env python3
"""
Turn timestamped exam transcript text into:
  - questions.txt  — numbered stem + A.–D. lines (no answers, no explanations)
  - answers.txt    — lines like `1 B`, `2 C`
  - explanations.txt — lines like `1 <explanation text>`

Strips `HH:MM:SS.mmm` timestamps, parses `question one|two|...|n|10`
and `answer X` markers (with ASR fallbacks: `answer beat`→B, `answer see`→C, `question n`→9).

Options: prefer peeling D→A from the right; if that fails, scan the last 3–4 ` A ` / ` B ` markers.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_TS = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*")
_QHEAD = re.compile(
    r"(?i)\bquestion\s+(one|two|three|four|five|six|seven|eight|nine|ten|n|\d+)\b"
)
_EXPL = re.compile(r"(?i)\bexplanation\b")

_WORD_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "n": 9,
}

_QSTRIP = re.compile(
    r"(?i)^\s*question\s+(one|two|three|four|five|six|seven|eight|nine|ten|n|\d+)\s+"
)


def strip_timestamps(text: str) -> str:
    text = _TS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def qword_to_int(w: str) -> int:
    w = w.lower()
    if w in _WORD_NUM:
        return _WORD_NUM[w]
    return int(w)


def peel_options_right_to_left(body: str) -> tuple[str, dict[str, str]]:
    tail = body.strip()
    choices: dict[str, str] = {}
    for letter in ("D", "C", "B", "A"):
        pat = re.compile(rf"(?is)^(.*)\s{letter}\s+(.+)$")
        m = pat.match(tail)
        if not m:
            raise ValueError(f"peel failed at {letter}")
        tail, opt = m.group(1).strip(), m.group(2).strip()
        choices[letter] = opt
    stem = _QSTRIP.sub("", tail, count=1).strip()
    return stem, choices


def scan_options_from_markers(before: str) -> tuple[str, dict[str, str]]:
    ms = list(re.finditer(r"(?i)\s([a-dA-D])\s+", before))
    if len(ms) < 3:
        raise ValueError("need at least 3 option markers")
    take = ms[-4:] if len(ms) >= 4 else ms[-3:]
    stem = before[: take[0].start()].strip()
    stem = _QSTRIP.sub("", stem, count=1).strip()
    choices: dict[str, str] = {"A": "", "B": "", "C": "", "D": ""}
    for i, m in enumerate(take):
        L = m.group(1).upper()
        a = m.end()
        b = take[i + 1].start() if i + 1 < len(take) else len(before)
        choices[L] = before[a:b].strip()
    return stem, choices


def extract_stem_choices(before: str) -> tuple[str, dict[str, str]]:
    try:
        return peel_options_right_to_left(before)
    except ValueError:
        return scan_options_from_markers(before)


def find_answer(block: str) -> tuple[int, str]:
    """Return (start_index, letter)."""
    fixed = [
        (r"(?i)\banswer\s+beat\b", "B"),
        (r"(?i)\banswer\s+be\b", "B"),
        (r"(?i)\banswer\s+the\s+insurable", "B"),
        (r"(?i)\banswer\s+see\b", "C"),
        (r"(?i)\band survey\s+to\b", "A"),
    ]
    for pat, letter in fixed:
        m = re.search(pat, block)
        if m:
            return m.start(), letter

    m = re.search(r"(?i)\bando\s+([a-dA-D])\b", block)
    if m:
        return m.start(), m.group(1).upper()
    m = re.search(r"(?i)\band so\s+([a-dA-D])", block)
    if m:
        return m.start(), m.group(1).upper()

    m = re.search(r"(?i)\banswer\s+([a-dA-D])(?=\s|subreg|explanation|$|,|\.)", block)
    if m:
        return m.start(), m.group(1).upper()
    raise ValueError(f"No answer marker in block: {block[:200]}…")


def split_questions(text: str) -> list[tuple[int, str]]:
    text = strip_timestamps(text)
    parts = _QHEAD.split(text)
    out: list[tuple[int, str]] = []
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        key = parts[i].strip().lower()
        body = parts[i + 1]
        n = qword_to_int(key)
        out.append((n, body.strip()))
    return out


def parse_block(block: str) -> tuple[str, dict[str, str], str, str]:
    """
    Return (stem, choices, answer_letter, explanation_text).
    explanation_text is "" if no explanation marker found.
    """
    m_expl = _EXPL.search(block)
    expl_text = ""
    if m_expl:
        expl_text = block[m_expl.end() :].strip()
        block_for_parse = block[: m_expl.start()].strip()
    else:
        block_for_parse = block

    apos, letter = find_answer(block_for_parse)
    before = block_for_parse[:apos].strip()
    stem, choices = extract_stem_choices(before)
    return stem, choices, letter, expl_text


def export_questions_txt(items: list[tuple[int, str, dict[str, str]]], path: Path) -> None:
    blocks: list[str] = []
    for n, stem, ch in items:
        lines = [f"{n}. {stem}"]
        for L in "ABCD":
            lines.append(f"{L}. {ch.get(L, '').strip()}")
        blocks.append("\n".join(lines))
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def export_answers_txt(rows: list[tuple[int, str]], path: Path) -> None:
    lines = [f"{n} {L}" for n, L in sorted(rows, key=lambda x: x[0])]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def export_explanations_txt(rows: list[tuple[int, str]], path: Path) -> None:
    """
    Write one explanation per line: `N <text>`.
    Explanation text is single-line normalized.
    """
    lines: list[str] = []
    for n, t in sorted(rows, key=lambda x: x[0]):
        clean = re.sub(r"\s+", " ", (t or "").strip())
        lines.append(f"{n} {clean}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_text", type=Path, help="UTF-8 transcript .txt")
    ap.add_argument("--questions-out", type=Path, required=True)
    ap.add_argument("--answers-out", type=Path, required=True)
    ap.add_argument("--explanations-out", type=Path, default=None, help="Optional explanations output path")
    args = ap.parse_args()

    raw = args.input_text.read_text(encoding="utf-8")
    chunks = split_questions(raw)
    by_num: dict[int, tuple[str, dict[str, str], str, str]] = {}
    errors: list[str] = []
    for n, block in chunks:
        try:
            stem, choices, letter, expl = parse_block(block)
        except ValueError as e:
            errors.append(f"Q{n}: {e}")
            continue
        by_num[n] = (stem, choices, letter, expl)

    if errors:
        import sys

        for line in errors[:25]:
            print("WARN:", line, file=sys.stderr)
        if len(errors) > 25:
            print(f"WARN: ... and {len(errors) - 25} more", file=sys.stderr)

    if not by_num:
        raise SystemExit("No questions parsed; fix errors above.")

    order = sorted(by_num)
    parsed = [(n, by_num[n][0], by_num[n][1]) for n in order]
    answers = [(n, by_num[n][2]) for n in order]
    expls = [(n, by_num[n][3]) for n in order]

    args.questions_out.parent.mkdir(parents=True, exist_ok=True)
    export_questions_txt(parsed, args.questions_out)
    export_answers_txt(answers, args.answers_out)
    if args.explanations_out is not None:
        args.explanations_out.parent.mkdir(parents=True, exist_ok=True)
        export_explanations_txt(expls, args.explanations_out)
    print(f"Wrote {len(parsed)} questions -> {args.questions_out}")
    print(f"Wrote {len(answers)} answers -> {args.answers_out}")
    if args.explanations_out is not None:
        print(f"Wrote {len(expls)} explanations -> {args.explanations_out}")


if __name__ == "__main__":
    main()
