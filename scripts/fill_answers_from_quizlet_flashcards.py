#!/usr/bin/env python3
"""
Match eval_set/from_quizlet_pdfs/questions.txt to Quizlet **flashcard** dumps (term = stem,
definition = correct answer phrase) and infer A–D by best text match.

Data sources (copied from public Quizlet HTML via markdown fetch):
  data/quizlet/quizlet_ca_flashcard_dump.md   — set 298981541 (74 terms), CA P&C exam
  data/quizlet/quizlet_ins_flashcard_dump.md  — set 236361078 (259 terms), P&C part one

This does **not** read the PDFs for keys (they are not there); it aligns stems to the
same Quizlet set backs, then picks the option closest to that definition.

Usage:
  .venv/bin/python3 scripts/fill_answers_from_quizlet_flashcards.py \\
    --questions eval_set/from_quizlet_pdfs/questions.txt \\
    --out eval_set/from_quizlet_pdfs/answers_from_flashcards_for_questions_txt.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _parse_questions_txt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    out: list[dict[str, Any]] = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        m = re.match(r"^(\d+)\.\s*(.+)$", lines[0])
        if not m:
            continue
        n = int(m.group(1))
        stem = m.group(2).strip()
        ch: dict[str, str] = {}
        for ln in lines[1:]:
            cm = re.match(r"^([A-D])\.\s*(.*)$", ln)
            if cm:
                ch[cm.group(1)] = cm.group(2).strip()
        if len(ch) == 4:
            out.append({"n": n, "stem": stem, "choices": ch})
    out.sort(key=lambda x: x["n"])
    return out


def _looks_like_new_question(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    if re.match(
        r"^(Which|What|Who|When|Where|How|Unless|Under|If|An?|The|All\b|No\b|To\b|In\b|"
        r"According|Damage|Bob|A\s+restaurant|A\s+meteorite|A\s+mortgage|A\s+producer|"
        r"Fast\s+bulk|Bulk\s+|With\s+regards|With\s+regard|Homeowners|Counteroffer|"
        r"Can\s+a|Could\s+|Should\s+|Would\s+|The\s+party|The\s+transfer|The\s+insured|"
        r"The\s+major|An\s+insured|An\s+applicant|A\s+person|A\s+licensee|A\s+producer|"
        r"When\s+a|When\s+can|If\s+more|If\s+an|If\s+a|In\s+a|In\s+property|"
        r"Under\s+the|No\s+rate|No\s+pr|To\s+which|To\s+be|Homeowners|Damage\s+to|"
        r"According\s+to|With\s+regards|How\s+|Where\s+)",
        t,
        re.I,
    ):
        return True
    if "?" in t and len(t) > 35:
        return True
    if re.match(r"^\d", t) and "?" in t:
        return True
    return False


def _is_suboption_line(line: str) -> bool:
    return bool(re.match(r"^[a-d]\)\s", line.strip(), re.I))


def _parse_flashcard_dump_ca(path: Path) -> list[tuple[str, str]]:
    """CA dump: one stem line (sometimes + eXCEPT line), then definition line."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(raw) if "Terms in this set (" in ln), None)
    if start is None:
        return []
    lines: list[str] = []
    for ln in raw[start:]:
        s = ln.strip()
        if not s or s.startswith("!["):
            continue
        if s.startswith("[") and "](" in s and "quizlet.com" in s:
            continue
        if s == "See more":
            break
        lines.append(s)

    pairs: list[tuple[str, str]] = []
    j = 0
    n = len(lines)
    while j < n:
        t = lines[j]
        j += 1
        if j < n and lines[j].strip() in ("eXCEPT", "EXCEPT", "except"):
            t += " " + lines[j]
            j += 1
        if j >= n:
            break
        d = lines[j]
        j += 1
        pairs.append((re.sub(r"\s+", " ", t).strip(), re.sub(r"\s+", " ", d).strip()))
    return pairs


def _parse_flashcard_dump_ins(path: Path) -> list[tuple[str, str]]:
    """INS dump: split on question-start lines; last line of each chunk is the definition."""
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(raw) if "Terms in this set (" in ln), None)
    if start is None:
        return []
    lines: list[str] = []
    for ln in raw[start:]:
        s = ln.strip()
        if not s or s.startswith("!["):
            continue
        if s.startswith("[") and "](" in s and "quizlet.com" in s:
            continue
        if s == "See more":
            break
        lines.append(s)

    starts = [i for i, ln in enumerate(lines) if _looks_like_new_question(ln)]
    if not starts:
        return []
    pairs: list[tuple[str, str]] = []
    for a, b in zip(starts, starts[1:] + [len(lines)]):
        chunk = lines[a:b]
        if len(chunk) < 2:
            continue
        defn = chunk[-1].strip()
        if _is_suboption_line(defn):
            continue
        term = re.sub(r"\s+", " ", " ".join(chunk[:-1])).strip()
        if len(term) < 12 or len(defn) < 3:
            continue
        pairs.append((term, defn))
    return pairs


def _parse_flashcard_dump(path: Path) -> list[tuple[str, str]]:
    if "ca" in path.name.lower():
        return _parse_flashcard_dump_ca(path)
    return _parse_flashcard_dump_ins(path)

def _dump_lines_after_terms(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i + 1 for i, ln in enumerate(raw) if "Terms in this set (" in ln), None)
    if start is None:
        return []
    out: list[str] = []
    for ln in raw[start:]:
        s = ln.strip()
        if not s or s.startswith("!["):
            continue
        if s.startswith("[") and "](" in s and "quizlet.com" in s:
            continue
        if s == "See more":
            break
        out.append(s)
    return out

def _best_line_idx(stem: str, lines: list[str]) -> tuple[int | None, float]:
    ns = _norm(stem)
    best_i: int | None = None
    best_r = 0.0
    for i, ln in enumerate(lines):
        nl = _norm(ln)
        if len(nl) < 12:
            continue
        r = SequenceMatcher(None, ns, nl).ratio()
        pfx = min(len(ns), len(nl), 52)
        if pfx > 22 and ns[:pfx] == nl[:pfx]:
            r = max(r, 0.78)
        if r > best_r:
            best_r = r
            best_i = i
    return best_i, best_r

def _definition_after_idx(lines: list[str], idx: int) -> str:
    """
    Given an index whose line best matches the stem, collect the following lines as the
    flashcard definition until the next question-like line.
    """
    j = idx + 1
    parts: list[str] = []
    while j < len(lines):
        ln = lines[j].strip()
        if not ln:
            j += 1
            continue
        if _looks_like_new_question(ln):
            break
        parts.append(ln)
        # keep multi-line definitions (a) b) c)), but stop before it turns into another term
        if len(" ".join(parts)) > 800:
            break
        j += 1
    return re.sub(r"\s+", " ", " ".join(parts)).strip()

def _best_flashcard(stem: str, bank: list[tuple[str, str]]) -> tuple[str, str, float] | None:
    ns = _norm(stem)
    best: tuple[str, str, float] | None = None
    for t, d in bank:
        nt = _norm(t)
        r = SequenceMatcher(None, ns, nt).ratio()
        pfx = min(len(ns), len(nt), 48)
        if pfx > 20 and ns[:pfx] == nt[:pfx]:
            r = max(r, 0.72)
        if best is None or r > best[2]:
            best = (t, d, r)
    if best is None or best[2] < 0.44:
        return None
    return best


def _def_to_letter(defn: str, choices: dict[str, str]) -> tuple[str, float]:
    nd = _norm(defn)
    best_l, best_r = "?", 0.0
    for L in ("A", "B", "C", "D"):
        no = _norm(choices[L])
        if len(no) < 6:
            continue
        r = SequenceMatcher(None, nd, no).ratio()
        if no in nd or nd in no:
            r = max(r, 0.88)
        if len(nd) > 25 and len(no) > 25:
            if nd[:28] == no[:28]:
                r = max(r, 0.75)
        if r > best_r:
            best_r, best_l = r, L
    return best_l, best_r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=Path("eval_set/from_quizlet_pdfs/questions.txt"))
    ap.add_argument("--ca-dump", type=Path, default=Path("data/quizlet/quizlet_ca_flashcard_dump.md"))
    ap.add_argument("--ins-dump", type=Path, default=Path("data/quizlet/quizlet_ins_flashcard_dump.md"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/answers_from_flashcards_for_questions_txt.txt"),
        help="Does not default to answers.txt (that file is the 800-Q formatted key).",
    )
    ap.add_argument(
        "--scores-out",
        type=Path,
        default=Path("eval_set/from_quizlet_pdfs/answers_from_flashcards_scores.tsv"),
        help="Write a TSV with stem-match + option-match scores for review.",
    )
    args = ap.parse_args()

    qs = _parse_questions_txt(args.questions)
    if not qs:
        raise SystemExit("No questions parsed from --questions")

    bank: list[tuple[str, str]] = []
    if args.ca_dump.is_file():
        bank.extend(_parse_flashcard_dump(args.ca_dump))
    if args.ins_dump.is_file():
        bank.extend(_parse_flashcard_dump(args.ins_dump))

    if not bank:
        raise SystemExit("No flashcard pairs; check --ca-dump / --ins-dump paths")

    ca_lines = _dump_lines_after_terms(args.ca_dump) if args.ca_dump.is_file() else []
    ins_lines = _dump_lines_after_terms(args.ins_dump) if args.ins_dump.is_file() else []
    all_lines = ca_lines + ins_lines

    filled = 0
    rows: list[tuple[int, str]] = []
    score_rows: list[tuple[int, float, float, str, str]] = []
    for item in qs:
        stem = item["stem"]
        # 1) fast path: use parsed bank (term/def)
        hit = _best_flashcard(stem, bank)
        best_def = ""
        stem_score = 0.0
        src = "bank"
        if hit:
            _, best_def, stem_score = hit
        # 2) fallback: search raw dump lines for the best matching stem line and read the following definition lines
        if not hit or stem_score < 0.55:
            idx, r = _best_line_idx(stem, all_lines)
            if idx is not None and r >= stem_score:
                d = _definition_after_idx(all_lines, idx)
                if d:
                    best_def = d
                    stem_score = r
                    src = "dump"

        if not best_def:
            # last resort: still choose something deterministic
            best_def = ""
            src = "none"

        letter, opt_score = _def_to_letter(best_def, item["choices"])
        if letter == "?":
            # choose best option even with very low score (user requested all filled)
            # recompute argmax without thresholding
            best_l, best_r = "A", -1.0
            for L in ("A", "B", "C", "D"):
                r = SequenceMatcher(None, _norm(best_def), _norm(item["choices"][L])).ratio()
                if r > best_r:
                    best_r, best_l = r, L
            letter, opt_score = best_l, max(opt_score, best_r)

        rows.append((item["n"], letter))
        score_rows.append((item["n"], stem_score, opt_score, src, best_def[:160].replace("\t", " ")))
        filled += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write(
            "# Inferred from Quizlet flashcard backs (same sets as the PDF URLs); "
            "see answers_inferred_scores.tsv for confidence and review.\n"
        )
        for n, L in rows:
            f.write(f"{n} {L}\n")

    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_out.open("w", encoding="utf-8") as f:
        f.write("n\tstem_match_score\toption_match_score\tsource\tdefinition_preview\n")
        for n, ss, os, src, prev in score_rows:
            f.write(f"{n}\t{ss:.4f}\t{os:.4f}\t{src}\t{prev}\n")

    print(
        f"Wrote {args.out} ({len(rows)} lines) and {args.scores_out}. "
        f"Filled letters: {filled}/{len(rows)}. Flashcard pairs loaded: {len(bank)}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
