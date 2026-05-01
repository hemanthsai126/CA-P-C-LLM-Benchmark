#!/usr/bin/env python3
"""
Extract MCQs from a PDF where each item is:

  N <question line(s)>
  a. <option>
  b. <option>
  c. <option>
  d. <option>
  <explanation paragraph (answer is implied at the start)>

Writes a CSV with columns:
  question_number, question, option_A, option_B, option_C, option_D, answer, explanation
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


@dataclass
class Item:
    n: int
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str  # A/B/C/D or ""
    explanation: str


OPT_RE = re.compile(r"^(?P<k>[a-dA-D])\.\s*(?P<v>.+?)\s*$")
# Some PDF extractions remove the space after the question number (e.g. "1Which ...").
Q_RE = re.compile(r"^(?P<n>\d{1,3})\s*(?P<t>.+?)\s*$")


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 $%-]+", "", s)
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()

def token_overlap_score(option: str, head: str) -> float:
    a = [t for t in norm(option).split() if t]
    b = set([t for t in norm(head).split() if t])
    if not a:
        return 0.0
    hit = sum(1 for t in a if t in b)
    return hit / len(a)


def guess_answer(option_map: dict[str, str], explanation: str) -> str:
    """
    Heuristic: compare the first sentence (or first ~140 chars) of explanation to each option.
    Many of these PDFs start the explanation with the correct option text.
    """
    exp = explanation.strip()
    if not exp:
        return ""
    first = exp.split("\n", 1)[0].strip()
    # Take up to first period if it seems like a short leading sentence.
    if "." in first and len(first.split(".", 1)[0]) > 3:
        first = first.split(".", 1)[0].strip()
    head = (first if first else exp)[:140]

    head_n = norm(head)
    best_k = ""
    best_s = 0.0
    for k in ("A", "B", "C", "D"):
        v = option_map.get(k, "").strip()
        if not v:
            continue
        s1 = similarity(norm(v), head_n)
        s2 = token_overlap_score(v, head)
        s = max(s1, s2)
        if s > best_s:
            best_s = s
            best_k = k

    # Accept only if it's not a total guess.
    return best_k if best_s >= 0.25 else ""


def extract_text(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    lines: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        # Normalize line breaks
        for ln in t.splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def parse_items(lines: list[str], *, start: int = 1, end: int = 50) -> list[Item]:
    # Filter obvious headers/footers.
    cleaned: list[str] = []
    promo_patterns = [
        r"\bPROPERTY AND CASUALTY INSURANCE AGENT LICENSING PRACTICE EXAM\b",
        r"\bTake this free property and casualty insurance practice test\b",
        r"\bOur property and casualty insurance test prep\b",
        r"\binsurance instructors\b",
        r"\bfull-length 500 question\b",
        r"\bIf you are interested in life and health\b",
        r"\bSee the below links\b",
        r"\bPractice Exam\b",
        r"\bSupportLogin\b",
        r"\bSupport Login\b",
        r"\bScore My Practice Test\b",
        r"\bRELATED LINKS\b",
        r"\bQUALITY STARTS WITH WHO WROTE THE MATERIAL\b",
        r"\bTerms of Use\b",
        r"\bPrivacy Policy\b",
        r"\bAll Rights Reserved\b",
        r"\bPASS Guarantee\b",
    ]
    for ln in lines:
        if ln.startswith("-- ") and " of " in ln:
            continue
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2},", ln):
            continue
        if ln.startswith("http://") or ln.startswith("https://"):
            continue
        if ln in {"Study Online Instantly", "Click to Save 50% Now"}:
            continue
        # Remove year/edition header lines that can look like question numbers (e.g. "2026 EDITION").
        if re.match(r"^\d{4}\s+EDITION\b", ln, flags=re.I):
            continue
        if any(re.search(p, ln, flags=re.I) for p in promo_patterns):
            continue
        cleaned.append(ln)

    items: list[Item] = []
    i = 0
    while i < len(cleaned):
        m = Q_RE.match(cleaned[i])
        if not m:
            i += 1
            continue

        n = int(m.group("n"))
        if n < start:
            i += 1
            continue
        # Some headers start with big numbers (e.g. 2026). Ignore those.
        if n > end:
            if n > 200:
                i += 1
                continue
            break

        q_parts = [m.group("t").strip()]
        i += 1

        # Collect question continuation lines until option a.
        while i < len(cleaned) and not OPT_RE.match(cleaned[i]):
            # Stop if we accidentally hit next question
            if Q_RE.match(cleaned[i]):
                break
            q_parts.append(cleaned[i])
            i += 1

        # Collect options a-d (may wrap).
        opts: dict[str, list[str]] = {"A": [], "B": [], "C": [], "D": []}
        current_k: str | None = None
        while i < len(cleaned):
            om = OPT_RE.match(cleaned[i])
            if om:
                current_k = om.group("k").upper()
                opts[current_k] = [om.group("v").strip()]
                i += 1
                continue

            # End of options when we hit the explanation (usually not starting with a./b./c./d.)
            if current_k and not OPT_RE.match(cleaned[i]):
                # If next question starts, we failed to parse explanation.
                if Q_RE.match(cleaned[i]):
                    break
                # Treat as option wrap if we haven't seen all four yet and line looks like continuation.
                if len([k for k, v in opts.items() if v]) < 4 and cleaned[i][0].islower() is False:
                    # Often explanations start with caps too; avoid infinite option wrapping.
                    pass
                # If we already have all options, stop option parsing.
                if all(opts[k] for k in ("A", "B", "C", "D")):
                    break
                # Otherwise, treat as continuation of the last option
                if current_k:
                    opts[current_k].append(cleaned[i])
                i += 1
                continue

            break

        option_map = {k: " ".join(v).strip() for k, v in opts.items()}

        # Explanation: collect until the next question number.
        exp_parts: list[str] = []
        while i < len(cleaned) and not Q_RE.match(cleaned[i]):
            exp_parts.append(cleaned[i])
            i += 1

        explanation = " ".join(exp_parts).strip()
        answer = guess_answer(option_map, explanation)

        items.append(
            Item(
                n=n,
                question=" ".join(q_parts).strip(),
                A=option_map.get("A", ""),
                B=option_map.get("B", ""),
                C=option_map.get("C", ""),
                D=option_map.get("D", ""),
                answer=answer,
                explanation=explanation,
            )
        )

    return items


def write_csv(items: list[Item], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "question_number",
                "question",
                "option_A",
                "option_B",
                "option_C",
                "option_D",
                "answer",
                "explanation",
            ]
        )
        for it in items:
            w.writerow([it.n, it.question, it.A, it.B, it.C, it.D, it.answer, it.explanation])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=50)
    args = ap.parse_args()

    lines = extract_text(args.pdf)
    items = parse_items(lines, start=args.start, end=args.end)
    write_csv(items, args.out)
    print(f"Wrote {len(items)} rows to {args.out}")
    # Warn on missing answers so you can spot-check.
    missing = [it.n for it in items if not it.answer]
    if missing:
        print(f"WARNING: Could not confidently infer answer for: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

