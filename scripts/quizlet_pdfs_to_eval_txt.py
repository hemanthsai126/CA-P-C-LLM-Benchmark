#!/usr/bin/env python3
"""
Convert Quizlet-exported insurance PDFs into eval_set-style questions.txt + a **PDF-parse** answer sidecar
(named ``answers_from_quizlet_pdf_parse.txt`` so it does not overwrite the curated ``answers.txt`` for the 800-Q set).

- **INS QAA**-style PDFs (options labeled ``1.`` … ``4.`` and an answer line that repeats
  the correct option) are parsed **locally** from text extracted after splitting on Quizlet URLs.

- **INS QA** / **CA INS QA**-style Quizlet **Test** PDFs (``k of M`` progress rows, four
  choices, ``Don't know``) are parsed **locally** by pairing each progress marker with the
  choice block above the matching ``Don't know`` row, with small PDF-specific heuristics.

- If a PDF does not match either pattern, pass ``--openai-model gpt-4.1`` (and
  ``OPENAI_API_KEY``) so URL-chunks can be reconstructed via the API.

Outputs (default ``--out-dir``):
  questions.txt
  answers_from_quizlet_pdf_parse.txt
  (plus ``sources.csv`` and ``answers_key_TEMPLATE.tsv`` sidecars)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pypdf import PdfReader

PdfKind = Literal["ins_qaa", "interactive"]

try:
    from openai import BadRequestError, OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def pdf_lines(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    out: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        for raw in t.splitlines():
            ln = " ".join(raw.split())
            if ln:
                out.append(ln)
    return out


def url_chunks(lines: list[str]) -> list[str]:
    blob = "\n".join(lines)
    parts = re.split(r"https://quizlet\.com\S*", blob, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


_JUNK = re.compile(
    r"^(?:"
    r"\d+/\d+|"
    r"\d+\s+of\s+\d+|"
    r"T\s*e\s*r\s*m|"
    r"Don\s*'\s*t\s+know\s*\?|"
    r"Give this one a go|"
    r"No problem\.|"
    r"No worries\.|"
    r"Not quite\.|"
    r"You'?ve got this!|"
    r"You'?re doing great!|"
    r"Excellent!|"
    r"Brilliant work!|"
    r"Retest using|"
    r"Turn these into|"
    r"Correct|"
    r"Incorrect|"
    r"Your time:|"
    r"Your answer|"
    r"Study Online|"
    r"Click to Save|"
    r"Property and Casualty Insurance Exam\| Quizlet|"
    r"California Property and Casualty Insurance Exam\| Quizlet|"
    r"CA Property and Casualty Insurance Practice Exam Flashcards\| Quizlet|"
    r"^\d{1,2}/\d{1,2}/\d{2},"
    r")",
    re.I,
)


def is_junk_line(ln: str) -> bool:
    if not ln.strip():
        return True
    if _JUNK.match(ln.strip()):
        return True
    if "quizlet.com" in ln.lower():
        return True
    return False


_OPT_START = re.compile(r"^([1-4])\s*\.\s*(.*)$")

def _line_starts_question(ln: str) -> bool:
    """Detect MCQ stem first line despite PDF inserting spaces inside words."""
    x = re.sub(r"\s+", "", ln[:88]).lower()
    prefixes = (
        "which",
        "what",
        "when",
        "who",
        "how",
        "where",
        "why",
        "allofthefollowing",
        "aspecial",
        "itisunlawful",
        "itis",
        "the",
        "california",
        "except",
        "under",
        "ifa",
        "during",
        "whois",
        "howlong",
        "whena",
        "apolicy",
        "aninsured",
        "incalifornia",
    )
    if any(x.startswith(p) for p in prefixes):
        return True
    if x.startswith("all") and "following" in ln.lower():
        return True
    return False


def _clean_ins_qaa_lines(ch: str) -> list[str]:
    lines = [ln.strip() for ln in ch.splitlines() if ln.strip()]
    return [ln for ln in lines if not is_junk_line(ln)]


def _read_four_numbered_options(lines: list[str], i: int) -> tuple[dict[str, str] | None, int]:
    """From line i (must start with ``1.``), read options 1–4; return (flat, index of answer/next line)."""
    if i >= len(lines):
        return None, i
    m0 = _OPT_START.match(lines[i])
    if not m0 or m0.group(1) != "1":
        return None, i
    seq = ["1", "2", "3", "4"]
    opts: dict[str, list[str]] = {k: [] for k in seq}
    si = 0
    cur = "1"
    opts["1"].append(m0.group(2).strip())
    i += 1
    four_started = False
    while i < len(lines):
        m = _OPT_START.match(lines[i])
        if not m:
            opts[cur].append(lines[i])
            i += 1
            continue
        d, rest = m.group(1), m.group(2).strip()
        if four_started and d in seq:
            break
        if d == cur:
            opts[cur].append(rest)
            i += 1
            continue
        if si < 3 and d == seq[si + 1]:
            si += 1
            cur = seq[si]
            opts[cur].append(rest)
            i += 1
            if cur == "4":
                four_started = True
            continue
        break

    flat = {k: " ".join(v).strip() for k, v in opts.items()}
    if not all(flat[k] for k in seq):
        return None, i
    return flat, i


def parse_ins_qaa_chunks(chunks: list[str]) -> list[dict[str, Any]]:
    """Parse numeric 1–4 option style blocks (INS QAA.pdf)."""
    items: list[dict[str, Any]] = []
    for ch in chunks:
        lines = _clean_ins_qaa_lines(ch)
        starts = [i for i, ln in enumerate(lines) if _line_starts_question(ln)]
        if not starts:
            continue
        for si, start in enumerate(starts):
            end = starts[si + 1] if si + 1 < len(starts) else len(lines)
            block = lines[start:end]
            opt_idx = None
            for k, ln in enumerate(block):
                if _OPT_START.match(ln) and ln.lstrip().startswith("1."):
                    opt_idx = k
                    break
            if opt_idx is None:
                continue
            stem = " ".join(block[:opt_idx]).strip()
            if len(stem) < 25:
                continue
            sub = block[opt_idx:]
            flat, j = _read_four_numbered_options(sub, 0)
            if flat is None:
                continue

            ans_digit = ""
            k = j
            if k < len(sub):
                am = _OPT_START.match(sub[k])
                if am:
                    tail = [am.group(2).strip()]
                    d0 = am.group(1)
                    k += 1
                    while k < len(sub) and not _OPT_START.match(sub[k]):
                        if _line_starts_question(sub[k]):
                            break
                        tail.append(sub[k])
                        k += 1
                        if len(" ".join(tail)) > 600:
                            break
                    rest = " ".join(tail).strip()
                    rlow = re.sub(r"\s+", " ", rest.lower())
                    for d in ("1", "2", "3", "4"):
                        olow = re.sub(r"\s+", " ", flat[d].lower())
                        if len(olow) >= 15 and olow[: min(60, len(olow))] in rlow:
                            ans_digit = d
                            break
                    if not ans_digit and d0 in ("1", "2", "3", "4"):
                        olow = re.sub(r"\s+", " ", flat[d0].lower())
                        if olow[:40] in rlow:
                            ans_digit = d0

            if ans_digit not in ("1", "2", "3", "4"):
                continue

            letter = "ABCD"[int(ans_digit) - 1]
            items.append(
                {
                    "stem": stem,
                    "A": flat["1"],
                    "B": flat["2"],
                    "C": flat["3"],
                    "D": flat["4"],
                    "answer": letter,
                }
            )
    return items


def _compact_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _is_progress_line(ln: str) -> tuple[int, int] | None:
    c = _compact_alnum(ln)
    m = re.search(r"(\d+)of(\d+)$", c) or re.match(r"^(\d+)of(\d+)", c)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _is_dont_know(ln: str) -> bool:
    return "dontknow" in _compact_alnum(ln)


def _is_feedback_line(ln: str) -> bool:
    c = _compact_alnum(ln)
    keys = (
        "givethisoneagolater",
        "givethisoneago",
        "noworries",
        "notquite",
        "noproblem",
        "brilliant",
        "excellent",
        "yourestilllearning",
        "youvegothis",
        "youredoinggreat",
    )
    return any(k in c for k in keys)


def _is_interactive_junk(ln: str) -> bool:
    if "quizlet.com" in ln.lower():
        return True
    if re.match(r"^\d{1,2}/\d{1,2}/\d{2},", ln):
        return True
    c = _compact_alnum(ln)
    if "retest" in c and "missed" in c:
        return True
    if "turn" in c and "questions" in c:
        return True
    if "| quizlet" in ln.lower():
        return True
    return False


def _is_term_line(ln: str) -> bool:
    return bool(re.match(r"^T\s*e\s*r\s*m$", ln, re.I))


def _try_split_three_raw(raw: list[str]) -> list[str]:
    """Some PDF rows glue two answers into the first line; split into four rows."""
    if len(raw) != 3:
        return raw
    top = re.sub(r"\s+", " ", raw[0]).strip()
    m = re.search(r"\s+(?=P\s+e)", top, re.I)
    if not m or m.start() < 15:
        m = re.search(r"\s+(?=T\s+o)", top, re.I)
    if not m or m.start() < 15:
        return raw
    a, b = top[: m.start()].strip(), top[m.start() :].strip()
    if len(b) < 8:
        return raw
    return [a, b, re.sub(r"\s+", " ", raw[1]).strip(), re.sub(r"\s+", " ", raw[2]).strip()]


def _fold_interactive_options(raw: list[str]) -> list[str] | None:
    """Fold wrapped PDF lines into exactly four option strings."""
    raw = [re.sub(r"\s+", " ", ln).strip() for ln in raw if ln.strip()]
    if len(raw) == 3:
        raw = _try_split_three_raw(raw)
    if len(raw) < 4:
        return None
    if len(raw) == 4:
        return raw
    if len(raw) == 5:
        a, b, c, d, e = raw
        return [a, b, c, re.sub(r"\s+", " ", (d + " " + e)).strip()]
    out: list[str] = []
    for ln in raw:
        t = re.sub(r"\s+", " ", ln).strip()
        if not t:
            continue
        if out and (
            t[0].islower()
            or (t[0] in "$" and not out[-1].rstrip().endswith(("?", ".", "!")))
        ):
            out[-1] = (out[-1] + " " + t).strip()
        else:
            out.append(t)
    out = [re.sub(r"\s+", " ", x).strip() for x in out]
    if len(out) == 4:
        return out
    if len(out) > 4:
        while len(out) > 4:
            out[-2] = (out[-2] + " " + out[-1]).strip()
            out.pop()
        return out
    return None


def _stem_before_progress(lines: list[str], pi: int) -> str:
    """Stem text immediately above a ``k of M`` progress line (Quizlet test export)."""
    j = pi - 1
    parts: list[str] = []
    while j >= 0:
        ln = lines[j]
        if _is_term_line(ln) or _is_dont_know(ln):
            break
        if _is_progress_line(ln):
            j -= 1
            continue
        if _is_feedback_line(ln) or _is_interactive_junk(ln):
            j -= 1
            continue
        c = _compact_alnum(ln)
        if c in (
            "correct",
            "incorrect",
            "youranswers",
            "yourtime1min",
            "bekindtoyourselfandkeeppractising",
            "0",
            "074",
            "0100",
        ):
            j -= 1
            continue
        parts.append(ln)
        j -= 1
    parts.reverse()
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _stem_after_nearest_term(lines: list[str], pi: int) -> str:
    """
    Recover stems when PDF text order places an orphan ``Term`` directly under
    ``k of M`` so the usual walk-back hits ``Term`` first and returns empty.
    """
    lo = max(0, pi - 120)
    best = ""
    for t in range(lo, pi):
        if not _is_term_line(lines[t]):
            continue
        chunk: list[str] = []
        for j in range(t + 1, pi):
            ln = lines[j]
            if _is_term_line(ln):
                break
            if _is_progress_line(ln):
                continue
            if _is_feedback_line(ln) or _is_interactive_junk(ln):
                continue
            if _is_dont_know(ln):
                break
            chunk.append(ln)
        cand = re.sub(r"\s+", " ", " ".join(chunk)).strip()
        if len(cand) <= len(best):
            continue
        if len(cand) < 25:
            continue
        c0 = _compact_alnum(cand)
        if "?" in cand or "xcep" in c0 or "except" in cand.lower() or len(cand) > 70:
            best = cand
    return best


def _stem_for_interactive(lines: list[str], pi: int) -> str:
    stem = _stem_before_progress(lines, pi)
    if len(stem) >= 12:
        return stem
    fb = _stem_after_nearest_term(lines, pi)
    return fb if len(fb) >= 12 else stem


def _backward_raw_opts_before_dont(lines: list[str], di: int) -> list[str]:
    """Lines belonging to the four choices immediately above a ``Don't know`` row."""
    j = di - 1
    while j >= 0 and _is_feedback_line(lines[j]):
        j -= 1
    raw: list[str] = []
    while j >= 0:
        ln = lines[j]
        if _is_dont_know(ln) or _is_term_line(ln):
            break
        if _is_interactive_junk(ln) or _is_feedback_line(ln):
            j -= 1
            continue
        if _is_progress_line(ln):
            break
        raw.append(ln)
        j -= 1
    raw.reverse()
    return raw


def _skip_feedback_junk_idx(lines: list[str], j: int, n: int) -> int:
    while j < n and (_is_feedback_line(lines[j]) or _is_interactive_junk(lines[j])):
        j += 1
    return j


def _collect_opts_forward_from_progress(lines: list[str], pi: int) -> list[str]:
    """Fallback: scan forward from a progress line until a real ``Don't know``."""
    n = len(lines)
    j = _skip_feedback_junk_idx(lines, pi + 1, n)
    buf: list[str] = []
    empty_dont = 0
    while j < n:
        if _is_dont_know(lines[j]):
            if not buf:
                empty_dont += 1
                if empty_dont > 6:
                    break
                j += 1
                j = _skip_feedback_junk_idx(lines, j, n)
                continue
            break
        empty_dont = 0
        ln = lines[j]
        if _is_term_line(ln):
            j += 1
            while j < n and not _is_progress_line(lines[j]) and not _is_dont_know(lines[j]):
                j += 1
            if j < n and _is_progress_line(lines[j]):
                j += 1
                j = _skip_feedback_junk_idx(lines, j, n)
            continue
        if _is_progress_line(ln):
            j += 1
            j = _skip_feedback_junk_idx(lines, j, n)
            continue
        buf.append(ln)
        j += 1
    return buf


def parse_interactive_quizlet_test(lines: list[str]) -> list[dict[str, Any]]:
    """
    Parse Quizlet **Test** PDFs where each question has ``k of M`` progress, four
    choices, and ``Don't know``. Pair stem ``by_k`` with options taken from the
    region just above the *k*th ``Don't know`` (with a forward-scan fallback).
    """
    didx = [i for i, ln in enumerate(lines) if _is_dont_know(ln)]
    pidx = [i for i, ln in enumerate(lines) if _is_progress_line(ln)]
    if len(didx) < 10 or len(pidx) < 10:
        return []
    prog_meta = [_is_progress_line(lines[i]) for i in pidx]
    M = max(p[1] for p in prog_meta if p)
    if len(didx) != M:
        return []
    by_k: dict[int, int] = {}
    for i, ln in enumerate(lines):
        pr = _is_progress_line(ln)
        if pr and pr[1] == M:
            by_k[pr[0]] = i
    if len(by_k) != M:
        return []

    items: list[dict[str, Any]] = []
    prev_fold: tuple[str, str, str, str] | None = None
    for k in range(1, M + 1):
        stem = _stem_for_interactive(lines, by_k[k])
        raw_b = _backward_raw_opts_before_dont(lines, didx[k - 1])
        fold = _fold_interactive_options(raw_b)
        if not fold or len(stem) < 12:
            raw_f = _collect_opts_forward_from_progress(lines, by_k[k])
            fold2 = _fold_interactive_options(raw_f)
            if fold2 and len(stem) >= 12:
                fold = fold2
        if fold and prev_fold is not None and tuple(fold) == prev_fold:
            if k < M:
                raw_b2 = _backward_raw_opts_before_dont(lines, didx[k])
                fold3 = _fold_interactive_options(raw_b2)
                if fold3:
                    fold = fold3
        if not fold or len(stem) < 12:
            continue
        prev_fold = tuple(fold)
        items.append(
            {
                "stem": stem,
                "A": fold[0],
                "B": fold[1],
                "C": fold[2],
                "D": fold[3],
                "answer": "?",
            }
        )
    return items


def classify_pdf(pdf: Path) -> PdfKind:
    """INS QAA uses ``1.``…``4.`` options; other Quizlet PDFs use test/learn layout."""
    s = pdf.stem.lower().replace(" ", "")
    if "qaa" in s:
        return "ins_qaa"
    return "interactive"


def dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        key = re.sub(r"\s+", " ", (it["stem"] + "|" + it["A"] + "|" + it["B"])).lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def load_answer_merge_file(path: Path) -> dict[int, str]:
    """Load ``N LETTER`` or ``N<TAB>LETTER`` lines into question_index -> A..D."""
    out: dict[int, str] = {}
    raw = path.read_text(encoding="utf-8", errors="replace")
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "\t" in ln:
            left, right = ln.split("\t", 1)
            n_s, letter = left.strip(), right.strip().split()[0] if right.strip() else ""
        else:
            parts = ln.split()
            if len(parts) < 2:
                continue
            n_s, letter = parts[0], parts[1]
        if not n_s.isdigit():
            continue
        L = letter.strip().upper()
        if L not in ("A", "B", "C", "D"):
            continue
        out[int(n_s)] = L
    return out


def apply_answer_overrides(items: list[dict[str, Any]], overrides: dict[int, str]) -> None:
    for i, it in enumerate(items, start=1):
        if i in overrides:
            it["answer"] = overrides[i]


def write_eval_txt(items: list[dict[str, Any]], q_path: Path, a_path: Path) -> None:
    q_path.parent.mkdir(parents=True, exist_ok=True)
    a_path.parent.mkdir(parents=True, exist_ok=True)
    with q_path.open("w", encoding="utf-8") as fq, a_path.open("w", encoding="utf-8") as fa:
        fa.write(
            "# Letter keys A-D below: from PDF when available; otherwise ? until you use "
            "--answers-merge-from with a filled answers_key_TEMPLATE.tsv (or your own N LETTER file).\n"
        )
        for n, it in enumerate(items, start=1):
            stem = re.sub(r"\s+", " ", it["stem"]).strip()
            fq.write(f"{n}. {stem}\n")
            for L, k in (("A", "A"), ("B", "B"), ("C", "C"), ("D", "D")):
                opt = re.sub(r"\s+", " ", it[k]).strip()
                fq.write(f"{L}. {opt}\n")
            fq.write("\n")
            fa.write(f"{n} {it['answer']}\n")


def write_answer_key_sidecar(out_dir: Path, items: list[dict[str, Any]]) -> None:
    """Explain missing keys and emit a TSV template users can fill and merge."""
    n_unknown = sum(
        1 for it in items if str(it.get("answer", "")).strip().upper() not in ("A", "B", "C", "D")
    )
    if not n_unknown:
        return
    notice = out_dir / "answer_key_notice.txt"
    tmpl = out_dir / "answers_key_TEMPLATE.tsv"
    notice.write_text(
        "About answers_from_quizlet_pdf_parse.txt\n"
        "-----------------------------------------\n"
        "Quizlet **Test** PDFs (CA INS QA, INS QA, …) are usually printed after tapping "
        "\"Don't know\" on each item, so the exported text does **not** include which "
        "choice was correct. The converter therefore writes `?` on those lines.\n\n"
        "INS **QAA**-style flashcard PDFs (numbered options 1–4 plus a duplicate answer line) "
        "do include a recoverable letter; those lines show A–D.\n\n"
        "To supply keys for test PDFs:\n"
        "  1) Fill the `answer` column in answers_key_TEMPLATE.tsv (tab-separated).\n"
        "  2) Re-run:\n"
        "       python scripts/quizlet_pdfs_to_eval_txt.py --pdfs ... --out-dir ... \\\n"
        "         --answers-merge-from answers_key_TEMPLATE.tsv\n"
        "Or merge any plain file with one row per question: `N LETTER` (e.g. `12 B`).\n",
        encoding="utf-8",
    )
    with tmpl.open("w", encoding="utf-8") as f:
        f.write("# question_num\tanswer\tstem_preview\n")
        for n, it in enumerate(items, start=1):
            prev = re.sub(r"\s+", " ", it["stem"])[:120].replace("\t", " ")
            f.write(f"{n}\t\t{prev}\n")
    print(
        f"Wrote {notice} and {tmpl} ({n_unknown} / {len(items)} rows still need A–D unless merged)",
        file=sys.stderr,
    )


def openai_reconstruct_chunk(
    client: Any,
    *,
    model: str,
    chunk: str,
    temperature: float,
) -> list[dict[str, Any]]:
    """Ask model to extract 0+ MCQs from noisy Quizlet chunk text."""
    sysm = (
        "You extract multiple-choice questions from messy PDF-to-text dumps. "
        "Return JSON ONLY: {\"items\":[{\"question\":\"...\",\"A\":\"...\",\"B\":\"...\","
        '"C":"...","D":"...","answer":"A|B|C|D"}]} '
        "Rules: only include items with exactly four distinct options and a clear answer letter. "
        "If nothing is recoverable, return {\"items\":[]}. "
        "Do not invent facts; only reconstruct stems/options that appear in the text."
    )
    user = "TEXT DUMP:\n\n" + chunk[:12000]
    req: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": sysm},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_output_tokens": 4096,
    }
    try:
        resp = client.responses.create(**req)
    except BadRequestError as e:
        if "temperature" in str(e).lower() and "not supported" in str(e).lower():
            req.pop("temperature", None)
            resp = client.responses.create(**req)
        else:
            raise
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        parts: list[str] = []
        for item in resp.output:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    parts.append(getattr(c, "text", "") or "")
        text = "".join(parts).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return []
        data = json.loads(m.group(0))
    raw_items = data.get("items") or []
    norm: list[dict[str, Any]] = []
    for it in raw_items:
        try:
            ans = str(it.get("answer", "")).strip().upper()
            if ans not in ("A", "B", "C", "D"):
                continue
            q = str(it.get("question", "")).strip()
            a = str(it.get("A", "")).strip()
            b = str(it.get("B", "")).strip()
            c = str(it.get("C", "")).strip()
            d = str(it.get("D", "")).strip()
            if len({a, b, c, d}) < 4 or not q:
                continue
            norm.append({"stem": q, "A": a, "B": b, "C": c, "D": d, "answer": ans})
        except Exception:
            continue
    return norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdfs",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to PDF files (INS QAA, INS QA, CA INS QA, …)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("eval_set/from_quizlet_pdfs"))
    ap.add_argument(
        "--openai-model",
        default=None,
        help="If set (e.g. gpt-4.1), use OpenAI to reconstruct MCQs from scrambled test PDF chunks",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument(
        "--openai-max-chunks",
        type=int,
        default=0,
        help="Max URL-chunks per scrambled PDF when using --openai-model (0 = no limit)",
    )
    ap.add_argument(
        "--answers-merge-from",
        type=Path,
        default=None,
        help="Optional file of `N LETTER` or `N<TAB>LETTER` rows to replace ? placeholders in the PDF-parse answers file",
    )
    args = ap.parse_args()

    all_items: list[dict[str, Any]] = []
    client = None
    if args.openai_model:
        if OpenAI is None:
            raise SystemExit("openai package not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY required when --openai-model is set")
        client = OpenAI()

    for pdf in args.pdfs:
        if not pdf.is_file():
            print(f"SKIP missing file: {pdf}", file=sys.stderr)
            continue
        lines = pdf_lines(pdf)
        kind = classify_pdf(pdf)
        if kind == "ins_qaa":
            chunks = url_chunks(lines)
            local = parse_ins_qaa_chunks(chunks)
            print(f"{pdf.name}: INS QAA (local) -> {len(local)} items", file=sys.stderr)
            all_items.extend(local)
            continue

        interactive = parse_interactive_quizlet_test(lines)
        if interactive:
            print(f"{pdf.name}: interactive test (local) -> {len(interactive)} items", file=sys.stderr)
            all_items.extend(interactive)
            continue

        if not args.openai_model:
            print(
                f"{pdf.name}: skipped (no local parser match). "
                f"Re-run with --openai-model gpt-4.1 to reconstruct.",
                file=sys.stderr,
            )
            continue

        chunks = url_chunks(lines)
        n_chunk = 0
        before = len(all_items)
        for ch in chunks:
            if len(ch) < 80:
                continue
            if args.openai_max_chunks and n_chunk >= args.openai_max_chunks:
                break
            n_chunk += 1
            got = openai_reconstruct_chunk(
                client,
                model=args.openai_model,
                chunk=ch,
                temperature=args.temperature,
            )
            all_items.extend(got)
            if args.sleep_ms:
                import time as _t

                _t.sleep(args.sleep_ms / 1000.0)
        added = len(all_items) - before
        print(f"{pdf.name}: openai chunks={n_chunk}, items_added={added}", file=sys.stderr)

    merged = dedupe_items(all_items)
    if not merged:
        raise SystemExit("No MCQs extracted. Check PDF paths and optional --openai-model.")

    if args.answers_merge_from:
        if not args.answers_merge_from.is_file():
            raise SystemExit(f"--answers-merge-from not found: {args.answers_merge_from}")
        ov = load_answer_merge_file(args.answers_merge_from)
        apply_answer_overrides(merged, ov)
        print(f"Merged {len(ov)} answer overrides from {args.answers_merge_from}", file=sys.stderr)

    q_path = args.out_dir / "questions.txt"
    a_path = args.out_dir / "answers_from_quizlet_pdf_parse.txt"
    write_eval_txt(merged, q_path, a_path)
    write_answer_key_sidecar(args.out_dir, merged)
    print(f"Wrote {len(merged)} questions -> {q_path}")
    print(f"Wrote {len(merged)} answers -> {a_path}")

    # small manifest for provenance
    man = args.out_dir / "sources.csv"
    with man.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdf_path"])
        for p in args.pdfs:
            w.writerow([str(p.resolve())])
    print(f"Wrote {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
