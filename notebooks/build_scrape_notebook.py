"""One-off generator for scrape_mcq_sources.ipynb — run: python notebooks/build_scrape_notebook.py"""
import json
from pathlib import Path


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": s.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(
    md(
        """# MCQ extraction: local text → `questions.txt` + `answers.txt`

This notebook helps you **normalize** broker-style multiple-choice items into:

- **`questions.txt`**: numbered stems and four choices (`A`–`D`).
- **`answers.txt`**: one line per item, format `N L` (e.g. `1 A`, `12 B`).

It also includes a **web template** you can adapt **only for sources you are legally allowed** to copy or scrape (your own CMS, licensed dumps, public-domain material, or explicit permission). Many commercial prep sites forbid scraping in their terms of service and may assert copyright over their item text—**do not** use generic scrapers against those sites without clearance.

**Recommended workflow:** export or paste permitted content into a `.txt` file, then run the **Local file parsing** section below."""
    )
)

cells.append(md("""## 1. Configuration\n\nSet paths for input dumps, and where to write paired outputs."""))

cells.append(
    code(
        """from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

# Repo root (parent of notebooks/)
NOTEBOOK_DIR = Path.cwd()
if NOTEBOOK_DIR.name == "notebooks":
    REPO_ROOT = NOTEBOOK_DIR.parent
else:
    REPO_ROOT = NOTEBOOK_DIR

OUT_DIR = REPO_ROOT / "notebooks" / "extracted"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS_OUT = OUT_DIR / "questions.txt"
ANSWERS_OUT = OUT_DIR / "answers.txt"
JSONL_COMBINED = OUT_DIR / "mcq_combined_with_answers.jsonl"  # internal; split before eval
JSONL_PUBLIC = REPO_ROOT / "data" / "mcq_import_public.jsonl"
JSONL_LABELS = REPO_ROOT / "data" / "mcq_import_labels.json"
"""
    )
)

cells.append(
    md(
        """## 2. Parsers

### 2a. Numbered question file (`1.` / `1)` stem, then `A.` … `D.`)

Continuation lines attach to the stem until the first `A`/`B`/`C`/`D` choice line appears; continuation after choices attaches to the last choice."""
    )
)

cells.append(
    code(
        """_Q_START = re.compile(r"^\\s*(\\d+)[.)]\\s+(.*)$")
_CHOICE = re.compile(r"^\\s*([A-Da-d])[.)]\\s+(.*)$")


def parse_numbered_mcq_file(text: str) -> list[dict[str, Any]]:
    "Return dicts: number (int), question (str), choices (list of four str in A..D order)."
    lines = text.splitlines()
    items: list[dict[str, Any]] = []
    n: int | None = None
    q_parts: list[str] = []
    choices: dict[str, str] = {}
    last_choice_key: str | None = None

    def flush() -> None:
        nonlocal n, q_parts, choices, last_choice_key
        if n is None:
            return
        ordered = [choices.get(L, "").strip() for L in "ABCD"]
        if any(ordered):
            items.append({"number": n, "question": " ".join(q_parts).strip(), "choices": ordered})
        n, q_parts, choices, last_choice_key = None, [], {}, None

    for raw in lines:
        line = raw.rstrip()
        if not line:
            continue
        mq = _Q_START.match(line)
        mc = _CHOICE.match(line)
        if mq:
            flush()
            n = int(mq.group(1))
            q_parts = [mq.group(2).strip()]
            last_choice_key = None
        elif mc and n is not None:
            k = mc.group(1).upper()
            last_choice_key = k
            choices[k] = mc.group(2).strip()
        elif n is not None and last_choice_key is None:
            q_parts.append(line.strip())
        elif n is not None and last_choice_key is not None:
            choices[last_choice_key] = (choices[last_choice_key] + " " + line.strip()).strip()
    flush()
    items.sort(key=lambda x: x["number"])
    return items


_ANS_LINE = re.compile(r"^\\s*(\\d+)\\s*[.)]?\\s*([A-Da-d])\\s*(?:#.*)?$")


def parse_answers_file(text: str) -> dict[int, str]:
    "Map question number -> A..D. Lines like: 1 A  |  12. b  |  3 C # comment"
    out: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _ANS_LINE.match(line)
        if not m:
            continue
        out[int(m.group(1))] = m.group(2).upper()
    return out


def letter_to_index(letter: str) -> int:
    return "ABCD".index(letter.upper())


def export_questions_txt(items: list[dict[str, Any]], path: Path) -> None:
    blocks: list[str] = []
    for it in items:
        n = it["number"]
        stem = it["question"].strip()
        lines = [f"{n}. {stem}"]
        labels = "ABCD"
        for i, ctext in enumerate(it["choices"]):
            lines.append(f"{labels[i]}. {ctext.strip()}")
        blocks.append("\\n".join(lines))
    path.write_text("\\n\\n".join(blocks) + "\\n", encoding="utf-8")


def export_answers_txt(mapping: dict[int, str], path: Path) -> None:
    lines = [f"{n} {letter.upper()}" for n, letter in sorted(mapping.items())]
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def attach_answers(items: list[dict[str, Any]], answers: dict[int, str]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for it in items:
        n = it["number"]
        if n not in answers:
            raise KeyError(f"Missing answer for question number {n}")
        letter = answers[n]
        idx = letter_to_index(letter)
        enriched.append({**it, "answer_letter": letter, "correct_index": idx})
    return enriched


def write_eval_split(enriched: list[dict[str, Any]], *, source_tag: str) -> None:
    "Writes repo-style public jsonl + labels json (eval runner uses separate label load)."
    labels_payload: dict[str, dict[str, int]] = {}
    public_lines: list[str] = []
    for it in enriched:
        qid = str(uuid.uuid4())
        public = {
            "id": qid,
            "source": source_tag,
            "topic": "",
            "question": it["question"],
            "choices": it["choices"],
        }
        public_lines.append(json.dumps(public, ensure_ascii=False))
        labels_payload[qid] = {"correct_index": int(it["correct_index"])}
    JSONL_PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    JSONL_PUBLIC.write_text("\\n".join(public_lines) + "\\n", encoding="utf-8")
    JSONL_LABELS.write_text(
        json.dumps(
            {"schema": "eval_only_do_not_load_into_prompts", "labels": labels_payload},
            indent=2,
            ensure_ascii=False,
        )
        + "\\n",
        encoding="utf-8",
    )
"""
    )
)

cells.append(
    md(
        """## 3. Your text exports (questions + answers files)

There are **no** bundled sample questions in this repo. Put permitted dumps under e.g. `data/raw/`, set paths in the next cell, then run it."""
    )
)

cells.append(
    code(
        """MY_QUESTIONS = REPO_ROOT / "data" / "raw" / "questions.txt"  # your numbered Q file
MY_ANSWERS = REPO_ROOT / "data" / "raw" / "answers.txt"  # lines: 1 A, 2 B, ...

pq = parse_numbered_mcq_file(MY_QUESTIONS.read_text(encoding="utf-8"))
pa = parse_answers_file(MY_ANSWERS.read_text(encoding="utf-8"))
missing = [it["number"] for it in pq if it["number"] not in pa]
extra = sorted(set(pa) - {it["number"] for it in pq})
assert not missing, f"answers missing for: {missing[:20]}"
if extra:
    print("warning: answer lines with no matching question:", extra[:20])

export_questions_txt(pq, QUESTIONS_OUT)
export_answers_txt(pa, ANSWERS_OUT)
write_eval_split(attach_answers(pq, pa), source_tag="manual_import")
print("done ->", QUESTIONS_OUT, ANSWERS_OUT, JSONL_PUBLIC)
"""
    )
)

cells.append(
    md(
        """## 4. Web scraping template (you supply legal URLs + selectors)

**Before fetching:** confirm robots.txt, terms of use, and copyright. Prefer official PDFs you download manually, or APIs that grant a license.

The cell below defines:

- `fetch_html(url)` — polite delay + browser-like User-Agent.
- `parse_mcqs_from_soup` — implement per permitted site.

A **toy HTML** string demonstrates parsing a simple `<div class="mcq">` pattern—replace with your permitted markup."""
    )
)

cells.append(
    code(
        """DEFAULT_UA = (
    "Mozilla/5.0 (compatible; BrokerEvalBot/0.1; +https://example.invalid; "
    "contact: you@yourdomain.invalid)"
)


def fetch_html(url: str, *, delay_s: float = 1.0, timeout: int = 30) -> str:
    time.sleep(delay_s)
    resp = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_demo_mcqs_from_html(html: str) -> tuple[list[dict[str, Any]], dict[int, str]]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    answers: dict[int, str] = {}
    for div in soup.select("div.mcq"):
        n = int(div.get("data-n", "0"))
        stem_el = div.select_one("p.q") or div.select_one(".question")
        if not stem_el:
            continue
        stem = stem_el.get_text(" ", strip=True)
        lis = div.select("ol.choices li")
        if len(lis) != 4:
            continue
        choices = [li.get_text(" ", strip=True) for li in lis]
        key_el = div.select_one(".key") or div.select_one(".answer")
        if not key_el:
            continue
        letter = key_el.get_text(strip=True).upper()[:1]
        items.append({"number": n, "question": stem, "choices": choices})
        answers[n] = letter
    items.sort(key=lambda x: x["number"])
    return items, answers


# Example: items, ans = parse_demo_mcqs_from_html(your_html_string)
# then export_questions_txt / export_answers_txt / write_eval_split as above.
"""
    )
)

cells.append(
    md(
        """### 4b. URLs: official references vs MCQ targets

- **Full table with descriptions:** `notebooks/urls_official_licensing.md`
- **`OFFICIAL_REFERENCE_URLS`** — CDI + PSI + Sircon links below (exam info, scheduling, bulletin PDF, apply). **Not** MCQ banks.
- **`PERMITTED_URLS`** — Only pages **you** may automate for **MCQ-shaped HTML**. Implement `site_extract_placeholder` for those hosts."""
    )
)

cells.append(
    code(
        """OFFICIAL_REFERENCE_URLS: list[str] = [
    # CDI — exam & licensing (see notebooks/urls_official_licensing.md for descriptions)
    "https://www.insurance.ca.gov/0200-industry/",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/ExamTimesandQuestion.cfm",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/Remote-Testing-Frequently-Asked-Question.cfm",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/0100-indiv-resident/index.cfm",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/referral-from-psi.cfm",
    "https://www.insurance.ca.gov/0200-industry/0030-seek-pre-lic/",
    "https://www.insurance.ca.gov/0200-industry/0090-faq/",
    # PSI — scheduling & candidate bulletin (PDF)
    "https://test-takers.psiexams.com/cadi",
    "https://www.psiexams.com/test-takers/cadi/",
    "http://candidate.psiexams.com/bulletin/display_bulletin.jsp?ro=yes&actionname=83&bulletinid=506&bulletinurl=.pdf",
    # Application portal (linked from CDI)
    "https://www.sircon.com/products/apply.jsp",
]

# MCQ scrape targets: keep empty until you have permitted pages + a working parser.
PERMITTED_URLS: list[str] = []

SAVE_REFERENCE_HTML = False  # set True to download OFFICIAL_REFERENCE_URLS into notebooks/extracted/fetched_cache_official/


def site_extract_placeholder(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], dict[int, str]]:
    raise NotImplementedError("Implement selectors for your permitted MCQ source.")


def scrape_urls(urls: list[str]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    "Renumbers sequentially 1..N across pages (each page may reuse 1..M locally)."
    all_items: list[dict[str, Any]] = []
    all_answers: dict[int, str] = {}
    nxt = 1
    for url in urls:
        html = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        items, ans = site_extract_placeholder(soup)
        for it in sorted(items, key=lambda x: x["number"]):
            all_items.append({"number": nxt, "question": it["question"], "choices": it["choices"]})
            all_answers[nxt] = ans[it["number"]]
            nxt += 1
    return all_items, all_answers


def snapshot_urls(urls: list[str], *, subdir: str = "fetched_cache_official") -> None:
    "Save raw HTML for inspection while you write parsers (polite delay per request)."
    cache = OUT_DIR / subdir
    cache.mkdir(parents=True, exist_ok=True)
    for url in urls:
        html = fetch_html(url, delay_s=1.5)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", url.replace("https://", "").replace("http://", ""))[:180]
        (cache / f"{safe}.html").write_text(html, encoding="utf-8", errors="replace")
        print("saved", cache / f"{safe}.html")


print(f"OFFICIAL_REFERENCE_URLS ({len(OFFICIAL_REFERENCE_URLS)} pages):")
for u in OFFICIAL_REFERENCE_URLS:
    print(" ", u)

if SAVE_REFERENCE_HTML:
    snapshot_urls(OFFICIAL_REFERENCE_URLS)

if PERMITTED_URLS:
    items, ans = scrape_urls(PERMITTED_URLS)
    export_questions_txt(items, OUT_DIR / "scraped_questions.txt")
    export_answers_txt(ans, OUT_DIR / "scraped_answers.txt")
else:
    print("PERMITTED_URLS (MCQ scrape) empty — add permitted quiz pages and implement site_extract_placeholder.")
"""
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parent / "scrape_mcq_sources.ipynb"
out_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out_path)
