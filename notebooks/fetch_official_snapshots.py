#!/usr/bin/env python3
"""Download official CDI / PSI / Sircon pages listed for the project (same URLs as the notebook).

Usage (from repo root):  python notebooks/fetch_official_snapshots.py
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

URLS: list[str] = [
    "https://www.insurance.ca.gov/0200-industry/",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/ExamTimesandQuestion.cfm",
    "https://www.insurance.ca.gov/0200-industry/0010-producer-online-services/0200-exam-info/Remote-Testing-Frequently-Asked-Question.cfm",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/0100-indiv-resident/index.cfm",
    "https://www.insurance.ca.gov/0200-industry/0020-apply-license/referral-from-psi.cfm",
    "https://www.insurance.ca.gov/0200-industry/0030-seek-pre-lic/",
    "https://www.insurance.ca.gov/0200-industry/0090-faq/",
    "https://test-takers.psiexams.com/cadi",
    "https://www.psiexams.com/test-takers/cadi/",
    "http://candidate.psiexams.com/bulletin/display_bulletin.jsp?ro=yes&actionname=83&bulletinid=506&bulletinurl=.pdf",
    "https://www.sircon.com/products/apply.jsp",
]

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def safe_name(url: str) -> str:
    s = re.sub(r"^https?://", "", url)
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s[:200]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "notebooks" / "extracted" / "fetched_cache_official"
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    for url in URLS:
        time.sleep(1.2)
        try:
            r = session.get(url, timeout=60, allow_redirects=True)
            r.raise_for_status()
        except requests.RequestException as e:
            print("FAIL", url, e)
            continue
        ct = (r.headers.get("content-type") or "").lower()
        raw = r.content
        is_pdf = raw[:5] == b"%PDF-" or ("application/pdf" in ct and b"<html" not in raw[:2000].lower())
        ext = ".pdf" if is_pdf else ".html"
        base = safe_name(url).removesuffix(".pdf")
        path = out / f"{base}{ext}"
        if ext == ".html":
            path.write_text(raw.decode("utf-8", errors="replace"), encoding="utf-8")
        else:
            path.write_bytes(raw)
        print("OK", path.name, len(r.content), "bytes")
    print("Done ->", out)


if __name__ == "__main__":
    main()
