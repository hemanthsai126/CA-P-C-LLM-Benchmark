#!/usr/bin/env python3
"""
Run multiple-choice questions (from a plain-text file) through a local Ollama model
and write a 3-column CSV (exactly these headers, no other columns):

  question_number, answer, reason

Important:
- This script reads ONLY the provided questions file.
- It does NOT read any answer key files.

Speed: use ``--workers N`` (default 4) to send concurrent requests to Ollama. Ensure the
server can handle the load (e.g. set ``OLLAMA_NUM_PARALLEL`` / run a build with enough GPU
memory); if you see timeouts, lower ``--workers``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Option = Literal["A", "B", "C", "D"]


@dataclass(frozen=True)
class MCQ:
    number: int
    stem: str
    choices: dict[str, str]


_Q_START = re.compile(r"^(\d+)\.\s*(.*)$")
_CHOICE = re.compile(r"^([A-D])\.\s*(.*)$")


def _split_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    cur: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return blocks


def parse_questions(path: Path) -> list[MCQ]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _split_blocks(text)
    out: list[MCQ] = []

    for b in blocks:
        m = _Q_START.match(b[0])
        if not m:
            continue
        n = int(m.group(1))
        stem = m.group(2).strip()
        choices: dict[str, str] = {}
        for line in b[1:]:
            cm = _CHOICE.match(line)
            if not cm:
                if not choices:
                    stem = (stem + " " + line).strip()
                continue
            choices[cm.group(1)] = cm.group(2).strip()

        out.append(MCQ(number=n, stem=stem, choices=choices))

    out.sort(key=lambda q: q.number)
    return out


def choice_display_order(question_number: int) -> tuple[str, ...]:
    """Rotate A–D so the first listed choice is not always A (reduces position bias)."""
    order = ("A", "B", "C", "D")
    r = (question_number - 1) % 4
    return order[r:] + order[:r]


def build_prompt(q: MCQ, *, compact: bool = False) -> str:
    if compact:
        # Shorter prompt = faster decode on small models; keeps strict 2-line contract.
        lines = [
            "California P&C exam style MCQ. Pick the best letter.",
            "Reply with EXACTLY 2 lines and nothing else:",
            "OPTION: X",
            "REASON: 2 short sentences (why that letter). Do not echo the question's A./B./C./D. lines.",
            "",
            f"Question {q.number}: {q.stem}",
        ]
    else:
        lines = [
            "You are a California P&C licensing exam tutor.",
            "Read the stem and all four labeled options (A–D). Pick the single best answer using insurance reasoning.",
            "Do not favor A because it is listed first — each letter is equally plausible until you rule three out.",
            "Do NOT write step-by-step chain-of-thought. After you decide, give a short justification (2–3 sentences).",
            "You MUST output exactly one letter A, B, C, or D. You may NOT refuse or leave the option blank.",
            "Do NOT copy these instructions into your REASON line.",
            "",
            "Output format (EXACTLY 2 lines, nothing before or after):",
            "First line:  OPTION: X     where X is exactly one letter A, B, C, or D (no punctuation after the letter).",
            "Second line: REASON: your 2–3 sentence justification (do not repeat the letters A|B|C|D as a pattern).",
            "",
            f"Question {q.number}: {q.stem}",
        ]
    for k in choice_display_order(q.number):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    return "\n".join(lines)


def _looks_like_echoed_choice_line(ln: str) -> bool:
    """Model sometimes pastes the four options; those lines look like 'B. longer text…'."""
    return bool(re.match(r"^[A-D]\.\s+.{12,}", ln.strip()))


def loose_infer_option(text: str) -> str:
    """
    Best-effort letter when the model skips the OPTION: header.
    Skips lines that look like pasted A–D choice lines to avoid matching stem echoes.
    """
    t = text.replace("\x00", "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    tail = lines[-12:] if len(lines) > 12 else lines

    def scan_line(ln: str) -> str:
        up = ln.upper()
        if _looks_like_echoed_choice_line(ln) and len(ln) > 50:
            return ""
        for pat in (
            r"\bOPTION\s*[:=]\s*([A-D])\b",
            r"\bANSWER\s*[:=]\s*([A-D])\b",
            r"\b(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE|OPTION)\s+IS\s+([A-D])\b",
            r"\b(?:CHOOSE|SELECT|PICK)\s+([A-D])\b",
            r"\b(?:answer|choice)\s+is\s+[:']?\s*\(?([A-D])\)?\b",
        ):
            m = re.search(pat, up)
            if m:
                return m.group(1).upper()
        m = re.match(r"^([A-D])\s*[.):\-–]\s*(?:$|\S)", ln.strip(), re.I)
        if m and len(ln) < 120:
            return m.group(1).upper()
        m = re.match(r"^\(?([A-D])\)?\s*$", ln.strip(), re.I)
        if m:
            return m.group(1).upper()
        return ""

    for ln in reversed(tail):
        hit = scan_line(ln)
        if hit:
            return hit
    for ln in lines:
        if _looks_like_echoed_choice_line(ln) and len(ln) > 50:
            continue
        hit = scan_line(ln)
        if hit:
            return hit
    return ""


def parse_model_response(text: str) -> tuple[str, str]:
    t = text.replace("\x00", "").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    opt = ""
    reason = ""

    def parse_option_tail(tail: str) -> str:
        tail = tail.strip().upper()
        # Single letter after colon (best case).
        m1 = re.match(r"^([A-D])\s*$", tail)
        if m1:
            return m1.group(1)
        # "OPTION: B - because ..." on one line
        m2 = re.match(r"^([A-D])\b", tail)
        if m2:
            return m2.group(1)
        return ""

    # Prefer explicit OPTION/ANSWER lines. Do NOT use generic \b([A-D])\b on the OPTION line:
    # old prompts used "OPTION: A|B|C|D" which always matched A.
    for ln in lines[:12]:
        up = ln.upper()
        if up.startswith("OPTION:") or up.startswith("ANSWER:"):
            tail = ln.split(":", 1)[1] if ":" in ln else ""
            letter = parse_option_tail(tail)
            if letter:
                opt = letter
            # Reason may trail on same line after the letter (e.g. "OPTION: B because …")
            if ":" in ln:
                tail_rest = ln.split(":", 1)[1]
                because = re.split(r"\b(?:BECAUSE|SINCE|AS)\b", tail_rest, maxsplit=1, flags=re.I)
                if len(because) > 1 and because[1].strip():
                    extra = because[1].strip()
                    if len(extra) > 15:
                        reason = extra

    # REASON may continue on following lines until a new OPTION/ANSWER or blank break.
    reason_parts: list[str] = []
    in_reason = False
    for ln in lines:
        up = ln.upper()
        if up.startswith("REASON:"):
            in_reason = True
            reason_parts.append(ln.split(":", 1)[-1].strip())
            continue
        if in_reason:
            if up.startswith("OPTION:") or up.startswith("ANSWER:"):
                break
            if ln:
                reason_parts.append(ln)
    reason_from_block = " ".join(reason_parts).strip()
    if reason_from_block:
        reason = reason_from_block
    elif not reason:
        reason = ""

    if opt not in ("A", "B", "C", "D"):
        loose = loose_infer_option(t)
        if loose:
            opt = loose

    if not reason:
        reason = t or ""
    return opt, reason


_JUNK_MARKERS = (
    "FORMAT FIX",
    "YOU MUST",
    "LINE 1:",
    "LINE 2:",
    "OUTPUT EXACTLY",
    "CRITICAL:",
    "[FORCED_OPTION]",
    "OPTION: X",
)


def _reason_is_prompt_echo(s: str) -> bool:
    u = s.upper()
    return any(m in u for m in _JUNK_MARKERS) or ("OPTION:" in u and u.count("OPTION:") > 1)


def _extract_reason_after_keyword(text: str) -> str:
    """Pull text after a REASON: line from raw model output."""
    parts: list[str] = []
    in_r = False
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln:
            if in_r:
                break
            continue
        up = ln.upper()
        if up.startswith("REASON:"):
            in_r = True
            parts.append(ln.split(":", 1)[-1].strip())
            continue
        if in_r:
            if up.startswith("OPTION:") or up.startswith("ANSWER:"):
                break
            parts.append(ln)
    return " ".join(parts).strip()


def narrative_reason_from_raws(raw_responses: list[str]) -> str:
    """Pull the longest prose-like span from model output, skipping echoed MCQ option lines."""
    best = ""
    for raw in raw_responses:
        lines: list[str] = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s:
                continue
            if _looks_like_echoed_choice_line(s) and len(s) > 45:
                continue
            up = s.upper()
            if up.startswith(("LINE 1:", "LINE 2:", "OUTPUT EXACTLY", "FORMAT FIX.")):
                continue
            if up.startswith("OPTION:") or up.startswith("ANSWER:"):
                tail = s.split(":", 1)[-1].strip()
                tail = re.sub(r"^[A-D]\b\s*", "", tail, flags=re.I).strip()
                if len(tail) > 25:
                    lines.append(tail)
                continue
            if up.startswith("REASON:"):
                lines.append(s.split(":", 1)[-1].strip())
                continue
            if re.match(r"^Question\s+\d+\s*:", s, re.I):
                continue
            lines.append(s)
        blob = re.sub(r"\s+", " ", " ".join(lines)).strip()
        if len(blob) > len(best) and len(blob) >= 25 and not _reason_is_prompt_echo(blob):
            best = blob
    return best[:500]


def sanitize_reason_for_export(reason: str) -> str:
    """
    Strip echoed instructions / pasted stems from the reason cell so exports match
    a clean ``question_number, answer, reason`` CSV (prose only).
    """
    if not reason:
        return ""
    orig = reason
    r = re.sub(r"\s+", " ", reason.replace("\x00", "").strip())
    low = r.lower()
    for needle in (
        "output format (exactly",
        "first line: option:",
        "second line: reason:",
        "format fix.",
        "do not copy these instructions",
        "nothing before or after",
        "line 1: option:",
        "line 2: reason:",
    ):
        j = low.find(needle)
        if j >= 20:
            r = r[:j].strip()
            low = r.lower()
    r = re.sub(r"^\s*Question\s+\d+\s*:\s*", "", r, count=1, flags=re.I).strip()
    if len(r) > 180 and not re.search(
        r"\b(insurance|coverage|policy|claim|liability|insured|premium|underwriting)\b", r, re.I
    ):
        parts = re.split(r"(?i)\bREASON:\s*", orig)
        if len(parts) > 1:
            tail = re.sub(r"\s+", " ", parts[-1].strip())
            for needle in (
                "output format (exactly",
                "first line:",
                "format fix.",
            ):
                j = tail.lower().find(needle.lower())
                if j >= 15:
                    tail = tail[:j].strip()
            if len(tail) >= 15:
                r = tail[:500]
    return r[:500].strip()


def clean_reason_for_csv(
    reason: str,
    *,
    forced_option: bool,
    raw_responses: list[str],
) -> str:
    """Single short justification for export — no format-fix or prompt echo."""
    r = re.sub(r"\s+", " ", (reason or "").strip())
    if r and not _reason_is_prompt_echo(r) and r.lower() != "no concise justification could be extracted.":
        return r[:500]

    for raw in reversed(raw_responses):
        cand = _extract_reason_after_keyword(raw)
        cand = re.sub(r"\s+", " ", cand).strip()
        if cand and not _reason_is_prompt_echo(cand):
            return cand[:500]

    # First sentence-like chunk without option labels from first response
    blob = re.sub(r"\s+", " ", (raw_responses[0] if raw_responses else "").replace("\x00", ""))
    blob = re.sub(r"\b[ABCD]\.\s*", " ", blob)
    for chunk in re.split(r"(?<=[.!?])\s+", blob):
        c = chunk.strip()
        if 25 <= len(c) <= 500 and not _reason_is_prompt_echo(c):
            return c

    narr = narrative_reason_from_raws(raw_responses)
    if narr:
        return narr

    if forced_option:
        return (
            "The model did not follow OPTION/REASON format; letter was inferred. "
            "No usable justification text was extracted from its output."
        )
    return "No concise justification could be extracted."


def build_format_fix_prompt(q: MCQ, prior_response: str) -> str:
    """
    Second attempt prompt: force strict output (2 lines) and explicitly disallow blank answers.
    """
    prior = prior_response.replace("\x00", "").strip()
    lines = [
        "FORMAT FIX. You MUST follow the output format exactly.",
        "You MUST choose one letter A/B/C/D based on the insurance question — not by defaulting to A.",
        "Output EXACTLY 2 lines and nothing else:",
        "Line 1: OPTION: X   (X is only one letter: A, B, C, or D)",
        "Line 2: REASON: 2-3 sentences explaining your choice",
        "",
        f"Question {q.number}: {q.stem}",
    ]
    for k in choice_display_order(q.number):
        if q.choices.get(k):
            lines.append(f"{k}. {q.choices[k]}")
    if prior:
        lines += ["", "Your previous response (for context):", prior[:1500]]
    return "\n".join(lines)


def force_pick_option(*, stem: str, text: str) -> str:
    """
    Last-resort fallback if the model still won't emit OPTION:.
    Prefer any plausible letter from the model text; otherwise a deterministic hash of the stem
    (avoids always defaulting to A).
    """
    t = (text or "").replace("\x00", "").upper()
    for pat in (
        r"\bCORRECT\s+(?:ANSWER|OPTION)\s+IS\s+([A-D])\b",
        r"\b(?:ANSWER|CHOICE)\s+IS\s+([A-D])\b",
        r"\bOPTION\s+([A-D])\b",
    ):
        m = re.search(pat, t)
        if m:
            return m.group(1)
    letters = re.findall(r"\b([A-D])\b", t)
    if letters:
        return letters[-1]
    h = int(hashlib.sha256(stem.encode("utf-8", errors="replace")).hexdigest(), 16)
    return "ABCD"[h % 4]


def answer_one_mcq(q: MCQ, *, ollama: str, model: str) -> tuple[int, str, str, float]:
    """
    Return (question_number, option, reason, seconds).
    For malformed stems, option is empty and reason explains skip.
    """
    present = [k for k in ("A", "B", "C", "D") if q.choices.get(k)]
    if len(present) < 2:
        return q.number, "", f"SKIPPED: malformed question (found choices: {present})", 0.0

    raw_responses: list[str] = []
    forced_option = False
    compact = "tinyllama" in model.lower()

    prompt = build_prompt(q, compact=compact)
    t0 = time.time()
    resp = ollama_generate_with_retry(base_url=ollama, model=model, prompt=prompt, temperature=0.0)
    dt = time.time() - t0
    raw_responses.append(resp)

    opt, reason = parse_model_response(resp)
    if opt not in ("A", "B", "C", "D"):
        fix_prompt = build_format_fix_prompt(q, resp)
        t1 = time.time()
        resp2 = ollama_generate_with_retry(base_url=ollama, model=model, prompt=fix_prompt, temperature=0.0)
        dt += time.time() - t1
        raw_responses.append(resp2)
        opt, reason = parse_model_response(resp2)
    if opt not in ("A", "B", "C", "D"):
        merged = "\n".join(raw_responses)
        lo = loose_infer_option(merged)
        if lo:
            opt = lo
    if opt not in ("A", "B", "C", "D"):
        opt = force_pick_option(stem=q.stem, text=merged if raw_responses else "")
        forced_option = True

    reason_out = clean_reason_for_csv(
        reason,
        forced_option=forced_option,
        raw_responses=raw_responses,
    )
    reason_out = sanitize_reason_for_export(reason_out)
    if len(reason_out) > 500:
        reason_out = reason_out[:497] + "..."
    return q.number, opt, reason_out, dt


def _ollama_generate_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    num_ctx: int,
) -> dict:
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": int(num_predict), "num_ctx": int(num_ctx)},
    }
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def ollama_generate_with_retry(*, base_url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Normal models return text in `response`.
    DeepSeek-R1 may return empty `response` early; it may put interim text in `thinking`.
    """
    if "deepseek-r1" in model:
        first_num_predict, retry_num_predict, num_ctx = 48, 128, 256
    elif model.startswith("qwen3:"):
        # qwen3:14b can be extremely slow; keep responses tightly bounded.
        first_num_predict, retry_num_predict, num_ctx = 64, 96, 128
    elif "tinyllama" in model.lower():
        # Enough headroom for OPTION + REASON in one shot (truncation → slow retry + bad CSV).
        first_num_predict, retry_num_predict, num_ctx = 256, 384, 1536
    elif "gemma" in model.lower():
        first_num_predict, retry_num_predict, num_ctx = 192, 320, 2048
    else:
        first_num_predict, retry_num_predict, num_ctx = 96, 192, 512

    try:
        data1 = _ollama_generate_once(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=first_num_predict,
            num_ctx=num_ctx,
        )
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"Ollama HTTP error {e.code}: {detail}".strip()) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach Ollama at {base_url}: {e.reason}") from e

    resp1 = str(data1.get("response", "")).replace("\x00", "").strip()
    think1 = str(data1.get("thinking", "")).replace("\x00", "").strip()
    # For qwen3 models on Ollama, the output may appear only in `thinking`.
    if resp1:
        return resp1
    if think1:
        return think1

    done_reason = str(data1.get("done_reason", "")).lower()
    if done_reason in ("length", "max_tokens", "max_length") or data1.get("thinking"):
        data2 = _ollama_generate_once(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=retry_num_predict,
            num_ctx=num_ctx,
        )
        resp2 = str(data2.get("response", "")).replace("\x00", "").strip()
        if resp2:
            return resp2
        think2 = str(data2.get("thinking", "")).replace("\x00", "").strip()
        if think2:
            return think2
        return resp2

    return resp1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, type=Path, help="Path to questions.txt (MCQs only)")
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. mistral:7b")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of questions (0 = all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent Ollama requests (1 = sequential). Default 4. Reduce if Ollama errors or OOM.",
    )
    args = ap.parse_args()

    if args.questions.name.lower().endswith("answers.txt"):
        raise SystemExit("Refusing to run: --questions points at answers.txt")

    qs = parse_questions(args.questions)
    if not qs:
        print(f"No questions parsed from {args.questions}", file=sys.stderr)
        return 2

    if args.limit and args.limit > 0:
        qs = qs[: args.limit]

    print(
        f"Running {len(qs)} question(s) from {args.questions} → {args.out} (workers={args.workers})",
        file=sys.stderr,
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, len(qs)))

    def run_q(q: MCQ) -> tuple[int, str, str, float]:
        return answer_one_mcq(q, ollama=args.ollama, model=args.model)

    by_num: dict[int, tuple[str, str, float]] = {}
    total = len(qs)
    prog_lock = threading.Lock()
    done_n = 0

    def _report(n: int, opt: str, dt: float) -> None:
        nonlocal done_n
        with prog_lock:
            done_n += 1
            cur = done_n
        print(f"[{cur}/{total}] Q{n} -> {opt or '?'} ({dt:.2f}s)", file=sys.stderr, flush=True)

    if workers == 1:
        for q in qs:
            n, opt, reason, dt = run_q(q)
            by_num[n] = (opt, reason, dt)
            _report(n, opt, dt)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(run_q, q): q for q in qs}
            for fut in as_completed(fut_map):
                n, opt, reason, dt = fut.result()
                by_num[n] = (opt, reason, dt)
                _report(n, opt, dt)

    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_number", "answer", "reason"])
        for q in sorted(qs, key=lambda x: x.number):
            opt, reason, _dt = by_num.get(q.number, ("", "No result.", 0.0))
            w.writerow([q.number, opt, reason])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
