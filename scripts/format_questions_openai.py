#!/usr/bin/env python3
"""
Format OCR-broken MCQs using OpenAI while preserving numbering/options.

Input format (blocks separated by blank lines):
  N. <stem>
  A. <option>
  B. <option>
  C. <option>
  D. <option>

This script rewrites each block to remove broken words/spaces and normalize punctuation,
but keeps:
  - the same question numbers
  - exactly four options labeled A–D
  - no changes to meaning (best effort)

Usage:
  export OPENAI_API_KEY=...
  .venv/bin/python3 scripts/format_questions_openai.py \
    --in eval_set/from_quizlet_pdfs/questions.txt \
    --out eval_set/from_quizlet_pdfs/questions_formatted_165.txt \
    --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


_QHEAD = re.compile(r"^(\d+)\.\s*(.+)$")
_OPT = re.compile(r"^([A-D])\.\s*(.*)$")


@dataclass
class Block:
    n: int
    stem: str
    A: str
    B: str
    C: str
    D: str


def parse_blocks(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())
    out: list[Block] = []
    for b in blocks:
        lines = [ln.rstrip() for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        m = _QHEAD.match(lines[0].strip())
        if not m:
            continue
        n = int(m.group(1))
        stem = m.group(2).strip()
        opts: dict[str, str] = {}
        for ln in lines[1:]:
            cm = _OPT.match(ln.strip())
            if cm:
                opts[cm.group(1)] = cm.group(2).strip()
        if set(opts) == {"A", "B", "C", "D"}:
            out.append(Block(n=n, stem=stem, A=opts["A"], B=opts["B"], C=opts["C"], D=opts["D"]))
    out.sort(key=lambda x: x.n)
    return out


def format_block_with_openai(client: Any, model: str, block: Block, temperature: float) -> Block:
    sysm = (
        "You are cleaning OCR-broken multiple-choice questions. "
        "Fix spacing inside words (e.g. 'co v er a ge' -> 'coverage'), normalize punctuation/casing, "
        "and keep meaning. Do NOT add or remove content. "
        "Return JSON ONLY with keys: stem, A, B, C, D. "
        "Constraints: exactly 4 options; do not change which option is which."
    )
    user = "\n".join(
        [
            f"stem: {block.stem}",
            f"A: {block.A}",
            f"B: {block.B}",
            f"C: {block.C}",
            f"D: {block.D}",
        ]
    )
    req: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": sysm},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_output_tokens": 1200,
    }
    resp = client.responses.create(**req)
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
            raise ValueError("Model did not return JSON.")
        data = json.loads(m.group(0))
    for k in ("stem", "A", "B", "C", "D"):
        if k not in data:
            raise ValueError(f"Missing key {k} in model JSON.")
        if not isinstance(data[k], str) or not data[k].strip():
            raise ValueError(f"Empty/invalid {k} in model JSON.")
    return Block(n=block.n, stem=data["stem"].strip(), A=data["A"].strip(), B=data["B"].strip(), C=data["C"].strip(), D=data["D"].strip())


def write_blocks(blocks: list[Block], out_path: Path) -> None:
    def squish(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            f.write(f"{b.n}. {squish(b.stem)}\n")
            f.write(f"A. {squish(b.A)}\n")
            f.write(f"B. {squish(b.B)}\n")
            f.write(f"C. {squish(b.C)}\n")
            f.write(f"D. {squish(b.D)}\n")
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep-ms", type=int, default=0)
    ap.add_argument("--max", type=int, default=0, help="Max blocks to format (0 = all)")
    args = ap.parse_args()

    if OpenAI is None:
        raise SystemExit("openai package not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    blocks = parse_blocks(args.inp)
    if not blocks:
        raise SystemExit("No MCQ blocks parsed from input.")
    if args.max:
        blocks = blocks[: args.max]

    client = OpenAI()
    out_blocks: list[Block] = []
    for i, b in enumerate(blocks, start=1):
        try:
            out_blocks.append(format_block_with_openai(client, args.model, b, args.temperature))
        except Exception as e:
            print(f"FAIL block {b.n}: {e}", file=sys.stderr)
            out_blocks.append(b)
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)
        if i % 10 == 0:
            print(f"formatted {i}/{len(blocks)}", file=sys.stderr)

    write_blocks(out_blocks, args.out)
    print(f"Wrote formatted questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

