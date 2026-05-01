from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from eval.prompts import build_user_message, system_prompt_for_method


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_labels(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, int] = {}
    for qid, payload in data["labels"].items():
        out[qid] = int(payload["correct_index"])
    return out


_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def parse_choice_letter(text: str) -> int | None:
    if not text:
        return None
    text = text.strip()
    m = _LETTER_RE.search(text)
    if not m:
        return None
    return "ABCD".index(m.group(1).upper())


async def run_ollama_chat(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        body = r.json()
    return (body.get("message") or {}).get("content") or ""


async def evaluate_split(
    *,
    questions: list[dict[str, Any]],
    labels: dict[str, int],
    base_url: str,
    model: str,
    method: str,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Returns correct, total, per_question records."""
    system = system_prompt_for_method(method)
    correct = 0
    total = 0
    details: list[dict[str, Any]] = []
    for q in questions:
        qid = q["id"]
        if qid not in labels:
            continue
        gold = labels[qid]
        user = build_user_message(q)
        raw = await run_ollama_chat(
            base_url=base_url,
            model=model,
            system=system,
            user=user,
            temperature=0.0,
        )
        pred = parse_choice_letter(raw)
        ok = pred == gold
        if ok:
            correct += 1
        total += 1
        details.append(
            {
                "id": qid,
                "gold_index": gold,
                "pred_index": pred,
                "raw_tail": (raw or "")[-500:],
                "correct": ok,
            }
        )
    return correct, total, details
