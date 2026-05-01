"""Prompt builders. Only use public fields from each question dict."""

from __future__ import annotations


def build_user_message(question: dict) -> str:
    lines = [question["question"].strip(), ""]
    labels = "ABCD"
    for i, text in enumerate(question["choices"]):
        lines.append(f"{labels[i]}. {text}")
    lines.append("")
    lines.append(
        "Respond with exactly one letter: A, B, C, or D (the single best answer). "
        "No explanation."
    )
    return "\n".join(lines)


def system_prompt_for_method(method: str) -> str:
    method = (method or "zero_shot").strip().lower()
    if method == "zero_shot":
        return (
            "You are answering California property and casualty broker-agent style "
            "multiple-choice questions. Choose the single best answer."
        )
    if method == "cot":
        return (
            "You are answering California property and casualty broker-agent style "
            "multiple-choice questions. Think step by step privately, then on the "
            "last line output exactly one letter: A, B, C, or D only."
        )
    raise ValueError(f"Unknown method: {method}")
