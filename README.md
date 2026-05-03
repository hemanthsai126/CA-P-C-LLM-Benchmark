# California P&C broker license — LLM benchmark

## Motto / objective

Demonstrate that **modern AI can write, reason about, and pass** California **property & casualty broker / agent** licensing-style material: realistic multiple-choice items, consistent explanations, and scores against held-out answer keys—without leaking the key into the prompt.

This is a **research benchmark repo**, not legal advice or a substitute for state licensing.

## What lives where (high level)

| Path | Role |
|------|------|
| [`DATA/`](DATA/README_data_imports.md) | **Data hub:** `eval_set/` (both benchmarks), `quizlet/` dumps, `processed/` CSVs, `external/`, `raw/`. On macOS with a case-insensitive volume, `DATA/` and `data/` are the **same folder**—use either spelling. |
| [`eval_set/`](eval_set/README.md) | **Symlink** → `DATA/eval_set/`. **All benchmark questions and keys**, split by provenance (same content as `DATA/eval_set/`). |
| [`eval_set/from_quizlet_pdfs/`](eval_set/from_quizlet_pdfs/README.md) | **Quizlet PDF pipeline** → raw `questions.txt`, frozen **~165** OpenAI-formatted snapshot, and **800** balanced items (`questions_formatted.txt` + `answers.txt` + bucket CSV/PNG). |
| [`eval_set/from_youtube_video/`](eval_set/from_youtube_video/README.md) | **YouTube / transcript track** (**150** MCQs): `questions.txt`, `answers.txt`, `explanations.txt`, GPT-4.1 bucket CSV/PNG, and [`Results.md`](eval_set/from_youtube_video/Results.md). |
| [`results/`](results/README.md) | Model-run CSVs **and** [`results/from_quizlet_pdfs/`](results/from_quizlet_pdfs/) / [`results/from_youtube_video/`](results/from_youtube_video/) **snapshot copies** of the 165-Q + 150-Q sets (same inode as `RESULTS/` on case-insensitive disks). |
| [`scripts/`](scripts/README.md) | PDF → text, OpenAI formatters, synthetic generation, bucket balance, model runners, judges, plots. |
| [`judge_runs_openai/`](judge_runs_openai/README.md) | Judge outputs (CSV + JSONL). |
| [`source_material/`](source_material/README.md) | Handbooks, code collections, transcripts. |
| [`Results_plots/`](Results_plots/README.md) | Exported figures for reports. |

Every major folder above includes a **README** that names important files and explains how they were produced.

## Quick runs

**Questions-only (no answer key in prompt)** — example with OpenAI (CSV columns: `question_number`, `answer`, `reason`):

```bash
.venv/bin/python3 scripts/run_questions_reasoned_openai.py \
  --questions eval_set/from_youtube_video/questions.txt \
  --model gpt-4.1-mini \
  --out results/openai_gpt-4.1-mini_reasoned.csv
```

For the **800-Q Quizlet** track in `eval_set/`, point `--questions` at `eval_set/from_quizlet_pdfs/questions_formatted.txt` and score against `eval_set/from_quizlet_pdfs/answers.txt`. The **`results/synthetic_data/`** snapshot is a separate **400-Q** synthetic set (see [`results/README.md`](results/README.md)).

**Judge (uses explanations track by default):**

```bash
.venv/bin/python3 scripts/judge_reasoning_openai.py \
  --questions eval_set/from_youtube_video/questions.txt \
  --answers eval_set/from_youtube_video/answers.txt \
  --explanations eval_set/from_youtube_video/explanations.txt \
  --run-dir judge_runs_openai/gpt-4.1-mini
```

## Historical accuracy note (YouTube track)

Older tables in [`eval_set/from_youtube_video/Results.md`](eval_set/from_youtube_video/Results.md) still describe an earlier **~113**-ID subset before dedupe and later merges. The current **`eval_set/from_youtube_video/answers.txt`** aligns with **150** question IDs. The **800-Q** Quizlet track is separate; size and difficulty differ.

## Hygiene

- Do **not** pass answer files into model prompts, retrieval corpora, or training data for the same benchmark you score.
