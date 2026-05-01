# California P&C broker license — LLM benchmark

## Motto / objective

Demonstrate that **modern AI can write, reason about, and pass** California **property & casualty broker / agent** licensing-style material: realistic multiple-choice items, consistent explanations, and scores against held-out answer keys—without leaking the key into the prompt.

This is a **research benchmark repo**, not legal advice or a substitute for state licensing.

## What lives where (high level)

| Path | Role |
|------|------|
| [`eval_set/`](eval_set/README.md) | **All benchmark questions and keys**, split by provenance. |
| [`eval_set/from_quizlet_pdfs/`](eval_set/from_quizlet_pdfs/README.md) | **Quizlet PDF pipeline** → raw `questions.txt`, frozen **~165** OpenAI-formatted snapshot, and **800** balanced items (`questions_formatted.txt` + `answers.txt` + bucket CSV/PNG). |
| [`eval_set/from_youtube_video/`](eval_set/from_youtube_video/README.md) | **YouTube / transcript track** (~113 comparable IDs): `questions.txt`, `answers.txt`, `explanations.txt`, GPT-4.1 bucket CSV/PNG, and [`Results.md`](eval_set/from_youtube_video/Results.md). |
| [`scripts/`](scripts/README.md) | PDF → text, OpenAI formatters, synthetic generation, bucket balance, model runners, judges, plots. |
| [`judge_runs_openai/`](judge_runs_openai/README.md) | Judge outputs (CSV + JSONL). |
| [`data/quizlet/`](data/README.md) | Flashcard dumps used to infer answers for raw Quizlet stems. |
| [`source_material/`](source_material/README.md) | Handbooks, code collections, transcripts. |
| [`results/`](results/README.md) | Model run CSVs. |
| [`Results_plots/`](Results_plots/README.md) | Exported figures for reports. |

Every folder above includes a **README.md** that names each important file and explains how it was produced.

## Quick runs

**Questions-only (no answer key in prompt)** — example with OpenAI:

```bash
.venv/bin/python3 scripts/run_questions_reasoned_openai.py \
  --questions eval_set/from_youtube_video/questions.txt \
  --model gpt-4.1-mini \
  --out results/openai_gpt-4.1-mini_reasoned.csv
```

For the **800-Q Quizlet** set, point `--questions` at `eval_set/from_quizlet_pdfs/questions_formatted.txt` and score against `eval_set/from_quizlet_pdfs/answers.txt`.

**Judge (uses explanations track by default):**

```bash
.venv/bin/python3 scripts/judge_reasoning_openai.py \
  --questions eval_set/from_youtube_video/questions.txt \
  --answers eval_set/from_youtube_video/answers.txt \
  --explanations eval_set/from_youtube_video/explanations.txt \
  --run-dir judge_runs_openai/gpt-4.1-mini
```

## Historical accuracy note (YouTube track)

Older tables in [`eval_set/from_youtube_video/Results.md`](eval_set/from_youtube_video/Results.md) reference **`eval_set/from_youtube_video/answers.txt`** (~**113** comparable question IDs). The newer **800-Q** Quizlet track is separate; size and difficulty differ.

## Hygiene

- Do **not** pass answer files into model prompts, retrieval corpora, or training data for the same benchmark you score.
