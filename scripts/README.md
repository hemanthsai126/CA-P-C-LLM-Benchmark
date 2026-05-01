# Scripts (`scripts/`)

Command-line utilities for building evals, running models, judging, and plotting. Run from the repo root unless noted.

## Eval construction

| Script | Purpose |
|--------|---------|
| `quizlet_pdfs_to_eval_txt.py` | Parse Quizlet-export PDFs → `questions.txt` + `answers_from_quizlet_pdf_parse.txt` under `--out-dir` (default `eval_set/from_quizlet_pdfs/`). Does **not** overwrite the curated `answers.txt` used for the 800-Q set. |
| `format_questions_openai.py` | OpenAI pass to **clean OCR / wording** on a block file; write e.g. `questions_formatted_165.txt` (use explicit `--out` so you do not overwrite the 800-Q file). |
| `fill_answers_from_quizlet_flashcards.py` | Match raw `questions.txt` stems to Quizlet **flashcard dumps** in `data/quizlet/`; writes a separate answers file (default: `answers_from_flashcards_for_questions_txt.txt`). |
| `generate_synthetic_handbook_mcqs.py` | Generate synthetic MCQs from Insurance Handbook excerpts + OpenAI (full rebalance mode). |
| `balance_eval_buckets_layout.py` | **Trim** overfull buckets and **fill** short ones to hit `--target` per bucket (default 100 → 800 total); rewrites formatted questions, answers, heuristic CSV/PNG. |
| `bucket_questions_heuristic.py` | Keyword **heuristic** bucket labels + CSV + PNG (no API). |
| `bucket_questions_openai.py` | **OpenAI** bucket labels + CSV + PNG (needs `OPENAI_API_KEY`). |

## Model runs

| Script | Purpose |
|--------|---------|
| `run_questions_reasoned_openai.py` | Questions-only: model picks A–D + short reason → CSV. |
| `run_questions_two_phase_openai.py` | Two-phase OpenAI protocol (stem reasoning, then options). |
| `run_questions_reasoned.py` | Ollama / local variant of the reasoned runner. |

## Judging and plots

| Script | Purpose |
|--------|---------|
| `judge_reasoning_openai.py` | Score model reasons vs reference explanations (OpenAI judge). |
| `judge_reasoning_anthropic.py` | Same idea on Anthropic. |
| `plot_judge_summary.py` | Matplotlib summaries from `summary.csv`. |

## Other

| Script | Purpose |
|--------|---------|
| `pdf_mcq_to_csv.py` | Ad-hoc PDF MCQ extraction helpers. |
| `transcript_to_validation_split.py` | Transcript-derived validation splits. |
| `refine_transcript_prose.py` | Transcript cleanup. |
