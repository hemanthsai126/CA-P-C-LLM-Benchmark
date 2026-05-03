# Scripts (`scripts/`)

Command-line utilities for building evals, running models, judging, and plotting. Run from the repo root unless noted.

**Paths:** defaults such as `eval_set/...` assume the repo-root **`eval_set`** symlink (→ `DATA/eval_set/`). Flashcard dumps default to `data/quizlet/` (same folder as `DATA/quizlet/` on a case-insensitive volume). See root [`README.md`](../README.md) and [`DATA/README_data_imports.md`](../DATA/README_data_imports.md).

## Eval construction

| Script | Purpose |
|--------|---------|
| `quizlet_pdfs_to_eval_txt.py` | Parse Quizlet-export PDFs → `questions.txt` + `answers_from_quizlet_pdf_parse.txt` under `--out-dir` (default `eval_set/from_quizlet_pdfs/`). Does **not** overwrite the curated `answers.txt` used for the 800-Q set. |
| `quizlet_test_pdf_to_questions_txt.py` | Quizlet **Test** print PDF (Term/Definition + 4 lines + ``N of M``) → block `questions.txt` (e.g. `results/from_quizlet_pdfs/CA555.pdf`). |
| `format_questions_openai.py` | OpenAI pass to **clean OCR / wording** on a block file; write e.g. `questions_formatted_165.txt` (use explicit `--out` so you do not overwrite the 800-Q file). |
| `fill_answers_from_quizlet_flashcards.py` | Match raw `questions.txt` stems to Quizlet **flashcard dumps** in `data/quizlet/`; writes a separate answers file (default: `answers_from_flashcards_for_questions_txt.txt`). |
| `generate_synthetic_handbook_mcqs.py` | Generate synthetic MCQs from Insurance Handbook excerpts + OpenAI (full rebalance mode). |
| `balance_eval_buckets_layout.py` | **Trim** overfull buckets and **fill** short ones to hit `--target` per bucket (default 100 → 800 total); rewrites formatted questions, answers, heuristic CSV/PNG. |
| `dedupe_questions_openai.py` | Exact + **gpt-4o-mini** semantic dedupe on `questions.txt`; optional `--answers` renumber; writes `questions_issues_report.txt` for manual cleanup. |
| `bucket_questions_heuristic.py` | Keyword **heuristic** bucket labels + CSV + PNG (no API). |
| `bucket_questions_openai.py` | **OpenAI** bucket labels + CSV + PNG (needs `OPENAI_API_KEY`). |
| `bucket_results_subfolders_openai.py` | Run **`bucket_questions_openai.py`** on every `results/**/questions.txt` and `results/**/questions_formatted*.txt`; writes `question_buckets_gpt-4.1.csv` + `.png` next to each file. |
| `write_ordered_layout_bucket_plot.py` | **No API:** CSV + PNG from **question index** blocks (`--per-bucket`, default 100) — matches `generate_synthetic_handbook_mcqs.py` layout; use when GPT-4.1 bucket plots are not the intended curriculum split. |
| `trim_synthetic_per_bucket_openai.py` | Shrink ordered-bucket synthetic sets (e.g. 100×8 → 50×8) with optional OpenAI pick per bucket; rewrites questions + answers (+ optional layout CSV). |

## Model runs

| Script | Purpose |
|--------|---------|
| `run_questions_reasoned.py` | **Local Ollama:** questions file only (no answer key) → CSV columns `question_number`, `answer`, `reason`. |
| `run_results_option_ollama_batch.py` | Run `run_questions_reasoned.py` for all three `results/{from_quizlet_pdfs,from_youtube_video,synthetic_data}/` question files; writes `option/<model>.csv` (add `--reasoned-csv` for `option/<model>_reasoned.csv`). |
| `write_option_accuracy_results_md.py` | Compare each `results/<source>/option/*.csv` to `answers.txt`; write `results.md` with accuracy tables per model CSV. |
| `run_questions_reasoned_openai.py` | Questions-only (Open API): A–D + short reason → same 3-column CSV as Ollama. |
| `run_questions_two_phase_openai.py` | Two-phase OpenAI protocol (stem reasoning, then options). |
| `repair_quizlet_mcqs_from_pdfs_openai.py` | Reformat **165** Quizlet MCQs + `answers.txt` from two PDFs via OpenAI (default **gpt-5.5**); or `--answers-only` for keys only. |

## Judging and plots

| Script | Purpose |
|--------|---------|
| `judge_reasoning_openai.py` | OpenAI **judge** scores each ``reason`` in ``results/from_youtube_video/option/*_reasoned.csv`` vs ``results/from_youtube_video/explanations.txt`` (0–3 alignment); outputs under ``judge_runs_openai/<judge-model>/``. Loads ``OPENAI_API_KEY`` from the environment or repo-root ``.env``. |
| `run_youtube_openai_judge_and_plots.sh` | Runs that judge for all YouTube runs, then ``plot_judge_summary.py`` → ``results/from_youtube_video/judge_plots/*.png`` and copies ``summary.csv`` to ``results/from_youtube_video/judge_summary_<model>.csv``. |
| `analyze_option_buckets.py` | Per-bucket MCQ accuracy for each ``results/*/option/*_reasoned*.csv``; writes ``option/analysis/`` (heatmaps, per-model bar charts, ``bucket_accuracy_long.csv``, ``README.md``). Uses ``question_buckets_gpt-4.1.csv`` or synthetic 50×8 layout. |
| `judge_reasoning_anthropic.py` | Same idea on Anthropic. |
| `plot_judge_summary.py` | Matplotlib summaries from `summary.csv`. |

## Other

| Script | Purpose |
|--------|---------|
| `pdf_mcq_to_csv.py` | Ad-hoc PDF MCQ extraction helpers. |
| `transcript_to_validation_split.py` | Transcript-derived validation splits. |
| `refine_transcript_prose.py` | Transcript cleanup. |
