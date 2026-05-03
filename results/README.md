# Results (`results/` / `RESULTS/`)

On a **case-insensitive** filesystem (typical macOS default), `results/` and `RESULTS/` are the **same directory**. Scripts and docs often use lowercase **`results/`** for `--out` paths to model-run CSVs.

## Model run CSVs

CSV outputs from **model runs** (questions-only: chosen letter + short reason, or two-phase logs). Every run file uses **exactly three columns:** `question_number`, `answer`, `reason` (no other columns). Names often encode provider and model, e.g. `openai_gpt-4.1-mini_reasoned.csv`.

**How we got them:** Run `scripts/run_questions_reasoned_openai.py` (or Ollama `scripts/run_questions_reasoned.py`, two-phase runner, etc.) with `--out results/<name>.csv`. Score accuracy offline by joining question IDs to the matching `eval_set/.../answers.txt`.

See root [`README.md`](../README.md) and [`eval_set/from_youtube_video/Results.md`](../eval_set/from_youtube_video/Results.md) for headline numbers on the **YouTube-track** eval (currently **150** question IDs in `questions.txt` / `answers.txt`).

---

## Benchmark snapshots (in this folder)

Curated **copies** for packaging or papers (full editable trees live under **`DATA/eval_set/`**, also reachable as **`eval_set/`** from the repo root—see root `README.md`).

| Path | Role |
|------|------|
| [`from_quizlet_pdfs/questions.txt`](from_quizlet_pdfs/questions.txt) | **552** block-format MCQs (Quizlet Test PDF → `quizlet_test_pdf_to_questions_txt.py`). |
| [`from_quizlet_pdfs/answers.txt`](from_quizlet_pdfs/answers.txt) | Ground-truth letters for IDs **1–552** where known (**`?`** for a couple of garbled items). |
| [`from_youtube_video/questions.txt`](from_youtube_video/questions.txt) | **150** block-format MCQs. |
| [`from_youtube_video/answers.txt`](from_youtube_video/answers.txt) | Matching answer key. |
| [`from_youtube_video/explanations.txt`](from_youtube_video/explanations.txt) | Reference explanations (may start with `#` comment lines). |
| [`synthetic_data/questions_formatted.txt`](synthetic_data/questions_formatted.txt) | **400** synthetic MCQs (**50 per bucket × 8**), trimmed from a larger handbook run; handbook + LEM PDF references at generation time. |
| [`synthetic_data/answers.txt`](synthetic_data/answers.txt) | Ground-truth letters for IDs **1–400**. |
| [`synthetic_data/question_buckets_generator_layout.png`](synthetic_data/question_buckets_generator_layout.png) | Bar chart for **generator layout** by ID order (**50 per bucket**: 1–50 → bucket 1, …, 351–400 → bucket 8). Produced with `scripts/write_ordered_layout_bucket_plot.py --per-bucket 50`. |

The synthetic folder intentionally keeps **only** questions, answers, this PNG, and **`option/`** model outputs (no separate layout CSV in-tree unless you add one).

### GPT-4.1 bucket plots (optional, other folders)

`question_buckets_gpt-4.1.*` (if present beside a questions file) is “what bucket would GPT-4.1 assign from reading the stem?” — **independent** of generator layout. It is **not** the curriculum split used for synthetic generation.

To classify every questions file under `results/` with **gpt-4.1** and save **`question_buckets_gpt-4.1.csv`** + **`question_buckets_gpt-4.1.png`** beside each file:

```bash
export OPENAI_API_KEY=...
.venv/bin/python3 scripts/bucket_results_subfolders_openai.py --model gpt-4.1 --sleep-ms 40
```

Dry-run (list files only): `--dry-run`. This issues **one API call per question** (e.g. 165 + 150 + 400 for the current snapshot tree).

### Local models (`option/` CSVs)

**Ollama** runs use **only** the questions file (no answer key). Each CSV has **`question_number`, `answer`, `reason`**.

- Single file: `scripts/run_questions_reasoned.py --questions … --out …/option/<model>.csv`
- All three `results/` snapshots:

```bash
.venv/bin/python3 scripts/run_results_option_ollama_batch.py --model tinyllama
```

Same columns, reasoned filename convention:

```bash
.venv/bin/python3 scripts/run_results_option_ollama_batch.py --model tinyllama --reasoned-csv
```

Writes e.g. `results/from_quizlet_pdfs/option/tinyllama_reasoned.csv`. Use `--only from_youtube_video` and `--limit 5` while testing. Requires `ollama serve` and `ollama pull tinyllama` (or your tag).

**Shrinking synthetic to 50/bucket:** `scripts/trim_synthetic_per_bucket_openai.py` (OpenAI or `--strategy first50`).

### Accuracy vs `answers.txt` (`results.md`)

After `option/*.csv` files change, regenerate **`results.md`** in each source folder:

```bash
.venv/bin/python3 scripts/write_option_accuracy_results_md.py
```
