# Data hub (`DATA/` / `data/`)

Static inputs that are **not** secret API keys: dumps, raw captures, intermediate corpora, and the **full eval benchmarks**.

On a **case-insensitive** filesystem (typical macOS default), `DATA/` and `data/` refer to the **same directory**. This file is named `README_data_imports.md` so it does not collide with other READMEs inside the hub.

## Layout

| Path (under this hub) | Role |
|----------------------|------|
| **`eval_set/`** | Canonical **benchmark files**: `from_quizlet_pdfs/`, `from_youtube_video/`, READMEs, backups, bucket CSV/PNGs. The repo root has a symlink **`eval_set` → `DATA/eval_set`**. |
| **`quizlet/`** | Flashcard dumps (see below). |
| **`processed/`** | Normalized tables / extractions (e.g. CSVs from PDF MCQ pulls). |
| **`external/`**, **`raw/`** | Project-specific imports. Add a one-line note when you add datasets. |

See [`eval_set/README.md`](eval_set/README.md) for the two benchmark tracks and [`results/README.md`](../results/README.md) for **snapshot copies** of the 165-Q and 150-Q sets.

## `quizlet/`

| File | Description |
|------|-------------|
| `quizlet_ca_flashcard_dump.md` | Markdown export of a **California P&C** Quizlet flashcard set (terms/definitions). Used by `scripts/fill_answers_from_quizlet_flashcards.py` to infer answers for stems. |
| `quizlet_ins_flashcard_dump.md` | Same pattern for a broader **P&C** flashcard set. |
