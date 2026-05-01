# Data (`data/`)

Static inputs that are **not** secret API keys: dumps, raw captures, and intermediate corpora.

## `quizlet/`

| File | Description |
|------|-------------|
| `quizlet_ca_flashcard_dump.md` | Markdown export of a **California P&C** Quizlet flashcard set (terms/definitions). Used by `scripts/fill_answers_from_quizlet_flashcards.py` to infer answers for stems. |
| `quizlet_ins_flashcard_dump.md` | Same pattern for a broader **P&C** flashcard set. |

## Other subfolders (`external/`, `processed/`, `raw/`)

Project-specific imports (e.g. scraped pages, normalized tables). Add a one-line note inside each subfolder when you introduce new datasets.
