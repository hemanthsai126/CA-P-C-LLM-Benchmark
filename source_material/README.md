# Source material (`source_material/`)

Primary **reference documents** and California-specific corpora used for question generation, RAG-style prompts, or study alignment.

## `Cali Data/`

| Item | Description |
|------|-------------|
| `Insurance_Handbook_20103.pdf` | Insurance Information Institute **Insurance Handbook** (reference text for synthetic / balanced MCQ generation in `scripts/generate_synthetic_handbook_mcqs.py` and `scripts/balance_eval_buckets_layout.py`). |
| `PC_LEM_FINALONLINE_2.pdf` | Additional California P&C / exam-oriented PDF reference. |
| `Cali_ins_codes/` | Collection of California insurance code snippets or exports (used when building code-aware items). |
| `Youtube transcripts/` | Raw or cleaned transcripts tied to the **YouTube-track** eval in `eval_set/from_youtube_video/`. |

## `pdfs/` / `text/`

Project PDFs and extracted text mirrors (e.g. benchmark plans, converted handbooks). Prefer keeping **one canonical binary** (PDF) plus generated text under `text/` when extraction is expensive.
