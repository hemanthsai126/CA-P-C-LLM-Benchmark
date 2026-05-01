# Judge runs (`judge_runs_openai/`)

Outputs from **rubric judges** that score model-written **reasons** (and optionally MCQ correctness) against the eval set—without putting the answer key in the prompt.

## Subfolders

| Path | Contents |
|------|----------|
| `gpt-4.1/` | Judge run using OpenAI **GPT-4.1** (e.g. `summary.csv`, JSONL transcripts). |
| `gpt-4.1-mini/` | Same pipeline with **GPT-4.1-mini** (often used for cheaper replays). Includes summary plots — see [`gpt-4.1-mini/README.md`](gpt-4.1-mini/README.md). |
| `external/` | Third-party or hand-copied judge artifacts, if present. |

Typical artifacts:

- **`summary.csv`** — one row per evaluated model response: scores, flags, optional alignment metrics.
- **`.jsonl`** — line-delimited full judge payloads for audit or replots.

## How these files were produced

Run from the repo root (requires the right API key):

```bash
.venv/bin/python3 scripts/judge_reasoning_openai.py \
  --questions eval_set/from_youtube_video/questions.txt \
  --answers eval_set/from_youtube_video/answers.txt \
  --explanations eval_set/from_youtube_video/explanations.txt \
  --run-dir judge_runs_openai/gpt-4.1-mini
```

Paths above match the **YouTube / transcript** eval (default for the judge scripts). Point `--questions` / `--answers` at `eval_set/from_quizlet_pdfs/` if you judge runs on the **800-Q** set instead.

Plots derived from `summary.csv` may be copied to `Results_plots/` for reports.
