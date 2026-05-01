# Results (`results/`)

CSV outputs from **model runs** (questions-only: option + reason, or two-phase logs). File names usually encode provider and model, e.g. `openai_gpt-4.1-mini_reasoned.csv`.

**How we got them:** Run `scripts/run_questions_reasoned_openai.py` (or Ollama `run_questions_reasoned.py`, two-phase runner, etc.) with `--out results/<name>.csv`. Score accuracy offline by joining question IDs to the matching `eval_set/.../answers.txt`.

See root `README.md` and [`eval_set/from_youtube_video/Results.md`](../eval_set/from_youtube_video/Results.md) for headline numbers on the historical **YouTube-track** subset (~113 IDs).
