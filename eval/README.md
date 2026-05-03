# Package `eval` (`eval/`)

Small **Python package** (not the same as the **`eval_set/`** benchmark directory) for running **local Ollama** MCQ evaluation: loading labels, building prompts, and writing CSV rows. The on-disk eval tree is **`DATA/eval_set/`**; **`eval_set/`** at the repo root is a symlink there.

| Module | Role |
|--------|------|
| `cli.py` | Command-line entry for Ollama-backed runs. |
| `ollama_runner.py` | Async evaluation loop against a running Ollama server. |
| `prompts.py` | Prompt templates (questions-only; answer key kept separate). |

See the root `README.md` for how this relates to `eval_set/from_youtube_video/` vs `eval_set/from_quizlet_pdfs/`.
