# Eval set (`eval_set/`)

This directory holds **two separate benchmarks**, split by how the items were produced. Each subfolder has its own `README.md` with a per-file breakdown. **Bucket overview PNGs** are embedded in those READMEs (Quizlet and YouTube folders); the Quizlet **`legacy/`** README also embeds an archived plot.

## Subfolders

| Folder | Role |
|--------|------|
| [`from_quizlet_pdfs/`](from_quizlet_pdfs/README.md) | MCQs derived from **Quizlet-style PDF exports** (parsed locally), then cleaned and expanded with OpenAI and the Insurance Handbook. **Primary large eval:** 800 items (100 per curriculum bucket). |
| [`from_youtube_video/`](from_youtube_video/README.md) | Earlier eval built from **YouTube / transcript-style** material: ~113 comparable MCQs with reference explanations, GPT-4.1 topic labels, and [`Results.md`](from_youtube_video/Results.md) for headline scores. Used by legacy judge scripts by default. |

## Which eval should I use?

- **Broker-style scale + bucket balance:** use `from_quizlet_pdfs/questions_formatted.txt` + `answers.txt` (800 rows).
- **Judge alignment vs human explanations:** use `from_youtube_video/` (`questions.txt`, `answers.txt`, `explanations.txt`).
