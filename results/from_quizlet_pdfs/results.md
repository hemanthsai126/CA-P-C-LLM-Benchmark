# Option run vs answer key — `from_quizlet_pdfs`

Predictions live under `option/*.csv` (`question_number`, `answer` or `option`, `reason`). Ground truth is `answers.txt`. Overlap excludes rows where the key is not A–D. **Accuracy** = correct / overlapped IDs.

| CSV | Parameters | Correct | Wrong | Accuracy |
|-----|------------|--------:|------:|----------:|
| `gpt-4o_reasoned_from_quizlet_pdfs.csv` | -- | 410 | 140 | **74.55%** |
| `gpt-4o-mini_reasoned_from_quizlet_pdfs.csv` | ~8B (est.) | 390 | 160 | **70.91%** |
| `gpt-4_1-mini_reasoned_from_quizlet_pdfs.csv` | ~7B (est.) | 378 | 172 | **68.73%** |
| `gemma2_9b_reasoned.csv` | 9B | 363 | 187 | **66.00%** |
| `llama3.1_8b_reasoned.csv` | 8B | 360 | 190 | **65.45%** |
| `gpt-3_5-turbo_reasoned_from_quizlet_pdfs.csv` | -- | 356 | 194 | **64.73%** |
| `gpt-4_1-nano_reasoned_from_quizlet_pdfs.csv` | ~4B (est.) | 352 | 198 | **64.00%** |
| `qwen2.5_7b_reasoned.csv` | 7B | 308 | 242 | **56.00%** |
| `mistral_7b_reasoned.csv` | 7B | 293 | 257 | **53.27%** |
| `gemma_2b_reasoned.csv` | 2B | 234 | 316 | **42.55%** |
| `phi3_mini_reasoned.csv` | 3.8B | 196 | 354 | **35.64%** |
| `tinyllama_reasoned.csv` | 1.1B | 161 | 389 | **29.27%** |

Rows sorted by accuracy (highest first). **Parameters:** Ollama weights use published sizes where listed (TinyLlama 1.1B, Gemma 2 2B, Gemma 2 9B, Phi-3 Mini 3.8B, Qwen2.5-7B, Mistral 7B, Llama 3.1 8B). OpenAI API models: “Not disclosed” / “est.” as in `synthetic_data/results.md`.
