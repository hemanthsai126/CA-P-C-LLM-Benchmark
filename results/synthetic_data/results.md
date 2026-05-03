# Option run vs answer key — `synthetic_data`

Outputs under `option/*.csv`: Ollama uses `answer`; OpenAI runs use **`option`**. Ground truth is `answers.txt` (**400** IDs). **Accuracy** = correct / overlap.

| CSV | Parameters | Correct | Wrong | Accuracy |
|-----|------------|--------:|------:|----------:|
| `gpt-4o_reasoned_synthetic_data.csv` | Not disclosed | 398 | 2 | **99.50%** |
| `gpt-4_1-mini_reasoned_synthetic_data.csv` | ~7B (est.) | 393 | 7 | **98.25%** |
| `gpt-4o-mini_reasoned_synthetic_data.csv` | ~8B (est.) | 392 | 8 | **98.00%** |
| `gpt-4_1-nano_reasoned_synthetic_data.csv` | ~4B (est.) | 382 | 18 | **95.50%** |
| `gpt-3_5-turbo_reasoned_synthetic_data.csv` | Not disclosed | 380 | 20 | **95.00%** |
| `gemma2_9b_reasoned.csv` | 9B | 379 | 21 | **94.75%** |
| `llama3.1_8b_reasoned.csv` | 8B | 347 | 53 | **86.75%** |
| `qwen2.5_7b_reasoned.csv` | 7B | 342 | 58 | **85.50%** |
| `mistral_7b_reasoned.csv` | 7B | 330 | 70 | **82.50%** |
| `gemma_2b_reasoned.csv` | 2B | 196 | 204 | **49.00%** |
| `phi3_mini_reasoned.csv` | 3.8B | 189 | 211 | **47.25%** |
| `tinyllama_reasoned.csv` | 1.1B | 110 | 290 | **27.50%** |

Rows sorted by accuracy (highest first). **Parameters:** Ollama weights use published sizes where listed (TinyLlama 1.1B, Gemma 2 2B, Gemma 2 9B, Phi-3 Mini 3.8B, Qwen2.5-7B, Mistral 7B, Llama 3.1 8B). OpenAI API models: “Not disclosed” / “est.” as in `synthetic_data/results.md`.
