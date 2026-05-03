# Option run vs answer key — `from_youtube_video`

Local outputs under `option/*.csv`: Ollama uses `answer`; OpenAI CSVs use **`option`**. Ground truth is `answers.txt` (**150** IDs). **Accuracy** = correct / overlap.

| CSV | Parameters | Correct | Wrong | Accuracy |
|-----|------------|--------:|------:|----------:|
| `gpt-4o_reasoned_from_youtube_video.csv` | Not disclosed | 137 | 13 | **91.33%** |
| `gpt-4_1-mini_reasoned_from_youtube_video.csv` | ~7B (est.) | 136 | 14 | **90.67%** |
| `gpt-3_5-turbo_reasoned_from_youtube_video.csv` | Not disclosed | 133 | 17 | **88.67%** |
| `gpt-4o-mini_reasoned_from_youtube_video.csv` | ~8B (est.) | 132 | 18 | **88.00%** |
| `gpt-4_1-nano_reasoned_from_youtube_video.csv` | ~4B (est.) | 129 | 21 | **86.00%** |
| `gemma2_9b_reasoned.csv` | 9B | 128 | 22 | **85.33%** |
| `qwen2.5_7b_reasoned.csv` | 7B | 118 | 32 | **78.67%** |
| `llama3.1_8b_reasoned.csv` | 8B | 116 | 34 | **77.33%** |
| `mistral_7b_reasoned.csv` | 7B | 103 | 47 | **68.67%** |
| `phi3_mini_reasoned.csv` | 3.8B | 100 | 50 | **66.67%** |
| `gemma_2b_reasoned.csv` | 2B | 66 | 84 | **44.00%** |
| `tinyllama_reasoned.csv` | 1.1B | 36 | 114 | **24.00%** |

Rows sorted by accuracy (highest first). **Parameters:** Ollama weights use published sizes where listed (TinyLlama 1.1B, Gemma 2 2B, Gemma 2 9B, Phi-3 Mini 3.8B, Qwen2.5-7B, Mistral 7B, Llama 3.1 8B). OpenAI API models: “Not disclosed” / “est.” as in `synthetic_data/results.md`.
