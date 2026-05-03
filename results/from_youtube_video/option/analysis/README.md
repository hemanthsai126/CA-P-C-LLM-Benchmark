# Bucket ablation — `from_youtube_video`

Ground truth: `../answers.txt` (relative to this folder). Eight curriculum buckets (1 = Basic concepts … 8 = Commercial insurance).

## Per-model strongest / weakest bucket (by accuracy)

| Model | Best bucket | Acc | Worst bucket | Acc |
|-------|-------------|-----|--------------|-----|
| `gpt-4o` | 1 Basic concepts | **100.0%** | 7 Personal liability & inland marine | **75.0%** |
| `gpt-4_1-mini` | 1 Basic concepts | **100.0%** | 5 Residential property | **75.0%** |
| `gpt-4o-mini` | 4 Tort & property fundamentals | **100.0%** | 5 Residential property | **70.8%** |
| `gpt-3_5-turbo` | 4 Tort & property fundamentals | **100.0%** | 7 Personal liability & inland marine | **62.5%** |
| `gpt-4_1-nano` | 4 Tort & property fundamentals | **100.0%** | 5 Residential property | **75.0%** |
| `gemma2_9b` | 6 Valuation & government programs | **100.0%** | 4 Tort & property fundamentals | **50.0%** |
| `llama3.1_8b` | 4 Tort & property fundamentals | **100.0%** | 5 Residential property | **58.3%** |
| `qwen2.5_7b` | 4 Tort & property fundamentals | **100.0%** | 5 Residential property | **58.3%** |
| `phi3_mini` | 4 Tort & property fundamentals | **100.0%** | 8 Commercial insurance | **54.2%** |
| `mistral_7b` | 6 Valuation & government programs | **100.0%** | 4 Tort & property fundamentals | **50.0%** |
| `gemma_2b` | 6 Valuation & government programs | **100.0%** | 4 Tort & property fundamentals | **0.0%** |
| `tinyllama` | 4 Tort & property fundamentals | **50.0%** | 7 Personal liability & inland marine | **0.0%** |

## Files

- `bucket_accuracy_long.csv` — all models × buckets
- `heatmap_models_x_buckets.png` — overview
- `mean_accuracy_by_bucket.png` — which buckets are hardest on average
- `*_by_bucket.png` — one bar chart per model CSV

