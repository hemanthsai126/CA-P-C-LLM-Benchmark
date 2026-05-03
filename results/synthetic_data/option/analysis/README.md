# Bucket ablation — `synthetic_data`

Ground truth: `../answers.txt` (relative to this folder). Eight curriculum buckets (1 = Basic concepts … 8 = Commercial insurance).

## Per-model strongest / weakest bucket (by accuracy)

| Model | Best bucket | Acc | Worst bucket | Acc |
|-------|-------------|-----|--------------|-----|
| `gpt-4o` | 2 Contract law | **100.0%** | 1 Basic concepts | **96.0%** |
| `gpt-4_1-mini` | 2 Contract law | **100.0%** | 4 Tort & property fundamentals | **96.0%** |
| `gpt-4o-mini` | 2 Contract law | **100.0%** | 7 Personal liability & inland marine | **94.0%** |
| `gpt-4_1-nano` | 3 CA laws & ethics | **100.0%** | 2 Contract law | **90.0%** |
| `gpt-3_5-turbo` | 7 Personal liability & inland marine | **100.0%** | 4 Tort & property fundamentals | **90.0%** |
| `gemma2_9b` | 7 Personal liability & inland marine | **100.0%** | 2 Contract law | **92.0%** |
| `llama3.1_8b` | 3 CA laws & ethics | **96.0%** | 2 Contract law | **74.0%** |
| `qwen2.5_7b` | 8 Commercial insurance | **92.0%** | 6 Valuation & government programs | **82.0%** |
| `mistral_7b` | 5 Residential property | **92.0%** | 2 Contract law | **74.0%** |
| `gemma_2b` | 8 Commercial insurance | **64.0%** | 4 Tort & property fundamentals | **34.0%** |
| `phi3_mini` | 4 Tort & property fundamentals | **56.0%** | 1 Basic concepts | **42.0%** |
| `tinyllama` | 2 Contract law | **36.0%** | 5 Residential property | **16.0%** |

## Files

- `bucket_accuracy_long.csv` — all models × buckets
- `heatmap_models_x_buckets.png` — overview
- `mean_accuracy_by_bucket.png` — which buckets are hardest on average
- `*_by_bucket.png` — one bar chart per model CSV

