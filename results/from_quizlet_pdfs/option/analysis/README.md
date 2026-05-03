# Bucket ablation — `from_quizlet_pdfs`

Ground truth: `../answers.txt` (relative to this folder). Eight curriculum buckets (1 = Basic concepts … 8 = Commercial insurance).

## Per-model strongest / weakest bucket (by accuracy)

| Model | Best bucket | Acc | Worst bucket | Acc |
|-------|-------------|-----|--------------|-----|
| `gpt-4o` | 6 Valuation & government programs | **95.0%** | 2 Contract law | **68.0%** |
| `gpt-4o-mini` | 6 Valuation & government programs | **85.0%** | 2 Contract law | **64.0%** |
| `gpt-4_1-mini` | 4 Tort & property fundamentals | **83.3%** | 3 CA laws & ethics | **60.5%** |
| `gpt-3_5-turbo` | 1 Basic concepts | **72.0%** | 3 CA laws & ethics | **60.5%** |
| `gpt-4_1-nano` | 6 Valuation & government programs | **80.0%** | 3 CA laws & ethics | **55.8%** |
| `gemma2_9b` | 1 Basic concepts | **78.0%** | 2 Contract law | **52.0%** |
| `llama3.1_8b` | 1 Basic concepts | **82.0%** | 2 Contract law | **52.0%** |
| `qwen2.5_7b` | 6 Valuation & government programs | **75.0%** | 3 CA laws & ethics | **44.2%** |
| `mistral_7b` | 6 Valuation & government programs | **70.0%** | 2 Contract law | **44.0%** |
| `gemma_2b` | 2 Contract law | **60.0%** | 1 Basic concepts | **32.0%** |
| `phi3_mini` | 6 Valuation & government programs | **45.0%** | 8 Commercial insurance | **28.0%** |
| `tinyllama` | 6 Valuation & government programs | **35.0%** | 1 Basic concepts | **16.0%** |

## Files

- `bucket_accuracy_long.csv` — all models × buckets
- `heatmap_models_x_buckets.png` — overview
- `mean_accuracy_by_bucket.png` — which buckets are hardest on average
- `*_by_bucket.png` — one bar chart per model CSV

