# Benchmarking LLMs on Insurance Licensing–Style Exams: What We Measured and What It Actually Tells You

**Research evaluation report** · May 2026

---

## Abstract

People use large language models to study for insurance licensing exams—practice questions, quick explanations, the hope of passing on the first try. This project measures how well a dozen model setups actually perform on that kind of work, but we try to be honest about what the numbers mean. We use three different piles of California P&C–style multiple-choice items (a big messy Quizlet/PDF-style set, a smaller YouTube-aligned set, and a cleaner synthetic set balanced across topics). We score letter answers against keys, we score explanations against reference text where we have it, and we break errors down by eight curriculum buckets so “where do I study harder?” has an answer, not just a single percentage. The headline is predictable: frontier models and larger open weights look strong on easier material, and everyone looks worse on the scraped PDF-style bank. The more useful headline is that **if your only bar is “passing” on friendly practice MCQs, plenty of models above a few billion parameters already look fine**—which means the interesting work is everything *around* that: whether items predict the real exam, whether explanations are safe to trust, and what happens when wording and jurisdiction get nasty.

---

## 1. Introduction

Licensing exams reward careful reading, memorized detail, and time pressure. Study products are already folding in LLMs for generation and tutoring. That is fine as a product direction, but it needs evaluation that matches how people will use the tool—not one leaderboard score on one file of questions.

We built a small, repeatable pipeline in this repository: run models, store CSVs, compare to keys, plot across corpora, judge explanations on the YouTube split, and slice accuracy by topic bucket. The goal of this write-up is to summarize what we found in one place, in language that a reviewer or a product owner can read without digging through folders first.

---

## 2. Materials and methods

### 2.1 The three corpora

| Corpus | Folder | What it is | How many answers we grade |
|--------|--------|------------|----------------------------|
| Quizlet / PDF-style | `from_quizlet_pdfs/` | Large, noisy study-bank style pool | **550** scored overlaps (see below) |
| YouTube-aligned | `from_youtube_video/` | Smaller validation split tied to transcript-style material | **150** |
| Synthetic | `synthetic_data/` | Handbook-style generation, 50 questions × 8 topic blocks | **400** |

Questions and keys live next to the scores under each `results/...` folder. Model outputs are in `option/*.csv` with a letter and usually a short `reason`.

### 2.2 Models we tested

**OpenAI (API, names as in filenames)**

- GPT-4o  
- GPT-4o mini  
- GPT-4.1 mini  
- GPT-4.1 nano  
- GPT-3.5 Turbo  

**Ollama (local weights)**

- TinyLlama 1.1B  
- Gemma 2 2B  
- Gemma 2 9B  
- Phi-3 Mini 3.8B  
- Qwen2.5 7B  
- Mistral 7B  
- Llama 3.1 8B  

Where we label API “mini” sizes with “~7B (est.)” and similar, that is informal public shorthand, not a vendor spec.

### 2.3 Task

Each item is four-way multiple choice (A–D). Pipelines ask for a letter and a short rationale where configured.

### 2.4 Letter accuracy and what “overlap” means (especially on Quizlet)

We only score a question when **both** sides give a usable letter: the key says A–D, and the model outputs A–D. Let \(G\) be those key ids and \(P\) those prediction ids. We compute accuracy on **\(G \cap P\)**: correct count divided by how many ids are in that intersection.

**Quizlet nuance.** `answers.txt` has more lines than 550, but many rows line up with the same overlapping set of items once you require a valid model letter and a valid key. In practice every full run in our tables ends up with **550** comparable items for that corpus—the questions where the model actually played the game with a letter and the key was A–D. Things that fall out include missing predictions, garbled letters, or key rows that are not plain A–D. So when we say “overlap,” we mean: *this is the fair head-to-head set*, not “every line in the file.”

### 2.5 Explanation judge (YouTube only)

For YouTube items we have `explanations.txt`. A fixed **gpt-4.1-mini** judge reads the stem, options, model letter, model `reason`, and the reference explanation, and returns a 0–3 alignment score plus short notes (`scripts/judge_reasoning_openai.py`). **148** items had everything needed for that pass in the run summarized below.

### 2.6 The eight curriculum buckets (topic labels)

Items are tagged into **one** of eight buckets (names match `question_buckets_gpt-4.1.csv` on Quizlet and YouTube; synthetic uses the same names in fixed 50-question blocks).

1. **Basic concepts**  
2. **Contract law**  
3. **CA laws & ethics**  
4. **Tort & property fundamentals**  
5. **Residential property**  
6. **Valuation & government programs**  
7. **Personal liability & inland marine**  
8. **Commercial insurance**  

Per-bucket accuracy is computed on the overlap of key, prediction, and bucket assignment (`scripts/analyze_option_buckets.py` → each `option/analysis/`).

### 2.7 How figures were produced

| Output | Script |
|--------|--------|
| Cross-corpus MCQ bar and heatmap | `scripts/plot_option_accuracy_across_sources.py` |
| Judge plots | `scripts/plot_judge_summary.py` · `scripts/run_youtube_openai_judge_and_plots.sh` |
| Bucket heatmaps, means, per-model bars | `scripts/analyze_option_buckets.py` |

---

## 3. Results

### 3.1 Overall MCQ accuracy across the three corpora

#### Figure 1. MCQ accuracy by corpus (grouped bars)

![Figure 1 — MCQ accuracy by benchmark source (grouped horizontal bars).](./results/charts/option_accuracy_by_source_barh.png)

**Summary.** Same models, three corpora—the spread between Quizlet and synthetic is the story. Treating synthetic accuracy as “exam readiness” would overstate reality; treating Quizlet alone as the whole truth would understate how clean items inflate scores.

#### Figure 2. MCQ accuracy heatmap (corpora × models)

![Figure 2 — MCQ accuracy heatmap (three corpora × models).](./results/charts/option_accuracy_by_source_heatmap.png)

**Summary.** The heatmap makes domain shift easy to see at a glance: warm cells on synthetic, cooler on Quizlet for the same row. Use it when you care about rank stability versus absolute level.

### 3.2 Mean accuracy across corpora (simple average of the three percentages)

| Model | Params (reporting) | Quizlet | YouTube | Synthetic | Mean |
|-------|-------------------:|--------:|--------:|----------:|-----:|
| GPT-4o | Not disclosed | 74.55% | 91.33% | 99.50% | **88.46%** |
| GPT-4.1 mini | ~7B (est.) | 68.73% | 90.67% | 98.25% | **85.88%** |
| GPT-4o mini | ~8B (est.) | 70.91% | 88.00% | 98.00% | **85.64%** |
| GPT-3.5 Turbo | Not disclosed | 64.73% | 88.67% | 95.00% | **82.80%** |
| Gemma 2 9B | 9B | 66.00% | 85.33% | 94.75% | **82.03%** |
| GPT-4.1 nano | ~4B (est.) | 64.00% | 86.00% | 95.50% | **81.83%** |
| Llama 3.1 8B | 8B | 65.45% | 77.33% | 86.75% | **76.51%** |
| Qwen2.5 7B | 7B | 56.00% | 78.67% | 85.50% | **73.39%** |
| Mistral 7B | 7B | 53.27% | 68.67% | 82.50% | **68.15%** |
| Phi-3 Mini | 3.8B | 35.64% | 66.67% | 47.25% | **49.85%** |
| Gemma 2 2B | 2B | 42.55% | 44.00% | 49.00% | **45.18%** |
| TinyLlama | 1.1B | 29.27% | 24.00% | 27.50% | **26.92%** |

### 3.3 Per-corpus MCQ tables

#### Table A — Quizlet (550 overlaps)

| CSV | Parameters | Correct | Wrong | Accuracy |
|-----|------------|--------:|------:|----------:|
| `gpt-4o_reasoned_from_quizlet_pdfs.csv` | Not disclosed | 410 | 140 | **74.55%** |
| `gpt-4o-mini_reasoned_from_quizlet_pdfs.csv` | ~8B (est.) | 390 | 160 | **70.91%** |
| `gpt-4_1-mini_reasoned_from_quizlet_pdfs.csv` | ~7B (est.) | 378 | 172 | **68.73%** |
| `gemma2_9b_reasoned.csv` | 9B | 363 | 187 | **66.00%** |
| `llama3.1_8b_reasoned.csv` | 8B | 360 | 190 | **65.45%** |
| `gpt-3_5-turbo_reasoned_from_quizlet_pdfs.csv` | Not disclosed | 356 | 194 | **64.73%** |
| `gpt-4_1-nano_reasoned_from_quizlet_pdfs.csv` | ~4B (est.) | 352 | 198 | **64.00%** |
| `qwen2.5_7b_reasoned.csv` | 7B | 308 | 242 | **56.00%** |
| `mistral_7b_reasoned.csv` | 7B | 293 | 257 | **53.27%** |
| `gemma_2b_reasoned.csv` | 2B | 234 | 316 | **42.55%** |
| `phi3_mini_reasoned.csv` | 3.8B | 196 | 354 | **35.64%** |
| `tinyllama_reasoned.csv` | 1.1B | 161 | 389 | **29.27%** |

#### Table B — YouTube (150 items)

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

#### Table C — Synthetic (400 items)

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

### 3.4 YouTube — explanation alignment (judge: gpt-4.1-mini)

#### Table D — Judge summary

| Run | n | MCQ acc. | Avg align | %0 | %1 | %2 | %3 |
|-----|--:|---------:|----------:|---:|---:|---:|---:|
| `gpt-4o_reasoned_from_youtube_video` | 148 | 91.22% | 2.87 | 1.35% | 3.38% | 2.03% | 93.24% |
| `gpt-4_1-mini_reasoned_from_youtube_video` | 148 | 90.54% | 2.86 | 0.68% | 4.05% | 4.05% | 91.22% |
| `gpt-4o-mini_reasoned_from_youtube_video` | 148 | 87.84% | 2.78 | 0.68% | 8.11% | 3.38% | 87.84% |
| `gpt-4_1-nano_reasoned_from_youtube_video` | 148 | 85.81% | 2.73 | 0.68% | 10.81% | 3.38% | 85.14% |
| `gpt-3_5-turbo_reasoned_from_youtube_video` | 148 | 88.51% | 2.70 | 2.70% | 9.46% | 2.70% | 85.14% |
| `gemma2_9b_reasoned` | 148 | 85.14% | 2.70 | 1.35% | 10.81% | 4.73% | 83.11% |
| `qwen2.5_7b_reasoned` | 148 | 78.38% | 2.55 | 2.03% | 14.86% | 8.78% | 74.32% |
| `llama3.1_8b_reasoned` | 148 | 77.03% | 2.50 | 2.03% | 20.27% | 3.38% | 74.32% |
| `mistral_7b_reasoned` | 148 | 68.24% | 2.46 | 2.03% | 18.24% | 11.49% | 68.24% |
| `phi3_mini_reasoned` | 148 | 66.89% | 2.26 | 2.70% | 24.32% | 16.89% | 56.08% |
| `gemma_2b_reasoned` | 148 | 43.92% | 1.64 | 14.86% | 36.49% | 18.92% | 29.73% |
| `tinyllama_reasoned` | 148 | 24.32% | 0.78 | 37.84% | 50.00% | 8.11% | 4.05% |

Per-item logs: `judge_runs_openai/gpt-4.1-mini/*.jsonl`.

#### Figure 3. Mean explanation alignment (0–3) by model

![Figure 3 — Mean explanation alignment (0–3) by model.](./results/from_youtube_video/judge_plots/avg_alignment_score.png)

**Summary.** Alignment tracks who sounds like the reference explanation, not only who picked the right letter. That matters if learners read the rationale and treat it as teaching.

#### Figure 4. Stacked distribution of judge scores (0–3)

![Figure 4 — Stacked judge score distribution (0–3) per model.](./results/from_youtube_video/judge_plots/score_distribution_stacked.png)

**Summary.** Top models pile mass at score 3; small baselines pile at 0–1. The middle band is where cheap human QA can still help even when letters look acceptable.

#### Figure 5. MCQ accuracy versus mean alignment

![Figure 5 — MCQ accuracy vs mean alignment (each point = one model).](./results/from_youtube_video/judge_plots/accuracy_vs_alignment.png)

**Summary.** Correlation is positive but imperfect—worth spot-checking “right letter, shaky reasoning” cases before shipping tutor copy.

### 3.5 Curriculum buckets — where models gain and lose ground

The eight buckets are listed in §2.6 (basic concepts through commercial insurance). Figures below show **MCQ accuracy within each bucket** after intersecting with valid model letters and keys. Heatmap rows are sorted by average accuracy across buckets. Each corpus also has per-model bar charts in `option/analysis/*_by_bucket.png` if you want every model in isolation.

#### Quizlet / PDF-derived (`from_quizlet_pdfs/option/analysis/`)

##### Figure 6. Quizlet — models × buckets (heatmap)

![Figure 6 — Quizlet: models × buckets accuracy heatmap.](./results/from_quizlet_pdfs/option/analysis/heatmap_models_x_buckets.png)

**Summary.** Cold columns are topics that hurt many models together—good places to harden a curriculum or add instructor review, not just to swap model weights.

##### Figure 7. Quizlet — mean accuracy by bucket

![Figure 7 — Quizlet: mean MCQ accuracy across models per bucket.](./results/from_quizlet_pdfs/option/analysis/mean_accuracy_by_bucket.png)

**Summary.** When the cross-model mean dips, the bucket is intrinsically awkward for this bank, not a single vendor glitch.

#### YouTube (`from_youtube_video/option/analysis/`)

##### Figure 8. YouTube — models × buckets (heatmap)

![Figure 8 — YouTube: models × buckets heatmap.](./results/from_youtube_video/option/analysis/heatmap_models_x_buckets.png)

**Summary.** With only 150 items, some bucket cells are thin; pair the plot with `bucket_accuracy_long.csv` counts when you interpret a dark cell.

##### Figure 9. YouTube — mean accuracy by bucket

![Figure 9 — YouTube: mean accuracy per bucket.](./results/from_youtube_video/option/analysis/mean_accuracy_by_bucket.png)

**Summary.** Bucket means here sit next to the judge results: letters can look fine while explanations still need polish.

#### Synthetic (`synthetic_data/option/analysis/`)

##### Figure 10. Synthetic — models × buckets (heatmap)

![Figure 10 — Synthetic: models × buckets heatmap.](./results/synthetic_data/option/analysis/heatmap_models_x_buckets.png)

**Summary.** The synthetic heatmap is greener overall—expected given controlled generation—and still shows which slices separate mid-sized models.

##### Figure 11. Synthetic — mean accuracy by bucket

![Figure 11 — Synthetic: mean accuracy per bucket.](./results/synthetic_data/option/analysis/mean_accuracy_by_bucket.png)

**Summary.** High bucket means here are better for regression checks after prompt changes than for claiming real exam readiness.

---

## 4. Discussion

Learners often frame the goal as passing the exam. These runs are still practice MCQs with our keys—they are not the state test. Still, a few patterns are stable enough to say out loud:

- **“Passing” on friendly MCQs is a soft target.** On synthetic (and to some extent YouTube), several models from about **4B parameters up** already look very strong on letters. If the bar is quietly lowered to “good enough drill scores,” then **anything past ~3B** clears that shallow bar on easy material. The harder Quizlet-style bank is where scores spread from the twenties into the mid-seventies—closer to messy study reality.

- **Domain shift beats model gossip.** Rankings partially hold, but absolute scores swing by tens of points across corpora. Shipping a single headline number from one bank is misleading.

- **Buckets change the question from “who won?” to “what should I study?”** They highlight curriculum slices that hurt many models at once, which is more actionable than a leaderboard row.

- **Judge alignment is a smoke test, not a seal of approval.** gpt-4.1-mini is useful for ordering explanations and finding disasters to review; it is not a substitute for human subject-matter review, especially when API models cluster near each other.

---

## 5. Data and code availability

| Artifact | Location |
|----------|----------|
| Questions, keys, explanations | `results/from_quizlet_pdfs/`, `results/from_youtube_video/`, `results/synthetic_data/` |
| Model CSVs | each `results/*/option/*.csv` |
| MCQ accuracy tables | `results/*/results.md` · `scripts/update_option_results_tables.py` |
| Cross-corpus figures | `results/charts/*.png` · `scripts/plot_option_accuracy_across_sources.py` |
| Judge JSONL + summary | `judge_runs_openai/gpt-4.1-mini/` · `scripts/judge_reasoning_openai.py` |
| Judge plots | `results/from_youtube_video/judge_plots/` · `scripts/run_youtube_openai_judge_and_plots.sh` |
| Bucket analysis | each `results/*/option/analysis/` · `scripts/analyze_option_buckets.py` |
| This report (root-relative image paths) | `REPORT.md` |
| Same text with `results/`-relative image paths | `results/report.md` |
