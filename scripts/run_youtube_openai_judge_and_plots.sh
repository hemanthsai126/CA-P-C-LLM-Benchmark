#!/usr/bin/env bash
# Run OpenAI judge on all results/from_youtube_video/option/*_reasoned.csv, then write plots
# under results/from_youtube_video/judge_plots/ and copy summary.csv next to them.
#
# Requires: OPENAI_API_KEY in the environment or in repo-root .env (loaded by the Python judge).
# Usage: from repo root,  ./scripts/run_youtube_openai_judge_and_plots.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${REPO}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python3; fi

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini}"
"$PY" scripts/judge_reasoning_openai.py --judge-model "$JUDGE_MODEL" --reasoning-effort off --sleep-ms 50

OUT="${REPO}/results/from_youtube_video/judge_plots"
mkdir -p "$OUT"
"$PY" scripts/plot_judge_summary.py \
  --summary "${REPO}/judge_runs_openai/${JUDGE_MODEL}/summary.csv" \
  --out-dir "$OUT"

cp "${REPO}/judge_runs_openai/${JUDGE_MODEL}/summary.csv" \
  "${REPO}/results/from_youtube_video/judge_summary_${JUDGE_MODEL}.csv"

echo "Judge JSONL + summary: ${REPO}/judge_runs_openai/${JUDGE_MODEL}/"
echo "Plots + summary copy:  ${OUT}/  and  results/from_youtube_video/judge_summary_${JUDGE_MODEL}.csv"
