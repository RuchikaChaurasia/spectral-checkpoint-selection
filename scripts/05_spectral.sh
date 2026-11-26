#!/usr/bin/env bash
# Step 5 (Ruchika): WeightWatcher alpha for every epoch of the 12 fine-tuning runs.
# About 2 s per checkpoint on CPU, so about 12 minutes in total. Safe to re-run: finished epochs are skipped.
#   bash scripts/05_spectral.sh                                  # checkpoints/ in the repo
#   bash scripts/05_spectral.sh /content/drive/MyDrive/DeepLearning/group9/checkpoints
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.spectral.ww --checkpoints-dir "${1:-checkpoints}" --out results/alpha
