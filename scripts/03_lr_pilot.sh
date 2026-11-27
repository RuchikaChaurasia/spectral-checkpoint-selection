#!/usr/bin/env bash
# Step 3 (Disha): learning-rate pilot for fine-tuning (1e-4 vs 3e-4, seed 42,
# 100% budget), chosen on Product validation loss only. Writes configs/lr_choice.json.
#   bash scripts/03_lr_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.run_all --pilot
python -m src.train.select_lr
python -m src.train.report_runs --root checkpoints/pilot
