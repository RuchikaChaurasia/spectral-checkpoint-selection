#!/usr/bin/env bash
# Step 4 (Disha): all 24 main runs (12 fine-tuning + 12 linear probe), budget
# order 100 -> 10 -> 25 -> 50. Safe to re-run: finished runs are skipped and an
# interrupted run continues where it stopped.
#   bash scripts/04_train_all.sh
#   bash scripts/04_train_all.sh --budgets 100      # minimum viable project only
#   bash scripts/04_train_all.sh --epoch-hook src.spectral.ww:on_epoch   # if Ruchika wants a per-epoch hook
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.run_all "$@"
python -m src.train.report_runs
