#!/usr/bin/env bash
# Step 1 (Disha): audit the dataset, then make the fixed split and label budgets.
# Run from the repo root:  bash scripts/01_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.audit
python -m src.data.make_splits
echo
echo "Next: commit configs/splits/ and results/data_audit/, then fill in docs/dataset_card.md."
