#!/usr/bin/env bash
# Step 6 (Ruchika): both rules and all ablation variants -> results/selections/frozen_selections.json.
# Run ONCE, before anyone scores Real-World images; then commit the file and tag the commit:
#   git add results/selections/frozen_selections.json results/alpha
#   git commit -m "Freeze selections" && git tag -a selections-frozen -m "all selections fixed before Real-World scoring"
#   bash scripts/06_select_and_freeze.sh [checkpoints-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.spectral.freeze --alpha results/alpha/mean_alpha.csv --runs-dir "${1:-checkpoints}"
