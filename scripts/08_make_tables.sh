#!/usr/bin/env bash
# Step 8: every table and figure of the results sections, from results/predictions/ and the frozen selections.
# Runs on CPU in about a minute. Writes results/tables/ and results/figures/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.eval.aggregate "$@"
