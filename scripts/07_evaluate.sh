#!/usr/bin/env bash
# Step 7: Real-World and Product-test evaluation of the FROZEN selections only. Needs a GPU for speed.
# Refuses to run unless the git tag `selections-frozen` exists and the selections file is unchanged since it.
#   bash scripts/07_evaluate.sh /content/drive/MyDrive/group9/checkpoints              # selected checkpoints
#   bash scripts/07_evaluate.sh /content/drive/MyDrive/group9/checkpoints --all-epochs # + every epoch (alpha vs F1)
# First it re-scores four checkpoints on Product VALIDATION and compares with the training log (sanity check).
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints}"
shift || true
python -m src.eval.sanity_check --auto --runs-dir results/runs --checkpoints-dir "$CKPT"
python -m src.eval.real_world --checkpoints-dir "$CKPT" "$@"
