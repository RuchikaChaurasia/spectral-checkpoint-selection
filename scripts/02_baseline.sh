#!/usr/bin/env bash
# Step 2 (Disha): the runnable baseline command for the baseline checkpoint.
# Linear probe on frozen ImageNet ResNet-18, 100% budget, seed 42, then its
# Product-validation predictions for Shruti's harness check, and the loss curve.
#   bash scripts/02_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.train --mode linear_probe --budget 100 --seed 42 --resume
python -m src.train.predict --run-dir checkpoints/lp_b100_s42 --split val
python -m src.train.report_runs --runs lp_b100_s42
