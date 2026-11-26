"""The two checkpoint-selection rules (owner: Ruchika).

Strong baseline: the epoch with the lowest Product validation loss.
Rule W (pre-specified in the proposal): look at the mean WeightWatcher alpha after every epoch
from epoch 1 on. The first time it is below the threshold (2.0) for `patience` consecutive
epochs, select the checkpoint just before the first of those epochs (epoch 1 if they start at
epoch 1, because epoch 0 is not used). If that never happens, select the final epoch (30).
A missing or non-finite mean alpha (every fit failed) cannot be below the threshold.

Both functions use only Product data and weight statistics. No Real-World image is involved.
"""
from __future__ import annotations

import math

import pandas as pd


def lowest_val_loss_epoch(metrics: pd.DataFrame) -> int:
    """Lowest Product validation loss over epochs >= 1; the earliest epoch wins a tie.

    Same rule as `lowest_val_loss_epoch` in src/train/train.py (written to run_info.json).
    """
    m = metrics[metrics.epoch >= 1].sort_values("epoch")
    return int(m.loc[m.val_loss.idxmin(), "epoch"])  # idxmin returns the first minimum


def rule_w(alpha_by_epoch: dict[int, float], patience: int = 2, threshold: float = 2.0,
           final_epoch: int = 30, first_epoch: int = 1) -> tuple[int, bool]:
    """Return (selected_epoch, fired)."""
    def below(epoch: int) -> bool:
        a = alpha_by_epoch.get(epoch)
        return a is not None and math.isfinite(a) and a < threshold

    for first in range(first_epoch, final_epoch - patience + 2):
        if all(below(first + i) for i in range(patience)):
            return max(first - 1, first_epoch), True
    return final_epoch, False
