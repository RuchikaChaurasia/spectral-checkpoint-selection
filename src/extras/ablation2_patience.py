"""Ablation 2: how sensitive is Rule W to its trigger patience?

Rule W (main method, patience k = 2): from epoch 1 on, look at the mean WeightWatcher alpha after every epoch.
The first time mean alpha is below 2.0 for k consecutive epochs, select the checkpoint just before the first of
those epochs (if that is epoch 1, select epoch 1). If it never happens, select the final epoch (30).
Ablation 2 compares k = 1, 2 and 3. Nothing else changes: same runs, same checkpoints, same alpha logs.

Two steps, run at different times:

  1. select  (no Real World data; can run any time, before the freeze)
     python -m src.extras.ablation2_patience select --alpha-log <alpha.csv> --out results/ablation2_selections.csv
     Input CSV columns:  run, budget, seed, epoch, mean_alpha   (one row per run and epoch; all-layer scope)
     Output: the selected epoch for every run and patience, and whether the rule fired.
     Compare this file with the team's frozen selection manifest; they must agree.

  2. score   (ONLY after the selections are frozen and Real World predictions exist)
     python -m src.extras.ablation2_patience score --selections results/ablation2_selections.csv \
            --pred-dir outputs/predictions_real_world --out results/
     Expects one prediction file per selected checkpoint: <run>_e<epoch>_real_world.csv with columns
     row_id, label, pred (same format as src/train/predict.py).
     Output: per-run macro-F1 and a budget x patience summary (mean, std, firing rate, paired difference vs k = 2).

Run the built-in checks:  python -m src.extras.ablation2_patience selftest
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 2.0
FINAL_EPOCH = 30
PATIENCES = (1, 2, 3)
MAIN_PATIENCE = 2
NUM_CLASSES = 65


def rule_w_epoch(alpha_by_epoch: dict, patience: int, threshold: float = THRESHOLD,
                 final_epoch: int = FINAL_EPOCH) -> tuple[int, bool]:
    """Return (selected_epoch, fired). Epoch 0 is ignored. A missing or NaN alpha (failed fit) breaks a streak."""
    def below(e):
        a = alpha_by_epoch.get(e)
        return a is not None and not math.isnan(a) and a < threshold

    for start in range(1, final_epoch - patience + 2):
        if all(below(start + i) for i in range(patience)):
            return (1 if start == 1 else start - 1), True
    return final_epoch, False


def select(alpha_log: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    if "scope" in alpha_log.columns:  # the log holds one row per scope; Ablation 2 uses the main scope only
        alpha_log = alpha_log[alpha_log.scope == scope]
    rows = []
    for (run, budget, seed), g in alpha_log.groupby(["run", "budget", "seed"]):
        alphas = dict(zip(g.epoch.astype(int), g.mean_alpha.astype(float)))
        for k in PATIENCES:
            epoch, fired = rule_w_epoch(alphas, k)
            rows.append({"run": run, "budget": budget, "seed": seed, "patience": k,
                         "selected_epoch": epoch, "fired": fired})
    return pd.DataFrame(rows)


def macro_f1(y_true, y_pred, num_classes: int = NUM_CLASSES) -> float:
    """Same definition as src/train/utils.py: macro over all 65 classes, 0 for classes never predicted or present."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))


def score(selections: pd.DataFrame, pred_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel = selections.copy()
    cache = {}
    for i, r in sel.iterrows():
        f = pred_dir / f"{r.run}_e{int(r.selected_epoch):02d}_real_world.csv"
        if f not in cache:
            if not f.exists():
                raise SystemExit(f"Missing predictions: {f}")
            p = pd.read_csv(f)
            cache[f] = macro_f1(p.label.to_numpy(), p.pred.to_numpy())
        sel.loc[i, "rw_macro_f1"] = cache[f]

    main = sel[sel.patience == MAIN_PATIENCE].set_index(["run"])["rw_macro_f1"]
    sel["diff_vs_k2"] = sel.rw_macro_f1 - sel.run.map(main)
    summary = (sel.groupby(["budget", "patience"])
                  .agg(n_seeds=("seed", "nunique"),
                       f1_mean=("rw_macro_f1", "mean"), f1_std=("rw_macro_f1", "std"),
                       diff_vs_k2_mean=("diff_vs_k2", "mean"),
                       epoch_mean=("selected_epoch", "mean"),
                       fired_rate=("fired", "mean"))
                  .round(4).reset_index())
    return sel, summary


def selftest():
    flat = {e: 2.5 for e in range(1, 31)}
    assert rule_w_epoch(flat, 2) == (30, False), "never fires -> final epoch"
    a = {**flat, 1: 1.9, 2: 1.9}
    assert rule_w_epoch(a, 2) == (1, True), "fires at epochs 1-2 -> epoch 1"
    b = {**flat, 10: 1.9}
    assert rule_w_epoch(b, 1) == (9, True) and rule_w_epoch(b, 2) == (30, False), "single dip: only k=1 fires"
    c = {**flat, 10: 1.9, 11: 1.9, 12: 1.9}
    assert [rule_w_epoch(c, k) for k in PATIENCES] == [(9, True), (9, True), (9, True)], "long dip: all agree"
    d = {**flat, 29: 1.9, 30: 1.9}
    assert rule_w_epoch(d, 2) == (28, True) and rule_w_epoch(d, 3) == (30, False), "dip at the very end"
    e = {**flat, 5: 1.9, 6: float("nan"), 7: 1.9}
    assert rule_w_epoch(e, 2) == (30, False), "failed fit breaks a streak"
    f = {**flat, 0: 1.0}
    assert rule_w_epoch(f, 1) == (30, False), "epoch 0 is ignored"
    g = {**flat, 4: 1.9, 8: 1.9, 9: 1.9}
    assert [rule_w_epoch(g, k) for k in PATIENCES] == [(3, True), (7, True), (30, False)], "noisy alpha"
    print("selftest: all 8 checks passed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--alpha-log", required=True); s.add_argument("--out", required=True)
    c = sub.add_parser("score"); c.add_argument("--selections", required=True)
    c.add_argument("--pred-dir", required=True); c.add_argument("--out", default="results")
    sub.add_parser("selftest")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "select":
        out = select(pd.read_csv(args.alpha_log))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(out.groupby(["budget", "patience"]).agg(epoch_mean=("selected_epoch", "mean"),
                                                      fired_rate=("fired", "mean")).round(2).to_string())
    else:
        per_run, summary = score(pd.read_csv(args.selections), Path(args.pred_dir))
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(out / "ablation2_per_run.csv", index=False)
        summary.to_csv(out / "ablation2_patience.csv", index=False)
        print(summary.to_string(index=False))
        print("\nThree seeds per cell: report each seed and the spread. No significance claims.")


if __name__ == "__main__":
    main()
