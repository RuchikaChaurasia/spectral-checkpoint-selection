"""Pick the fine-tuning learning rate from the pilot runs (owner: Disha).

Rule (fixed before any Real World image is scored): the candidate whose lowest
Product validation loss over epochs 1..E is smallest wins.

    python -m src.train.select_lr            # reads checkpoints/pilot/*/metrics.csv
Writes configs/lr_choice.json, which scripts/run_all.sh reads.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.data.officehome import load_config, repo_path
from src.train.run_all import pilot_run_name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pilot-dir", default=None, help="default: <checkpoints_dir>/pilot")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    t = cfg["train"]
    pilot_dir = repo_path(args.pilot_dir) if args.pilot_dir else repo_path(cfg["paths"]["checkpoints_dir"]) / "pilot"
    results = {}
    for lr in t["lr_candidates"]:
        m = pd.read_csv(pilot_dir / pilot_run_name(cfg, lr) / "metrics.csv")
        m = m[m.epoch >= 1]
        best = m.loc[m.val_loss.idxmin()]
        results[f"{lr:g}"] = {"lr": lr, "min_val_loss": float(best.val_loss), "at_epoch": int(best.epoch),
                              "val_acc_at_min": float(best.val_acc), "epochs_logged": int(m.epoch.max())}
    chosen = min(results.values(), key=lambda r: r["min_val_loss"])
    out = {
        "chosen_lr": chosen["lr"],
        "rule": "smallest minimum Product validation loss over epochs 1..E (pilot seed, pilot budget)",
        "pilot_seed": t["pilot_seed"],
        "pilot_budget": t["pilot_budget"],
        "candidates": results,
        "chosen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = repo_path(cfg["paths"]["lr_choice"])
    path.write_text(json.dumps(out, indent=2))
    for k, r in results.items():
        print(f"  lr {k}: min val loss {r['min_val_loss']:.4f} at epoch {r['at_epoch']}")
    print(f"Chosen learning rate: {chosen['lr']:g} -> {path}")


if __name__ == "__main__":
    main()
