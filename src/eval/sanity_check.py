"""Check the evaluation harness against the scores logged during training.

The harness re-scores saved checkpoints with its own code (cached uint8 images, fp32 inference, its own
macro-F1). Training logged Product validation accuracy and macro-F1 for the same checkpoints (mixed precision,
sklearn). If the two agree, the loading, preprocessing, checkpoint reading and metric are right.

    python -m src.eval.sanity_check --auto --runs-dir results/runs                # 4 checkpoints from the frozen file
    python -m src.eval.sanity_check --checkpoint ft_b100_s42:12 --runs-dir results/runs

Tolerance 0.01: training used half-precision on the GPU, the harness uses fp32, so a few borderline images can flip.
Uses Product VALIDATION images only (the same split training logged), never Real World.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.eval.metrics import accuracy, macro_f1
from src.eval.real_world import main as score


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", default=None, metavar="RUN:EPOCH")
    ap.add_argument("--auto", action="store_true",
                    help="ft_b100_s42, ft_b10_s42, lp_b100_s42, lp_b10_s42 at their validation-loss epochs in the frozen file")
    ap.add_argument("--runs-dir", default="results/runs", help="folder with <run>/metrics.csv")
    ap.add_argument("--checkpoints-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/predictions")
    ap.add_argument("--frozen", default="results/selections/frozen_selections.json")
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args(argv)

    if args.auto:
        fz = json.loads(Path(args.frozen).read_text())
        args.checkpoint = [f"{r}:{fz['selections'][r]['val_loss_epoch']}" for r in ("ft_b100_s42", "ft_b10_s42")] + \
                          [f"{r}:{fz['linear_probe'][r]['val_loss_epoch']}" for r in ("lp_b100_s42", "lp_b10_s42")]
    if not args.checkpoint:
        sys.exit("Give --checkpoint RUN:EPOCH ... or --auto")
    cli = ["--splits", "product_val", "--only", *args.checkpoint, "--out", args.out, "--frozen", args.frozen,
           "--no-tag-check"]          # Product validation only: no Real-World image is read
    if args.checkpoints_dir: cli += ["--checkpoints-dir", args.checkpoints_dir]
    if args.config: cli += ["--config", args.config]
    score(cli)

    ok = True
    print(f"\n{'checkpoint':<22}{'metric':<10}{'training log':>13}{'harness':>10}{'diff':>9}  result")
    for ck in args.checkpoint:
        run, epoch = ck.split(":"); epoch = int(epoch)
        log = pd.read_csv(Path(args.runs_dir) / run / "metrics.csv").set_index("epoch").loc[epoch]
        d = pd.read_csv(Path(args.out) / "product_val" / f"{run}_e{epoch:02d}_product_val.csv")
        for name, logged, mine in (("accuracy", log.val_acc, accuracy(d.label, d.pred)),
                                   ("macro-F1", log.val_macro_f1, macro_f1(d.label.to_numpy(), d.pred.to_numpy()))):
            good = abs(logged - mine) <= args.tol
            ok &= bool(good)
            print(f"{ck:<22}{name:<10}{logged:>13.4f}{mine:>10.4f}{mine - logged:>+9.4f}  {'PASS' if good else 'FAIL'}")
    if not ok:
        sys.exit("Sanity check FAILED: do not trust the harness until this is explained.")
    print("\nHarness agrees with the training log.")


if __name__ == "__main__":
    main()
