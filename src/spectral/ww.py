"""WeightWatcher alpha logging for saved fine-tuning checkpoints (owner: Ruchika).

What it does
  For every checkpoints/<run>/epoch_XX.pt of a fine-tuning run it
    1. rebuilds the ResNet-18 (65-way head) and loads the saved weights,
    2. runs WeightWatcher with its default settings,
    3. keeps one row per analysable layer: name, shape, alpha, fit status, warning,
    4. records how long the analysis took.
  A layer whose fit fails (status other than "success", or alpha missing / not finite)
  is kept in the log, marked ok=False, and left out of every average.

Outputs (small files, committed)
  results/alpha/<run>_layers.csv   one row per run, epoch and layer
  results/alpha/mean_alpha.csv     one row per run, epoch and scope: mean alpha, layers used, layers failed

Run from the repo root:
  python -m src.spectral.ww --checkpoints-dir checkpoints --out results/alpha
  python -m src.spectral.ww --checkpoints-dir checkpoints --runs ft_b100_s42     # one run
Re-running skips epochs that are already in the log, so an interrupted run continues.

Needs only torch, torchvision, pandas and weightwatcher; it does not import the training code.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18

NUM_CLASSES = 65
CKPT_RE = re.compile(r"epoch_(\d+)\.pt$")
RUN_RE = re.compile(r"^ft_b(\d+)_s(\d+)$")
KEEP = ["layer_id", "longname", "layer_type", "N", "M", "alpha", "status", "warning",
        "num_pl_spikes", "xmin", "xmax", "sigma", "stable_rank", "spectral_norm"]

# Layer scopes used by Rule W and ablation 1. A scope is a test on the layer name.
SCOPES = {
    "all": lambda name: True,
    "last_stage": lambda name: name.startswith("layer4") or name == "fc",
}


def load_model(path: str | Path, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Load a fine-tuning checkpoint (full state dict) into a fresh ResNet-18."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        raise SystemExit(f"{path} is not a full fine-tuning checkpoint (linear-probe checkpoints "
                         f"hold only the head and are not analysed): {err}")
    return model.eval()


def analyze_model(model: nn.Module) -> pd.DataFrame:
    """WeightWatcher with library defaults; one row per analysable layer."""
    import weightwatcher as ww

    df = ww.WeightWatcher(model=model).analyze()
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["ok"] = df["status"].astype(str).eq("success") & np.isfinite(df["alpha"])
    return df


def mean_alpha_table(layers: pd.DataFrame) -> pd.DataFrame:
    """Mean alpha per run, epoch and scope, over layers whose fit succeeded."""
    rows = []
    for (run, epoch), g in layers.groupby(["run", "epoch"]):
        for scope, in_scope in SCOPES.items():
            s = g[g.longname.map(in_scope)]
            ok = s[s.ok]
            rows.append({"run": run, "epoch": int(epoch), "scope": scope,
                         "mean_alpha": float(ok.alpha.mean()) if len(ok) else float("nan"),
                         "n_layers_used": int(len(ok)), "n_layers_failed": int(len(s) - len(ok))})
    out = pd.DataFrame(rows)
    parsed = out.run.str.extract(RUN_RE)
    out.insert(1, "budget", parsed[0].astype(float).astype("Int64"))
    out.insert(2, "seed", parsed[1].astype(float).astype("Int64"))
    return out.sort_values(["run", "scope", "epoch"]).reset_index(drop=True)


def analyze_run(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Analyse every epoch_XX.pt in run_dir; resume from an existing log."""
    run = run_dir.name
    log_path = out_dir / f"{run}_layers.csv"
    done = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    done_epochs = set(done.epoch.unique()) if len(done) else set()
    ckpts = sorted((int(CKPT_RE.search(p.name).group(1)), p) for p in run_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise SystemExit(f"No epoch_XX.pt files in {run_dir}")
    new = []
    for epoch, path in ckpts:
        if epoch in done_epochs:
            continue
        t0 = time.time()
        df = analyze_model(load_model(path))
        df.insert(0, "run", run)
        df.insert(1, "epoch", epoch)
        df["analysis_seconds"] = round(time.time() - t0, 2)  # whole-model time, repeated on each row
        new.append(df)
        print(f"{run} epoch {epoch:02d}: {int(df.ok.sum())}/{len(df)} layers ok, "
              f"mean alpha {df[df.ok].alpha.mean():.3f}, {df.analysis_seconds.iloc[0]:.1f}s", flush=True)
        # save after every epoch so a disconnect loses at most one checkpoint
        pd.concat([done, *new]).to_csv(log_path, index=False)
    return pd.concat([done, *new]) if new else done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints-dir", default="checkpoints")
    ap.add_argument("--out", default="results/alpha")
    ap.add_argument("--runs", nargs="+", default=None, help="default: every ft_b*_s* folder")
    args = ap.parse_args(argv)

    root, out = Path(args.checkpoints_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = args.runs or sorted(p.name for p in root.iterdir() if p.is_dir() and RUN_RE.match(p.name))
    if not runs:
        raise SystemExit(f"No ft_b*_s* run folders in {root}")
    for run in runs:
        analyze_run(root / run, out)

    layers = pd.concat([pd.read_csv(out / f"{r}_layers.csv") for r in sorted(
        p.name[:-len("_layers.csv")] for p in out.glob("ft_b*_s*_layers.csv"))])
    table = mean_alpha_table(layers)
    table.to_csv(out / "mean_alpha.csv", index=False)
    print(f"\nWrote {out/'mean_alpha.csv'}")
    print(table[table.scope == "all"].groupby("run").mean_alpha.agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
