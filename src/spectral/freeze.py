"""Compute every selection and write them to one file BEFORE any Real-World scoring (owner: Ruchika).

    python -m src.spectral.freeze --alpha results/alpha/mean_alpha.csv --runs-dir checkpoints

Writes results/selections/frozen_selections.json with, for each fine-tuning run:
  - the validation-loss epoch (strong baseline),
  - Rule W for both layer scopes (ablation 1) and patience 1, 2, 3 (ablation 2); the primary
    rule is scope "all", patience 2 (see configs/spectral.yaml),
and, for each linear-probe run, its lowest-validation-loss epoch (reference baseline).
Then commit the file and tag the commit:  git tag -a selections-frozen -m "..."  (see CONTRIBUTING.md).
Nothing here reads a Real-World prediction or label. The script refuses to overwrite an existing
file unless --force is given; after the tag exists, do not use --force (log a deviation instead).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.spectral.rule_w import lowest_val_loss_epoch, rule_w

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_selections(alpha: pd.DataFrame, runs_dir: Path, cfg: dict) -> dict:
    rw, abl, runs_cfg = cfg["rule_w"], cfg["ablations"], cfg["runs"]
    expected = [f"ft_b{b}_s{s}" for b in runs_cfg["budgets"] for s in runs_cfg["seeds"]]
    missing = [r for r in expected if r not in set(alpha.run)]
    if missing:
        sys.exit(f"Alpha log is missing runs: {missing}. Run src/spectral/ww.py on all 12 fine-tuning runs first.")
    need = set(range(rw["first_epoch"], rw["final_epoch"] + 1))
    out = {}
    for run in expected:
        entry = {}
        for scope in abl["scopes"]:
            g = alpha[(alpha.run == run) & (alpha.scope == scope)]
            have = set(g.epoch.astype(int))
            if not need <= have:
                sys.exit(f"{run} ({scope}): alpha missing for epochs {sorted(need - have)}")
            series = {int(e): float(a) for e, a in zip(g.epoch, g.mean_alpha)}
            entry[scope] = {str(k): dict(zip(("epoch", "fired"), rule_w(
                series, patience=k, threshold=rw["threshold"], final_epoch=rw["final_epoch"],
                first_epoch=rw["first_epoch"]))) for k in abl["patiences"]}
            entry[scope]["min_mean_alpha"] = round(min(series[e] for e in need), 4)
        metrics = pd.read_csv(runs_dir / run / "metrics.csv")
        out[run] = {"val_loss_epoch": lowest_val_loss_epoch(metrics), "rule_w": entry}
    return out


def build_linear_probe(runs_dir: Path, cfg: dict) -> dict:
    """Reference baseline: the linear probe at its lowest Product validation-loss epoch (same rule as above)."""
    runs_cfg = cfg["runs"]
    out = {}
    for b in runs_cfg["budgets"]:
        for s in runs_cfg["seeds"]:
            run = f"lp_b{b}_s{s}"
            path = runs_dir / run / "metrics.csv"
            if not path.exists():
                sys.exit(f"{path} not found: the linear-probe runs are needed for the reference baseline.")
            out[run] = {"val_loss_epoch": lowest_val_loss_epoch(pd.read_csv(path))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", default="results/alpha/mean_alpha.csv")
    ap.add_argument("--runs-dir", default="checkpoints", help="folder with <run>/metrics.csv")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "spectral.yaml"))
    ap.add_argument("--out", default="results/selections/frozen_selections.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config))
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists. Selections are frozen once; use --force only before the tag exists.")
    alpha_path = Path(args.alpha)
    sel = build_selections(pd.read_csv(alpha_path), Path(args.runs_dir), cfg)

    primary = cfg["rule_w"]["primary"]
    doc = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Computed from Product validation loss and WeightWatcher alpha only. No Real-World data used.",
        "rule_w": {k: cfg["rule_w"][k] for k in ("threshold", "final_epoch", "first_epoch")},
        "primary": {"scope": primary["scope"], "patience": primary["patience"]},
        "alpha_log": str(alpha_path), "alpha_log_sha256": sha256(alpha_path),
        "selections": sel,
        "linear_probe": build_linear_probe(Path(args.runs_dir), cfg),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))

    # summary for the team
    print(f"Wrote {out_path}  ({len(sel)} runs)\n")
    print(f"{'variant':<24}{'fired in':>10}   selected epochs")
    for scope in cfg["ablations"]["scopes"]:
        for k in cfg["ablations"]["patiences"]:
            sels = [(r, v["rule_w"][scope][str(k)]) for r, v in sel.items()]
            n = sum(s["fired"] for _, s in sels)
            print(f"{scope + ', patience ' + str(k):<24}{str(n) + '/' + str(len(sels)):>10}   "
                  f"{sorted({s['epoch'] for _, s in sels})}")
    prim = [v["rule_w"][primary["scope"]][str(primary["patience"])]["fired"] for v in sel.values()]
    if not any(prim):
        print("\nWARNING: the primary rule never fired, so Rule W selects epoch 30 for every run. "
              "Report this honestly; see docs/decisions.md before any extra rule is added.")


if __name__ == "__main__":
    main()
