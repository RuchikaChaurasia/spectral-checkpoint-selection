"""Loss curves and the training-time table from finished runs (owner: Disha).

    python -m src.train.report_runs                      # every run in checkpoints/
    python -m src.train.report_runs --runs ft_b100_s42   # just one (progress-report curve)

Writes
  results/figures/loss_curve_<run>.png      train and validation loss per epoch
  results/figures/loss_curves_finetune.png  all fine-tuning runs, one panel per budget
  results/training_time_summary.csv         time, memory, sizes and the lowest-val-loss epoch per run
The curves use Product data only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.officehome import load_config, repo_path  # noqa: E402


def load_runs(root: Path, names: list[str] | None):
    runs = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "metrics.csv").exists()):
        if names and d.name not in names:
            continue
        info = json.loads((d / "run_info.json").read_text()) if (d / "run_info.json").exists() else {}
        runs[d.name] = (pd.read_csv(d / "metrics.csv"), info)
    return runs


def plot_one(name, m, info, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    t = m[m.epoch >= 1]
    ax.plot(t.epoch, t.train_loss, "o-", ms=3, label="train loss")
    ax.plot(m.epoch, m.val_loss, "s-", ms=3, label="validation loss (Product)")
    best = info.get("lowest_val_loss_epoch")
    if best is not None:
        ax.axvline(best, color="grey", ls="--", lw=1, label=f"lowest val loss (epoch {best})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    mode = "fine-tuning" if info.get("mode") == "finetune" else "linear probe"
    ax.set_title(f"{name}: {mode}, {info.get('budget_percent', '?')}% labels, seed {info.get('seed', '?')}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_grid(runs, out):
    ft = {k: v for k, v in runs.items() if v[1].get("mode") == "finetune" and "_lr" not in k}
    budgets = sorted({v[1].get("budget_percent") for v in ft.values()})
    if not budgets:
        return
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.6), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], budgets):
        for name, (m, info) in ft.items():
            if info.get("budget_percent") == b:
                ax.plot(m.epoch, m.val_loss, label=f"seed {info.get('seed')}")
        ax.set_title(f"{b}% labels: Product val loss", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("validation loss")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="default: paths.checkpoints_dir")
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    root = Path(args.root) if args.root else repo_path(cfg["paths"]["checkpoints_dir"])
    fig_dir = repo_path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root, args.runs)
    if not runs:
        raise SystemExit(f"No runs with metrics.csv under {root}")

    rows = []
    for name, (m, info) in runs.items():
        plot_one(name, m, info, fig_dir / f"loss_curve_{name}.png")
        t = m[m.epoch >= 1]
        rows.append({
            "run_name": name,
            "mode": info.get("mode"),
            "budget_percent": info.get("budget_percent"),
            "seed": info.get("seed"),
            "lr": info.get("lr"),
            "n_train": info.get("n_train"),
            "trainable_params": info.get("trainable_params"),
            "epochs_done": int(m.epoch.max()),
            "train_seconds_total": info.get("train_seconds_total", round(m.epoch_seconds.sum(), 1)),
            "mean_epoch_seconds": round(t.epoch_seconds.mean(), 2) if len(t) else None,
            "peak_gpu_mem_mb": m.peak_gpu_mem_mb.max() if "peak_gpu_mem_mb" in m else None,
            "gpu": (info.get("env") or {}).get("gpu"),
            "lowest_val_loss_epoch": info.get("lowest_val_loss_epoch"),
            "min_val_loss": round(t.val_loss.min(), 4) if len(t) else None,
            "finished": bool(info.get("finished_at_utc")),
        })
    plot_grid(runs, fig_dir / "loss_curves_finetune.png")
    out = repo_path(cfg["paths"]["results_dir"]) / "training_time_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["mode", "budget_percent", "seed"]).to_csv(out, index=False)
    print(f"{len(runs)} runs -> {fig_dir} and {out}")


if __name__ == "__main__":
    main()
