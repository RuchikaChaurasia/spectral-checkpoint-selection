"""Final tables and figures from the saved predictions (evaluation harness).

    python -m src.eval.aggregate                      # reads results/predictions/ and the frozen selections

Everything here is a function of (a) results/selections/frozen_selections.json and (b) the prediction files
written by src/eval/real_world.py. Nothing is tuned. Outputs:
  results/tables/per_run.csv            one row per budget, seed and method (selected epoch, Real-World and
                                        Product-test scores, 95% bootstrap interval over Real-World images)
  results/tables/main_comparison.csv/.md   mean and SD over the three seeds, per budget and method
  results/tables/paired_differences.csv    Rule W minus validation-loss epoch, per budget and seed, with a paired
                                        bootstrap interval over Real-World images
  results/tables/ablation_scope.csv, ablation_patience.csv
  results/tables/per_class_recall_100.csv, top_confusions_100.csv   error analysis at the 100% budget
  results/figures/main_comparison.png
If predictions for all epochs exist (src/eval/real_world.py --all-epochs):
  results/tables/alpha_vs_f1.csv, results/tables/epoch_vs_f1.csv, results/figures/alpha_vs_f1.png
With three seeds we report each seed and the spread. Intervals describe resampling of test images only;
they do not include seed-to-seed variation. No significance claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import (NUM_CLASSES, accuracy, bootstrap_f1, bootstrap_indices, confusion, macro_f1,
                              per_class_recall, percentile_ci)

METHODS = ["Linear probe (val-loss epoch)", "Fine-tuning, val-loss epoch", "Fine-tuning, Rule W"]


def parse_run(run: str) -> tuple[str, int, int]:
    kind, b, s = run.split("_")
    return kind, int(b[1:]), int(s[1:])


def load_pred(root: Path, split: str, run: str, epoch: int) -> pd.DataFrame:
    path = root / split / f"{run}_e{int(epoch):02d}_{split}.csv"
    if not path.exists():
        raise SystemExit(f"Missing predictions: {path}. Run src/eval/real_world.py first.")
    return pd.read_csv(path)


def selection_table(frozen: dict) -> pd.DataFrame:
    """One row per (budget, seed, method): the selected epoch, taken only from the frozen file."""
    prim = frozen["primary"]
    rows = []
    for run, v in frozen["selections"].items():
        _, b, s = parse_run(run)
        rows.append((b, s, METHODS[1], run, int(v["val_loss_epoch"])))
        rows.append((b, s, METHODS[2], run, int(v["rule_w"][prim["scope"]][str(prim["patience"])]["epoch"])))
    for run, v in frozen.get("linear_probe", {}).items():
        _, b, s = parse_run(run)
        rows.append((b, s, METHODS[0], run, int(v["val_loss_epoch"])))
    return pd.DataFrame(rows, columns=["budget", "seed", "method", "run", "epoch"]).sort_values(
        ["budget", "method", "seed"]).reset_index(drop=True)


def score_selections(sel: pd.DataFrame, pred_root: Path, idx: np.ndarray) -> pd.DataFrame:
    out = []
    for r in sel.itertuples():
        rw = load_pred(pred_root, "real_world", r.run, r.epoch)
        y, p = rw.label.to_numpy(), rw.pred.to_numpy()
        boot = bootstrap_f1(y, p, idx)
        lo, hi = percentile_ci(boot)
        pt = load_pred(pred_root, "product_test", r.run, r.epoch)
        out.append({"budget": r.budget, "seed": r.seed, "method": r.method, "run": r.run, "epoch": r.epoch,
                    "rw_macro_f1": macro_f1(y, p), "rw_ci_lo": lo, "rw_ci_hi": hi, "rw_accuracy": accuracy(y, p),
                    "product_test_macro_f1": macro_f1(pt.label.to_numpy(), pt.pred.to_numpy())})
    return pd.DataFrame(out)


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    g = per_run.groupby(["budget", "method"])
    s = g.agg(n_seeds=("seed", "nunique"), rw_f1_mean=("rw_macro_f1", "mean"), rw_f1_sd=("rw_macro_f1", "std"),
              rw_acc_mean=("rw_accuracy", "mean"), ptest_f1_mean=("product_test_macro_f1", "mean"),
              ptest_f1_sd=("product_test_macro_f1", "std"), epoch_mean=("epoch", "mean")).reset_index()
    s["method"] = pd.Categorical(s.method, METHODS, ordered=True)
    return s.sort_values(["budget", "method"]).reset_index(drop=True)


def paired_differences(per_run: pd.DataFrame, pred_root: Path, idx: np.ndarray) -> pd.DataFrame:
    """Rule W minus validation-loss epoch (and validation-loss minus linear probe), same bootstrap resamples."""
    def f1s(row):
        d = load_pred(pred_root, "real_world", row.run, row.epoch)
        return d.label.to_numpy(), d.pred.to_numpy()
    rows = []
    pairs = [(METHODS[2], METHODS[1], "Rule W - val-loss (both fine-tuning)"),
             (METHODS[1], METHODS[0], "Fine-tuning val-loss - linear probe")]
    for b in sorted(per_run.budget.unique()):
        for s in sorted(per_run.seed.unique()):
            for a_name, b_name, label in pairs:
                a = per_run[(per_run.budget == b) & (per_run.seed == s) & (per_run.method == a_name)]
                c = per_run[(per_run.budget == b) & (per_run.seed == s) & (per_run.method == b_name)]
                if a.empty or c.empty:
                    continue
                a, c = a.iloc[0], c.iloc[0]
                ya, pa = f1s(a); yc, pc = f1s(c)
                boot = bootstrap_f1(ya, pa, idx) - bootstrap_f1(yc, pc, idx)
                lo, hi = percentile_ci(boot)
                rows.append({"comparison": label, "budget": b, "seed": s, "epoch_a": a.epoch, "epoch_b": c.epoch,
                             "diff_rw_macro_f1": a.rw_macro_f1 - c.rw_macro_f1, "ci_lo": lo, "ci_hi": hi,
                             "same_checkpoint": bool(a.run == c.run and a.epoch == c.epoch)})
    return pd.DataFrame(rows)


def ablation_tables(frozen: dict, pred_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prim = frozen["primary"]
    rows = []
    for run, v in frozen["selections"].items():
        _, b, s = parse_run(run)
        rw_cache = {}
        for scope, by_k in v["rule_w"].items():
            for k, sel in by_k.items():
                if not isinstance(sel, dict) or "epoch" not in sel:
                    continue
                e = int(sel["epoch"])
                if e not in rw_cache:
                    d = load_pred(pred_root, "real_world", run, e)
                    rw_cache[e] = macro_f1(d.label.to_numpy(), d.pred.to_numpy())
                rows.append({"scope": scope, "patience": int(k), "budget": b, "seed": s, "epoch": e,
                             "fired": bool(sel["fired"]), "rw_macro_f1": rw_cache[e]})
    df = pd.DataFrame(rows)
    agg = lambda g: g.agg(n_runs=("seed", "size"), fired_rate=("fired", "mean"), epoch_mean=("epoch", "mean"),
                          rw_f1_mean=("rw_macro_f1", "mean"), rw_f1_sd=("rw_macro_f1", "std")).reset_index()
    scope = agg(df[df.patience == prim["patience"]].groupby(["scope", "budget"]))
    patience = agg(df[df.scope == prim["scope"]].groupby(["patience", "budget"]))
    return scope, patience


def error_analysis(per_run: pd.DataFrame, pred_root: Path, class_names: dict[int, str], budget: int = 100):
    recall, conf = {}, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for m in METHODS:
        r = per_run[(per_run.budget == budget) & (per_run.method == m)]
        recalls = []
        for x in r.itertuples():
            d = load_pred(pred_root, "real_world", x.run, x.epoch)
            recalls.append(per_class_recall(d.label.to_numpy(), d.pred.to_numpy()))
            if m == METHODS[1]:
                conf += confusion(d.label.to_numpy().astype(np.int64), d.pred.to_numpy().astype(np.int64))
        recall[m] = np.nanmean(recalls, axis=0)
    rec = pd.DataFrame(recall)
    rec.insert(0, "class", [class_names[i] for i in range(NUM_CLASSES)])
    off = conf.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:15], off.shape))[0]
    tc = pd.DataFrame([{"true_class": class_names[i], "predicted_as": class_names[j], "count_over_3_seeds": int(off[i, j]),
                        "share_of_true_class": round(off[i, j] / conf[i].sum(), 3)} for i, j in top])
    return rec, tc


def _rw_path(pred_root: Path, extra_root: Path, run: str, e: int) -> Path:
    name = f"{run}_e{e:02d}_real_world.csv"
    return pred_root / "real_world" / name if (pred_root / "real_world" / name).exists() else extra_root / "real_world" / name


def descriptive_all_epochs(frozen: dict, pred_root: Path, alpha_csv: Path, scope: str, extra_root: Path):
    """alpha against Real-World macro-F1 over every epoch of the fine-tuning runs (descriptive only)."""
    files = [_rw_path(pred_root, extra_root, run, e) for run in frozen["selections"] for e in range(1, 31)]
    if not alpha_csv.exists() or not all(f.exists() for f in files):
        return None            # predictions for every epoch were not made (src/eval/real_world.py --all-epochs)
    alpha = pd.read_csv(alpha_csv)
    alpha = alpha[alpha.scope == scope]
    rows = []
    for run in frozen["selections"]:
        for e in range(1, 31):
            d = pd.read_csv(_rw_path(pred_root, extra_root, run, e))
            a = alpha[(alpha.run == run) & (alpha.epoch == e)].mean_alpha.iloc[0]
            _, b, s = parse_run(run)
            rows.append({"run": run, "budget": b, "seed": s, "epoch": e, "mean_alpha": a,
                         "rw_macro_f1": macro_f1(d.label.to_numpy(), d.pred.to_numpy())})
    df = pd.DataFrame(rows)
    sp = df.groupby("run").apply(lambda g: g.mean_alpha.corr(g.rw_macro_f1, method="spearman"), include_groups=False)
    df["spearman_within_run"] = df.run.map(sp)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", default="results/selections/frozen_selections.json")
    ap.add_argument("--predictions", default="results/predictions")
    ap.add_argument("--extra-predictions", default="outputs/predictions_all_epochs")
    ap.add_argument("--alpha", default="results/alpha/mean_alpha.csv")
    ap.add_argument("--target", default="configs/splits/target_real_world.csv", help="row_id, label, class")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=1000, help="bootstrap resamples")
    args = ap.parse_args(argv)

    out = Path(args.out); (out / "tables").mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.frozen).read_text())
    pred_root = Path(args.predictions)
    tgt = pd.read_csv(args.target)
    names = dict(zip(tgt.label, tgt["class"]))

    sel = selection_table(frozen)
    n_rw = len(load_pred(pred_root, "real_world", sel.run.iloc[0], sel.epoch.iloc[0]))
    idx = bootstrap_indices(n_rw, args.reps, seed=0)           # one set of resamples shared by every comparison
    per_run = score_selections(sel, pred_root, idx)
    per_run.to_csv(out / "tables/per_run.csv", index=False)
    summ = summarize(per_run); summ.to_csv(out / "tables/main_comparison.csv", index=False)
    with open(out / "tables/main_comparison.md", "w") as f:
        f.write("| Budget | Method | Seeds | Real-World macro-F1 (mean ± SD) | Product-test macro-F1 (mean ± SD) | Mean selected epoch |\n|---|---|---|---|---|---|\n")
        for r in summ.itertuples():
            f.write(f"| {r.budget}% | {r.method} | {r.n_seeds} | {r.rw_f1_mean:.3f} ± {r.rw_f1_sd:.3f} | "
                    f"{r.ptest_f1_mean:.3f} ± {r.ptest_f1_sd:.3f} | {r.epoch_mean:.1f} |\n")
    pd_ = paired_differences(per_run, pred_root, idx); pd_.to_csv(out / "tables/paired_differences.csv", index=False)
    a1, a2 = ablation_tables(frozen, pred_root)
    a1.to_csv(out / "tables/ablation_scope.csv", index=False); a2.to_csv(out / "tables/ablation_patience.csv", index=False)
    if 100 in set(per_run.budget):
        rec, top = error_analysis(per_run, pred_root, names, 100)
        rec.to_csv(out / "tables/per_class_recall_100.csv", index=False); top.to_csv(out / "tables/top_confusions_100.csv", index=False)

    make_main_figure(per_run, out / "figures/main_comparison.png")
    desc = descriptive_all_epochs(frozen, pred_root, Path(args.alpha), frozen["primary"]["scope"], Path(args.extra_predictions))
    if desc is not None:
        desc.to_csv(out / "tables/alpha_vs_f1.csv", index=False)
        make_alpha_figure(desc, out / "figures/alpha_vs_f1.png")
    print((out / "tables/main_comparison.md").read_text())
    print("Paired differences, Rule W minus val-loss (identical checkpoint = difference exactly 0):")
    print(pd_[pd_.comparison.str.startswith("Rule W")].round(4).to_string(index=False))


def make_main_figure(per_run: pd.DataFrame, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ["#00b894", "#257de0", "#f47230"]))
    budgets = sorted(per_run.budget.unique())
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    for j, m in enumerate(METHODS):
        d = per_run[per_run.method == m]
        if d.empty:
            continue
        off = (j - 1) * 0.9
        x = np.array([budgets.index(b) * 10 + off for b in d.budget])
        ax.plot(x, d.rw_macro_f1, "o", color=colors[m], ms=6, mec="white", mew=1.0, alpha=0.9, zorder=3)
        mean = d.groupby("budget").rw_macro_f1.mean()
        ax.plot([budgets.index(b) * 10 + off for b in mean.index], mean.values, "-", color=colors[m], lw=1.8,
                label=m, zorder=2)
    ax.set_xticks([i * 10 for i in range(len(budgets))]); ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_xlabel("share of Product training labels"); ax.set_ylabel("Real-World macro-F1")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Real-World macro-F1 by label budget (dots = seeds, lines = means)", fontsize=10.5, loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def make_alpha_figure(desc: pd.DataFrame, path: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=200)
    ax.scatter(desc.mean_alpha, desc.rw_macro_f1, s=10, color="#257de0", alpha=0.35, edgecolors="none")
    rho = desc.groupby("run").spearman_within_run.first()
    ax.set_xlabel("mean WeightWatcher alpha (all layers)"); ax.set_ylabel("Real-World macro-F1")
    ax.set_title(f"All 360 fine-tuning checkpoints (epochs 1-30): within-run Spearman rho {rho.min():.2f} to {rho.max():.2f}",
                 fontsize=9.5, loc="left", fontweight="bold")
    ax.axvline(2.0, color="#444444", lw=1.2, ls=(0, (4, 3)))
    ax.grid(color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
