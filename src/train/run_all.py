"""Launch the LR pilot or all 24 main runs, one subprocess per run (owner: Disha).

    python -m src.train.run_all --pilot          # 2 fine-tuning runs: lr 1e-4 and 3e-4 (seed 42, 100%)
    python -m src.train.select_lr                # writes configs/lr_choice.json
    python -m src.train.run_all                  # 12 fine-tuning + 12 linear-probe runs
    python -m src.train.run_all --budgets 100    # only the minimum viable project

Order follows the work plan: budget 100 first, then 10, 25, 50, so the runs that
must never be dropped finish first. Each run is started with --resume, so
re-running this command after a crash or disconnect skips finished runs and
continues an interrupted one where it stopped.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from src.data.officehome import load_config, repo_path

RUN_ORDER = [100, 10, 25, 50]


def pilot_run_name(cfg, lr: float) -> str:
    t = cfg["train"]
    return f"ft_b{t['pilot_budget']}_s{t['pilot_seed']}_lr{lr:g}"


def run(cmd: list[str], dry: bool) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="run the learning-rate pilot instead")
    ap.add_argument("--modes", nargs="+", default=["finetune", "linear_probe"],
                    choices=["finetune", "linear_probe"])
    ap.add_argument("--budgets", nargs="+", type=int, default=None, help="default: 100 10 25 50")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default: config budgets.seeds")
    ap.add_argument("--lr", type=float, default=None, help="override configs/lr_choice.json")
    ap.add_argument("--epoch-hook", default=None, help="passed through to every fine-tuning run")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--config", default=None)
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --, extra args for train.py")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    base = [sys.executable, "-m", "src.train.train"]
    if args.config:
        base += ["--config", args.config]
    extra = [a for a in args.extra if a != "--"]
    failed, t0 = [], time.time()

    if args.pilot:
        t = cfg["train"]
        out_root = str(repo_path(cfg["paths"]["checkpoints_dir"]) / "pilot")
        for lr in t["lr_candidates"]:
            cmd = base + ["--mode", "finetune", "--budget", str(t["pilot_budget"]), "--seed", str(t["pilot_seed"]),
                          "--lr", f"{lr:g}", "--run-name", pilot_run_name(cfg, lr), "--out-root", out_root,
                          "--no-checkpoints", "--resume"] + extra
            if run(cmd, args.dry_run):
                failed.append(pilot_run_name(cfg, lr))
    else:
        lr = args.lr
        if lr is None and "finetune" in args.modes:
            choice = repo_path(cfg["paths"]["lr_choice"])
            if not choice.exists():
                sys.exit("configs/lr_choice.json not found: run the pilot and select_lr first, or pass --lr.")
            lr = json.loads(choice.read_text())["chosen_lr"]
            print(f"Fine-tuning learning rate from configs/lr_choice.json: {lr:g}")
        budgets = args.budgets or RUN_ORDER
        seeds = args.seeds or cfg["budgets"]["seeds"]
        for budget in budgets:
            for mode in args.modes:
                for seed in seeds:
                    cmd = base + ["--mode", mode, "--budget", str(budget), "--seed", str(seed), "--resume"]
                    if mode == "finetune":
                        cmd += ["--lr", f"{lr:g}"]
                        if args.epoch_hook:
                            cmd += ["--epoch-hook", args.epoch_hook]
                    if run(cmd + extra, args.dry_run):
                        failed.append(f"{mode} b{budget} s{seed}")

    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
