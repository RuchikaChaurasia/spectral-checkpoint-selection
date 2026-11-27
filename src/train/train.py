"""Train one run: linear probe or full fine-tuning of ResNet-18 on Product (owner: Disha).

Examples (from the repo root)
    # linear-probe baseline, 100% budget, seed 42
    python -m src.train.train --mode linear_probe --budget 100 --seed 42

    # fine-tuning run (after the LR pilot has chosen the learning rate)
    python -m src.train.train --mode finetune --budget 10 --seed 123 --lr 3e-4

    # continue a run that was interrupted (e.g. a Colab disconnect)
    python -m src.train.train --mode finetune --budget 10 --seed 123 --lr 3e-4 --resume

Every run trains for the full number of epochs (no early stopping), because the
checkpoint is chosen afterwards by the selection rules. Per run it writes, in
checkpoints/<run_name>/:
    epoch_00.pt ... epoch_30.pt   weights only (fp32). Epoch 0 = before any training.
                                  Linear probe: only the 65x512 head is saved.
    metrics.csv                   epoch, train_loss, train_acc, val_loss, val_acc,
                                  val_macro_f1, lr, epoch_seconds, peak_gpu_mem_mb
    run_info.json                 settings, versions, hardware, file hashes, timing,
                                  and the lowest-Product-validation-loss epoch
    last_state.pt                 resume file (deleted when the run finishes)

Only Product images are used. Real World is never loaded here.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.datasets import OfficeHomeRows, build_transforms, product_rows
from src.data.officehome import load_config, load_officehome, repo_path, sha256_of_file
from src.train.utils import (
    build_model,
    checkpoint_state,
    env_info,
    macro_f1,
    seed_everything,
    seed_worker,
)

METRIC_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1",
                 "lr", "epoch_seconds", "peak_gpu_mem_mb"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["finetune", "linear_probe"], required=True)
    ap.add_argument("--budget", type=int, choices=[10, 25, 50, 100], required=True)
    ap.add_argument("--seed", type=int, required=True, help="training seed: 42, 123 or 999")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (required for finetune; probe defaults to config linear_probe_lr)")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs (testing only)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-name", default=None, help="default: ft_b<budget>_s<seed> or lp_b<budget>_s<seed>")
    ap.add_argument("--out-root", default=None, help="default: paths.checkpoints_dir in the config")
    ap.add_argument("--no-checkpoints", action="store_true", help="log metrics only (LR pilot)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing run folder first")
    ap.add_argument("--epoch-hook", default=None,
                    help="optional 'module:function' called after each saved checkpoint, "
                         "e.g. Ruchika's WeightWatcher logger; called as fn(model=, epoch=, run_dir=, checkpoint_path=)")
    ap.add_argument("--weights", default=None, help="'none' only for offline tests")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None, help="debug: stop each epoch early")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device, amp, num_classes):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, n, ys, ps = 0.0, 0, [], []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        total += loss_fn(logits.float(), y).item()
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return total / n, float((y == p).mean()), macro_f1(y, p, num_classes)


def train_one_epoch(model, loader, optimizer, scaler, device, amp, mode, max_batches=None):
    # probe: keep the frozen backbone in eval mode so BatchNorm statistics do not change
    model.train(mode == "finetune")
    loss_fn = nn.CrossEntropyLoss()
    total, correct, n = 0.0, 0, 0
    for b, (x, y, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total / max(n, 1), correct / max(n, 1)


def lowest_val_loss_epoch(rows: list[dict]) -> int | None:
    """Validation-loss selection: lowest Product val loss over epochs 1..E, earliest on ties."""
    cand = [(float(r["val_loss"]), int(r["epoch"])) for r in rows if int(r["epoch"]) >= 1]
    return min(cand)[1] if cand else None


def load_hook(spec: str | None):
    if not spec:
        return None
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    tcfg = cfg["train"]
    num_classes = int(cfg["data"]["num_classes"])
    epochs = int(args.epochs or tcfg["epochs"])

    if args.lr is None:
        if args.mode == "finetune":
            sys.exit("--lr is required for finetune. Run the LR pilot first (scripts/run_lr_pilot.sh).")
        args.lr = float(tcfg["linear_probe_lr"])
    weights = args.weights or tcfg["pretrained_weights"]

    prefix = "ft" if args.mode == "finetune" else "lp"
    run_name = args.run_name or f"{prefix}_b{args.budget}_s{args.seed}"
    out_root = Path(args.out_root) if args.out_root else repo_path(cfg["paths"]["checkpoints_dir"])
    run_dir = out_root / run_name
    metrics_path = run_dir / "metrics.csv"
    state_path = run_dir / "last_state.pt"

    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    if metrics_path.exists() and not args.resume:
        sys.exit(f"{run_dir} already has results. Use --resume to continue or --overwrite to start again.")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(tcfg["amp"]) and device.type == "cuda"
    seed_everything(args.seed)

    # ---- data (Product only) ------------------------------------------------
    hf = load_officehome(cfg)
    train_rows = product_rows(cfg, "train", args.budget, args.seed)
    val_rows = product_rows(cfg, "val")
    train_ds = OfficeHomeRows(hf, train_rows, build_transforms(cfg, train=True))
    val_ds = OfficeHomeRows(hf, val_rows, build_transforms(cfg, train=False))
    nw = tcfg["num_workers"] if args.num_workers is None else args.num_workers
    gen = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=nw,
                              worker_init_fn=seed_worker, generator=gen, pin_memory=device.type == "cuda",
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=nw,
                            pin_memory=device.type == "cuda")

    # ---- model ----------------------------------------------------------------
    model = build_model(num_classes, args.mode, weights).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=float(tcfg["weight_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    hook = load_hook(args.epoch_hook)

    splits_dir = repo_path(cfg["paths"]["splits_dir"])
    budget_file = splits_dir / f"budgets_seed{args.seed}.csv"
    info = {
        "run_name": run_name,
        "mode": args.mode,
        "budget_percent": args.budget,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": epochs,
        "batch_size": tcfg["batch_size"],
        "optimizer": "AdamW",
        "weight_decay": tcfg["weight_decay"],
        "amp": amp,
        "pretrained_weights": weights,
        "image": cfg["image"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "trainable_params": sum(p.numel() for p in trainable),
        "total_params": sum(p.numel() for p in model.parameters()),
        "hf_revision": cfg["data"]["hf_revision"],
        "split_seed": cfg["split"]["seed"],
        "product_split_sha256": sha256_of_file(splits_dir / "product_split.csv"),
        "budget_file_sha256": sha256_of_file(budget_file) if args.budget < 100 else None,
        "checkpoints_saved": not args.no_checkpoints,
        "command": " ".join(sys.argv),
        "env": env_info(),
    }

    # ---- resume -----------------------------------------------------------------
    start_epoch, rows = 0, []
    if args.resume and not state_path.exists() and metrics_path.exists():
        prev = json.loads((run_dir / "run_info.json").read_text()) if (run_dir / "run_info.json").exists() else {}
        if prev.get("finished_at_utc"):
            print(f"{run_name} is already complete; nothing to resume.")
            return
        sys.exit(f"{run_dir} has metrics but no resume file. Use --overwrite to start this run again.")
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        start_epoch = st["epoch"] + 1
        info["started_at_utc"] = st.get("started_at_utc")
        info["resumed_at_epoch"] = start_epoch
        print(f"Resuming {run_name} from epoch {start_epoch}")
    if metrics_path.exists():
        with open(metrics_path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < start_epoch]
    if start_epoch > epochs:
        print(f"{run_name} is already complete.")
        return
    with open(metrics_path, "w", newline="") as f:  # rewrite, dropping rows past the resume point
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(rows)
    info.setdefault("started_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))

    print(f"Run {run_name}: {args.mode}, budget {args.budget}%, seed {args.seed}, lr {args.lr}, "
          f"{len(train_ds)} train / {len(val_ds)} val images, device {device}, amp {amp}")

    t_run = time.time()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if epoch == 0:
            tr_loss = tr_acc = math.nan  # epoch 0 = pretrained backbone + random head, before any update
        else:
            gen.manual_seed(args.seed * 1000 + epoch)  # same batches/augmentations even after a resume
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, amp,
                                              args.mode, args.max_train_batches)
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, amp, num_classes)
        row = {
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss,
            "val_acc": va_acc, "val_macro_f1": va_f1, "lr": args.lr,
            "epoch_seconds": round(time.time() - t0, 2),
            "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda" else None,
        }
        ckpt_path = None
        if not args.no_checkpoints:
            ckpt_path = run_dir / f"epoch_{epoch:02d}.pt"
            torch.save(checkpoint_state(model, args.mode), ckpt_path)
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow(row)
        rows.append({k: str(v) for k, v in row.items()})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch,
                    "started_at_utc": info["started_at_utc"]}, state_path)
        print(f"  epoch {epoch:2d}  train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  "
              f"val_acc {va_acc:.3f}  val_F1 {va_f1:.3f}  ({row['epoch_seconds']} s)", flush=True)
        if hook and ckpt_path is not None:
            try:
                hook(model=model, epoch=epoch, run_dir=run_dir, checkpoint_path=ckpt_path)
            except Exception:  # noqa: BLE001 - a hook failure must not kill a training run
                (run_dir / "hook_errors.log").open("a").write(f"epoch {epoch}\n{traceback.format_exc()}\n")
                print(f"  (epoch hook failed at epoch {epoch}; see hook_errors.log)")

    info["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    secs = [float(r["epoch_seconds"]) for r in rows]
    info["train_seconds_total"] = round(sum(secs), 1)
    info["seconds_this_session"] = round(time.time() - t_run, 1)
    info["lowest_val_loss_epoch"] = lowest_val_loss_epoch(rows)
    info["lowest_val_loss_rule"] = "lowest Product val loss over epochs 1..E; earliest epoch on ties"
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    state_path.unlink(missing_ok=True)
    print(f"Finished {run_name} in {info['train_seconds_total']} s. "
          f"Lowest-val-loss epoch: {info['lowest_val_loss_epoch']}")


if __name__ == "__main__":
    main()
