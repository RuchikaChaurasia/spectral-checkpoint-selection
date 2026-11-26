"""Score the frozen selections on Real World and Product test (evaluation harness).

Run ONLY after results/selections/frozen_selections.json is committed and tagged `selections-frozen`
(the script checks the tag and that the file is unchanged since the tag). It scores exactly the
checkpoints that file names; it never chooses a checkpoint.

    python -m src.eval.real_world                         # the selected checkpoints only
    python -m src.eval.real_world --all-epochs            # also every epoch of the 12 fine-tuning runs
                                                          # (descriptive alpha-vs-F1 analysis)
Writes, for every scored (run, epoch):
    results/predictions/real_world/<run>_e<EE>_real_world.csv      row_id, label, pred, confidence
    results/predictions/product_test/<run>_e<EE>_product_test.csv  (in-domain reference)
(--splits product_val --only RUN:EPOCH is used by src/eval/sanity_check.py to compare this harness with the
validation scores logged during training.)
With --all-epochs the predictions of the non-selected epochs go to outputs/predictions_all_epochs/ (not committed).
Files that already exist are skipped, so an interrupted run continues.

Images are decoded once (Resize 256, CenterCrop 224, stored as uint8; the same eval preprocessing as
training) and cached in outputs/cache/, then every checkpoint is run on the cached tensors. Normalisation
is applied on the fly with the config's mean and std, which equals ToTensor + Normalize.
Needs a GPU for speed (a few seconds per checkpoint); CPU works but is slow.

Reads Real-World images through this script only; src/data/datasets.py cannot load them by design.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T

from src.data.officehome import load_config, load_officehome, repo_path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "selections-frozen"


def mode_of(run: str) -> str:
    return "finetune" if run.startswith("ft_") else "linear_probe"


def needed_checkpoints(frozen: dict, all_epochs: bool = False) -> list[tuple[str, int]]:
    """Every (run, epoch) named by the frozen file (plus all epochs of fine-tuning runs if asked)."""
    need: set[tuple[str, int]] = set()
    for run, v in frozen["selections"].items():
        need.add((run, int(v["val_loss_epoch"])))
        for scope, by_patience in v["rule_w"].items():
            for k, sel in by_patience.items():
                if isinstance(sel, dict) and "epoch" in sel:
                    need.add((run, int(sel["epoch"])))
        if all_epochs:
            need.update((run, e) for e in range(0, 31))
    for run, v in frozen.get("linear_probe", {}).items():
        need.add((run, int(v["val_loss_epoch"])))
    return sorted(need)


def check_frozen(path: Path) -> None:
    """Stop unless the freeze tag exists and the selections file is unchanged since it."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True)
    if git("rev-parse", "--verify", "-q", f"refs/tags/{TAG}").returncode != 0:
        sys.exit(f"Git tag '{TAG}' not found. Commit {path.name}, tag the commit, then run this again "
                 f"(see CONTRIBUTING.md). Use --no-tag-check only for offline tests.")
    rel = path.resolve().relative_to(REPO_ROOT)
    if git("diff", "--quiet", TAG, "--", str(rel)).returncode != 0:
        sys.exit(f"{rel} differs from the version tagged '{TAG}'. The selections must not change after the freeze.")


def cache_images(ds, row_ids: np.ndarray, cache: Path, size: int, resize: int) -> torch.Tensor:
    """uint8 tensor [N, 3, size, size] in the order of row_ids; decoded once and cached on disk."""
    if cache.exists():
        saved = torch.load(cache, weights_only=True)
        if torch.equal(saved["row_ids"], torch.as_tensor(row_ids)):
            return saved["images"]
    tf = T.Compose([T.Resize(resize), T.CenterCrop(size), T.PILToTensor()])
    out = torch.empty((len(row_ids), 3, size, size), dtype=torch.uint8)
    for i, r in enumerate(row_ids):
        out[i] = tf(ds[int(r)]["image"].convert("RGB"))
        if (i + 1) % 1000 == 0:
            print(f"  decoded {i + 1}/{len(row_ids)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"row_ids": torch.as_tensor(row_ids), "images": out}, cache)
    return out


@torch.no_grad()
def predict(model, images: torch.Tensor, mean, std, device, batch: int = 256):
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    preds, confs = [], []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device).float().div(255)
        prob = torch.softmax(model((x - m) / s).float(), 1)
        c, p = prob.max(1)
        preds.append(p.cpu()); confs.append(c.cpu())
    return torch.cat(preds).numpy(), torch.cat(confs).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", default="results/selections/frozen_selections.json")
    ap.add_argument("--config", default=None, help="configs/base.yaml by default")
    ap.add_argument("--checkpoints-dir", default=None, help="default: paths.checkpoints_dir of the config")
    ap.add_argument("--out", default="results/predictions", help="predictions of the selected checkpoints (committed)")
    ap.add_argument("--extra-out", default="outputs/predictions_all_epochs",
                    help="predictions of the other epochs from --all-epochs (not committed)")
    ap.add_argument("--cache-dir", default="outputs/cache")
    ap.add_argument("--splits", nargs="+", default=["real_world", "product_test"], choices=["real_world", "product_test", "product_val"])
    ap.add_argument("--all-epochs", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, metavar="RUN:EPOCH",
                    help="score just these checkpoints (used by the sanity check), e.g. ft_b100_s42:12")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tag-check", action="store_true", help="offline tests only")
    args = ap.parse_args(argv)

    from src.train.utils import build_model, load_checkpoint  # Disha's model and checkpoint code

    cfg = load_config(args.config)
    frozen_path = Path(args.frozen)
    if not frozen_path.exists():
        sys.exit(f"{frozen_path} not found. Run scripts/06_select_and_freeze.sh first.")
    if not args.no_tag_check:
        check_frozen(frozen_path)
    frozen = json.loads(frozen_path.read_text())
    selected = set(needed_checkpoints(frozen, False))
    todo = needed_checkpoints(frozen, args.all_epochs)
    if args.only:
        todo = [(r.split(":")[0], int(r.split(":")[1])) for r in args.only]
    ckpt_root = Path(args.checkpoints_dir) if args.checkpoints_dir else repo_path(cfg["paths"]["checkpoints_dir"])
    out_root, extra_root = Path(args.out), Path(args.extra_out)
    root_of = lambda run, epoch: out_root if (run, epoch) in selected or args.only else extra_root
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    icfg, num_classes = cfg["image"], int(cfg["data"]["num_classes"])

    # which rows to score
    sdir = repo_path(cfg["paths"]["splits_dir"])
    prod = pd.read_csv(sdir / "product_split.csv")
    rows = {"real_world": pd.read_csv(sdir / "target_real_world.csv")[["row_id", "label"]],
            "product_test": prod[prod.split == "test"][["row_id", "label"]],
            "product_val": prod[prod.split == "val"][["row_id", "label"]]}
    hf = None
    images = {}
    for name in args.splits:
        if hf is None:
            hf = load_officehome(cfg)
        r = rows[name].reset_index(drop=True)
        print(f"{name}: {len(r)} images")
        images[name] = cache_images(hf, r.row_id.to_numpy(), Path(args.cache_dir) / f"{name}_u8.pt",
                                    icfg["size"], icfg["resize_eval"])
        rows[name] = r

    print(f"{len(todo)} checkpoints to score on {args.splits} (device {device})")
    for n, (run, epoch) in enumerate(todo, 1):
        dest = root_of(run, epoch)
        missing = [s for s in args.splits if not (dest / s / f"{run}_e{epoch:02d}_{s}.csv").exists()]
        if not missing:
            continue
        mode = mode_of(run)
        weights = cfg["train"]["pretrained_weights"] if mode == "linear_probe" else "none"
        model = build_model(num_classes, mode, weights)
        model = load_checkpoint(model, ckpt_root / run / f"epoch_{epoch:02d}.pt", mode).to(device).eval()
        for s in missing:
            pred, conf = predict(model, images[s], icfg["mean"], icfg["std"], device)
            df = rows[s].assign(pred=pred, confidence=np.round(conf, 5))
            path = dest / s / f"{run}_e{epoch:02d}_{s}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"[{n}/{len(todo)}] {run} epoch {epoch:02d} {s}: acc {(df.label == df.pred).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
