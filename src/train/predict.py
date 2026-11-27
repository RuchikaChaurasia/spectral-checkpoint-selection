"""Write predictions of one saved checkpoint on a Product split (owner: Disha).

For Shruti's harness checks before the freeze, e.g. the linear-probe predictions:
    python -m src.train.predict --run-dir checkpoints/lp_b100_s42 --epoch 12 --split val

Only Product val and test are allowed. Real World scoring is done by the
evaluation harness (src/eval) after the Nov 16 freeze, never from here.

Output CSV columns: row_id, label, pred, confidence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import OfficeHomeRows, build_transforms, product_rows
from src.data.officehome import load_config, load_officehome, repo_path
from src.train.utils import build_model, load_checkpoint, macro_f1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epoch", type=int, default=None, help="default: the run's lowest-val-loss epoch")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default=None, help="default: outputs/predictions/<run>_e<epoch>_<split>.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    info = json.loads((run_dir / "run_info.json").read_text())
    epoch = args.epoch if args.epoch is not None else info.get("lowest_val_loss_epoch")
    if epoch is None:
        raise SystemExit("No --epoch given and the run has no lowest_val_loss_epoch yet.")
    ckpt = run_dir / f"epoch_{epoch:02d}.pt"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = int(cfg["data"]["num_classes"])
    model = build_model(num_classes, info["mode"], info["pretrained_weights"])
    model = load_checkpoint(model, ckpt, info["mode"]).to(device).eval()

    hf = load_officehome(cfg)
    rows = product_rows(cfg, args.split)
    loader = DataLoader(OfficeHomeRows(hf, rows, build_transforms(cfg, train=False)),
                        batch_size=128, shuffle=False, num_workers=args.num_workers)
    out_rows, preds, ys = [], [], []
    with torch.no_grad():
        for x, y, r in loader:
            prob = torch.softmax(model(x.to(device)).float(), 1).cpu()
            conf, pred = prob.max(1)
            out_rows += list(zip(r.tolist(), y.tolist(), pred.tolist(), conf.tolist()))
    df = pd.DataFrame(out_rows, columns=["row_id", "label", "pred", "confidence"])
    out = Path(args.out) if args.out else (
        repo_path(cfg["paths"]["outputs_dir"]) / "predictions" / f"{info['run_name']}_e{epoch:02d}_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    acc = float((df.label == df.pred).mean())
    f1 = macro_f1(df.label.to_numpy(), df.pred.to_numpy(), num_classes)
    print(f"{info['run_name']} epoch {epoch} on Product {args.split}: acc {acc:.4f}, macro-F1 {f1:.4f} -> {out}")


if __name__ == "__main__":
    main()
