"""PyTorch datasets and transforms built on the committed split files (owner: Disha).

Only Product splits are exposed here: "train" (at a label budget), "val" and
"test". Real World is deliberately not loadable from this module, so the training
code cannot touch the target domain. Shruti's evaluation harness reads
configs/splits/target_real_world.csv after the selections are frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.data.officehome import repo_path

PRODUCT_SPLITS = ("train", "val", "test")


def build_transforms(cfg: dict, train: bool):
    icfg = cfg["image"]
    norm = T.Normalize(mean=icfg["mean"], std=icfg["std"])
    if train:
        return T.Compose([
            T.RandomResizedCrop(icfg["size"], scale=tuple(icfg["train_crop_scale"])),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.Resize(icfg["resize_eval"]),
        T.CenterCrop(icfg["size"]),
        T.ToTensor(),
        norm,
    ])


def product_rows(cfg: dict, split: str, budget: int = 100, seed: int | None = None) -> pd.DataFrame:
    """Rows (row_id, label) of one Product split; for 'train', the label-budget subset."""
    if split not in PRODUCT_SPLITS:
        raise ValueError(f"split must be one of {PRODUCT_SPLITS}; Real World is not loadable here")
    sdir = repo_path(cfg["paths"]["splits_dir"])
    if split == "train" and budget < 100:
        if seed is None:
            raise ValueError("a training seed is needed to pick the label-budget subset")
        b = pd.read_csv(sdir / f"budgets_seed{seed}.csv")
        col = f"in_b{budget}"
        if col not in b:
            raise ValueError(f"budget {budget}% not in budgets_seed{seed}.csv")
        return b[b[col]][["row_id", "label"]].reset_index(drop=True)
    m = pd.read_csv(sdir / "product_split.csv")
    return m[m.split == split][["row_id", "label"]].reset_index(drop=True)


class OfficeHomeRows(Dataset):
    """Images from the HF dataset, selected by row_id."""

    def __init__(self, hf_ds, rows: pd.DataFrame, transform):
        self.ds = hf_ds
        self.row_ids = rows.row_id.to_numpy(dtype=np.int64)
        self.labels = rows.label.to_numpy(dtype=np.int64)
        self.transform = transform
        # guard against a manifest that does not match the loaded dataset revision
        labels_only = hf_ds.select_columns(["label"])  # no image decoding
        sample = min(len(self.row_ids), 50)
        for r, y in zip(self.row_ids[:sample], self.labels[:sample]):
            if int(labels_only[int(r)]["label"]) != int(y):
                raise RuntimeError(
                    f"Row {r}: manifest label {y} != dataset label. Is the HF revision the pinned one?"
                )

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, i):
        r = int(self.row_ids[i])
        img = self.ds[r]["image"].convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i]), r
