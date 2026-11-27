"""Load Office-Home from Hugging Face at a pinned revision, plus small helpers.

Image ID convention used everywhere in this project:
    row_id = row number of the image in the single "train" split of
             flwrlabs/office-home at the revision pinned in configs/base.yaml.
The HF dataset stores no file names, so the row number is the only stable ID,
and it is stable only because the revision is pinned.

For offline testing, set OFFICEHOME_LOCAL_PATH to a folder made with
`datasets.Dataset.save_to_disk` that has the same columns (image, domain, label).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "base.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(rel: str) -> Path:
    """Resolve a path from the config relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def load_officehome(cfg: dict, local_path: str | None = None):
    """Return the full HF dataset (all 4 domains) as one `datasets.Dataset`."""
    import datasets

    local_path = local_path or os.environ.get("OFFICEHOME_LOCAL_PATH")
    if local_path:
        ds = datasets.load_from_disk(local_path)
    else:
        ds = datasets.load_dataset(
            cfg["data"]["hf_name"],
            split="train",
            revision=cfg["data"]["hf_revision"],
        )
    missing = {"image", "domain", "label"} - set(ds.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns {missing}; got {ds.column_names}")
    return ds


def class_names(ds) -> list[str]:
    return list(ds.features["label"].names)


def domain_and_label_arrays(ds):
    """Domain strings and integer labels for every row, without decoding images."""
    domains = np.array(ds["domain"], dtype=object)
    labels = np.array(ds["label"], dtype=np.int64)
    return domains, labels


def rows_for_domain(ds, domain: str) -> np.ndarray:
    domains, _ = domain_and_label_arrays(ds)
    rows = np.flatnonzero(domains == domain)
    if len(rows) == 0:
        found = sorted(set(domains.tolist()))
        raise ValueError(f"No rows for domain {domain!r}. Domains present: {found}")
    return rows


def sha256_of_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
