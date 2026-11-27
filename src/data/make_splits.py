"""Make the fixed Product split and the nested label-budget subsets (owner: Disha).

Run the audit first (it finds the near-duplicate groups), then:
    python -m src.data.make_splits

What it does
  * Splits Product 80/10/10 (train/val/test), stratified by class, with the fixed
    split seed from configs/base.yaml (0). Near-duplicate groups found by the audit
    are kept whole inside one split. Unreadable images are left out.
  * Per-class quotas use the largest-remainder method, so the totals land on
    round(10% of Product) = 444 for val and for test, and 3,551 for train.
  * For each training seed (42, 123, 999) it ranks the training images of every
    class in a seeded random order. The 10/25/50/100% budgets take the first k
    images of each class, so the budgets are nested: 10% is inside 25%, and so on.
  * Writes the Real World (and Art, Clipart) target lists. These are never used
    for training, validation or any choice; they exist only for the final scoring.

Outputs (configs/splits/, committed to GitHub)
  product_split.csv        row_id, label, class, split, dup_group, group_size, phash
  budgets_seed{S}.csv      row_id, label, class, rank_in_class, in_b10, in_b25, in_b50, in_b100
  target_real_world.csv, target_art.csv, target_clipart.csv
  split_summary.json       counts, checks and SHA-256 of every file

The manifest is meant to be made once. If it already exists and the new result
differs, the script stops; pass --force only if the whole team agrees.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.officehome import (
    class_names,
    load_config,
    load_officehome,
    repo_path,
    sha256_of_file,
)


def apportion(counts: np.ndarray, frac: float) -> np.ndarray:
    """Largest-remainder rounding: per-class quotas that sum to round(frac * total)."""
    raw = counts * frac
    base = np.floor(raw + 1e-9).astype(int)
    target = int(round(frac * counts.sum()))
    remainder = target - base.sum()
    if remainder > 0:
        # biggest fractional parts first; ties broken by class index (stable sort)
        order = np.argsort(-(raw - base), kind="stable")
        base[order[:remainder]] += 1
    return base


def split_product(groups: pd.DataFrame, fractions: dict, seed: int) -> pd.Series:
    """Return a Series row_id -> 'train'/'val'/'test'."""
    # each duplicate group belongs to the class most of its images have (ties: lowest label)
    group_label = groups.groupby("dup_group")["label"].agg(lambda s: s.value_counts().sort_index().idxmax())
    group_size = groups.groupby("dup_group").size()
    g = pd.DataFrame({"label": group_label, "size": group_size})
    n_labels = int(groups.label.max()) + 1
    per_class = g.groupby("label")["size"].sum().reindex(range(n_labels), fill_value=0).to_numpy()

    test_q = apportion(per_class, fractions["test"])
    val_q = apportion(per_class, fractions["val"])

    rng = np.random.default_rng(seed)
    assign = {}
    for c in range(n_labels):
        gids = np.sort(g.index[g.label == c].to_numpy())  # sort first so the order is reproducible
        gids = rng.permutation(gids)
        left = {"test": int(test_q[c]), "val": int(val_q[c])}
        for gid in gids:
            size = int(g.at[gid, "size"])
            if size <= left["test"]:
                assign[gid] = "test"
                left["test"] -= size
            elif size <= left["val"]:
                assign[gid] = "val"
                left["val"] -= size
            else:
                assign[gid] = "train"
    return groups.dup_group.map(assign).set_axis(groups.row_id)


def make_budgets(train: pd.DataFrame, percents: list[int], seed: int) -> pd.DataFrame:
    """Nested per-class subsets of the training split for one training seed."""
    rng = np.random.default_rng(seed)
    labels = sorted(train.label.unique())
    per_class = train.groupby("label").size().reindex(labels).to_numpy()

    quotas, prev = {}, np.zeros(len(labels), dtype=int)
    for p in sorted(percents):
        q = per_class.copy() if p >= 100 else apportion(per_class, p / 100)
        q = np.maximum(q, 1)               # every class keeps at least one image
        q = np.maximum(q, prev)            # nested: never fewer than the smaller budget
        q = np.minimum(q, per_class)
        quotas[p], prev = q, q

    parts = []
    for k, c in enumerate(labels):
        rows = np.sort(train.row_id[train.label == c].to_numpy())
        rows = rng.permutation(rows)
        part = train.set_index("row_id").loc[rows, ["label", "class"]].reset_index()
        part["rank_in_class"] = np.arange(len(rows))
        for p in percents:
            part[f"in_b{p}"] = part.rank_in_class < quotas[p][k]
        parts.append(part)
    return pd.concat(parts).sort_values("row_id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite an existing, different manifest")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    scfg, bcfg, dcfg = cfg["split"], cfg["budgets"], cfg["data"]
    audit_dir = repo_path(cfg["paths"]["audit_dir"])
    out_dir = repo_path(cfg["paths"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    groups_path = audit_dir / "product_dup_groups.csv"
    if not groups_path.exists():
        sys.exit(f"Missing {groups_path}. Run `python -m src.data.audit` first.")
    groups = pd.read_csv(groups_path)

    # ---- Product split ------------------------------------------------------
    split = split_product(groups, scfg["fractions"], int(scfg["seed"]))
    manifest = groups.assign(split=groups.row_id.map(split))[
        ["row_id", "label", "class", "split", "dup_group", "group_size", "phash"]
    ].sort_values("row_id").reset_index(drop=True)

    man_path = out_dir / "product_split.csv"
    if man_path.exists():
        old = pd.read_csv(man_path)
        same = old[["row_id", "split"]].reset_index(drop=True).equals(manifest[["row_id", "split"]])
        if same:
            print("product_split.csv already exists and is unchanged.")
        elif not args.force:
            sys.exit("product_split.csv already exists and the new split DIFFERS. "
                     "The split is fixed once; rerun with --force only if the whole team agrees.")
    manifest.to_csv(man_path, index=False)

    # ---- checks ---------------------------------------------------------------
    counts = manifest.split.value_counts().to_dict()
    expected = scfg["expected"]
    groups_crossing = int(manifest.groupby("dup_group").split.nunique().gt(1).sum())
    per_class_split = pd.crosstab(manifest["class"], manifest.split)
    classes_missing = {s: per_class_split.index[per_class_split[s] == 0].tolist()
                       for s in ("train", "val", "test") if s in per_class_split and (per_class_split[s] == 0).any()}
    checks = {
        "counts": {s: {"found": int(counts.get(s, 0)), "expected": expected[s],
                       "status": "PASS" if counts.get(s, 0) == expected[s] else "MISMATCH"}
                   for s in ("train", "val", "test")},
        "duplicate_groups_crossing_splits": groups_crossing,
        "classes_missing_from_a_split": classes_missing or "none",
    }
    print("Split counts:")
    for s, v in checks["counts"].items():
        print(f"  {s:<5} {v['found']:>5}  (expected {v['expected']})  {v['status']}")
    print(f"Duplicate groups crossing splits: {groups_crossing}")

    # ---- label budgets ----------------------------------------------------------
    train = manifest[manifest.split == "train"][["row_id", "label", "class"]]
    budget_info = {}
    for seed in bcfg["seeds"]:
        b = make_budgets(train, bcfg["percents"], int(seed))
        path = out_dir / f"budgets_seed{seed}.csv"
        b.to_csv(path, index=False)
        sizes = {f"b{p}": int(b[f"in_b{p}"].sum()) for p in bcfg["percents"]}
        ps = sorted(bcfg["percents"])
        nested = all((~b[f"in_b{lo}"] | b[f"in_b{hi}"]).all() for lo, hi in zip(ps, ps[1:]))
        budget_info[str(seed)] = {"sizes": sizes, "nested": bool(nested),
                                  "min_images_per_class_b10": int(b[b.in_b10].groupby("label").size().min()),
                                  "sha256": sha256_of_file(path)}
        print(f"Budgets seed {seed}: {sizes}  nested={nested}")

    # ---- target-domain lists (final scoring only) ----------------------------
    ds = load_officehome(cfg)
    names = class_names(ds)
    domains = np.array(ds["domain"], dtype=object)
    labels = np.array(ds["label"], dtype=np.int64)
    flagged = audit_dir / "flagged_samples.csv"
    unreadable = set()
    if flagged.exists():
        f = pd.read_csv(flagged)
        unreadable = set(f.row_id[f.reasons.astype(str).str.startswith("unreadable")].astype(int))
    targets = {}
    for dom in [dcfg["target_domain"], *dcfg["extra_domains"]]:
        rows = [r for r in np.flatnonzero(domains == dom) if r not in unreadable]
        t = pd.DataFrame({"row_id": rows, "label": labels[rows], "class": [names[labels[r]] for r in rows]})
        path = out_dir / f"target_{dom.lower().replace(' ', '_')}.csv"
        t.to_csv(path, index=False)
        targets[dom] = {"file": path.name, "n": len(t), "sha256": sha256_of_file(path)}

    summary = {
        "made_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_revision": dcfg["hf_revision"],
        "split_seed": int(scfg["seed"]),
        "near_dup_hamming": int(scfg["near_dup_hamming"]),
        "product_split_sha256": sha256_of_file(man_path),
        "checks": checks,
        "budgets": budget_info,
        "targets": targets,
        "per_class_split_counts": per_class_split.to_dict(orient="index"),
    }
    with open(out_dir / "split_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
