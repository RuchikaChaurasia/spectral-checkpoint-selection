"""Data audit for Office-Home (owner: Disha).

What it checks
  1. Row and domain counts against the proposal's numbers (PASS / MISMATCH).
  2. Class list: 65 classes, no "Clock" class, no class empty in any domain.
  3. Every image decodes; records width, height and colour mode.
  4. Perceptual-hash (pHash) near-duplicates inside Product -> duplicate groups.
     make_splits.py keeps every group inside one split (no leakage).
  5. Near-duplicates between Product and Real World (cross-domain leakage check).
  6. Flags questionable samples for manual review.
  7. Class distribution tables, image-size statistics, example figures.

Run (from the repo root):
    python -m src.data.audit

Outputs
  results/data_audit/   small committed files: summary.json, class_counts.csv,
                        class_counts.md, image_size_stats.csv, product_dup_groups.csv,
                        product_near_dup_pairs.csv, cross_domain_near_dup_pairs.csv,
                        flagged_samples.csv
  results/figures/      product_class_distribution.png, examples_product_vs_realworld.png
  outputs/audit/        image dumps for manual review (not committed)
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.officehome import (
    class_names,
    load_config,
    load_officehome,
    repo_path,
)

TINY_SIDE = 64          # images with a side smaller than this are flagged
MAX_ASPECT = 3.0        # width/height (or inverse) above this is flagged


# --------------------------------------------------------------------------- hashing
def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """Number of set bits in each uint64 (works on numpy 1.x and 2.x)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def phash_u64(img: Image.Image) -> np.uint64:
    import imagehash

    bits = imagehash.phash(img, hash_size=8).hash.flatten()  # 64 booleans
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def near_pairs(h_a: np.ndarray, h_b: np.ndarray | None, max_dist: int, chunk: int = 512):
    """All (i, j, dist) with Hamming distance <= max_dist.

    If h_b is None, compares h_a with itself and returns only i < j.
    """
    same = h_b is None
    h_b = h_a if same else h_b
    out = []
    for start in range(0, len(h_a), chunk):
        block = h_a[start : start + chunk, None] ^ h_b[None, :]
        dist = _popcount_u64(block)
        ii, jj = np.nonzero(dist <= max_dist)
        ii = ii + start
        if same:
            keep = ii < jj
            ii, jj = ii[keep], jj[keep]
        out.extend(zip(ii.tolist(), jj.tolist(), dist[ii - start, jj].tolist()))
    return out


def union_find_groups(n: int, pairs) -> np.ndarray:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    # relabel roots to 0..G-1 in order of first appearance
    _, group = np.unique(roots, return_inverse=True)
    return group


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-dup-dumps", type=int, default=40, help="near-duplicate pairs saved as images")
    args = ap.parse_args(argv)

    t0 = time.time()
    cfg = load_config(args.config)
    dcfg = cfg["data"]
    audit_dir = repo_path(cfg["paths"]["audit_dir"])
    fig_dir = repo_path(cfg["paths"]["figures_dir"])
    dump_dir = repo_path(cfg["paths"]["outputs_dir"]) / "audit"
    for d in (audit_dir, fig_dir, dump_dir):
        d.mkdir(parents=True, exist_ok=True)

    src_dom, tgt_dom = dcfg["source_domain"], dcfg["target_domain"]
    max_dist = int(cfg["split"]["near_dup_hamming"])

    print("Loading dataset ...")
    ds = load_officehome(cfg)
    names = class_names(ds)
    summary: dict = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hf_name": dcfg["hf_name"],
        "hf_revision": dcfg["hf_revision"],
        "python": platform.python_version(),
    }

    # ---- 1. counts --------------------------------------------------------
    domains = np.array(ds["domain"], dtype=object)
    labels = np.array(ds["label"], dtype=np.int64)
    dom_counts = pd.Series(domains).value_counts().to_dict()
    expected = dcfg["expected_counts"]
    count_checks = {"total": {"found": len(ds), "expected": expected["total"]}}
    for dom in [src_dom, tgt_dom, *dcfg["extra_domains"]]:
        count_checks[dom] = {"found": int(dom_counts.get(dom, 0)), "expected": expected.get(dom)}
    for v in count_checks.values():
        v["status"] = "PASS" if v["found"] == v["expected"] else "MISMATCH"
    summary["domain_names_found"] = sorted(dom_counts)
    summary["count_checks"] = count_checks
    print("Counts:")
    for k, v in count_checks.items():
        print(f"  {k:<11} found {v['found']:>6}  expected {v['expected']}  {v['status']}")

    # ---- 2. class list ----------------------------------------------------
    lower = [n.lower() for n in names]
    per_dom_class = (
        pd.crosstab(pd.Series(labels, name="label"), pd.Series(domains, name="domain"))
        .reindex(range(len(names)), fill_value=0)
    )
    per_dom_class.index = [names[i] for i in per_dom_class.index]
    empty = {
        dom: per_dom_class.index[per_dom_class[dom] == 0].tolist()
        for dom in per_dom_class.columns
        if (per_dom_class[dom] == 0).any()
    }
    summary["classes"] = {
        "num_classes": len(names),
        "expected": dcfg["num_classes"],
        "status": "PASS" if len(names) == dcfg["num_classes"] else "MISMATCH",
        "has_class_named_clock": "clock" in lower,
        "has_alarm_clock": "alarm_clock" in lower,
        "duplicate_names_case_insensitive": sorted({n for n in lower if lower.count(n) > 1}),
        "classes_with_zero_images": empty,
        "note": (
            "The original Office-Home CSV lists a 'Clock' folder with no images; the HF version "
            "drops it, leaving 65 classes. Confirmed here if has_class_named_clock is false."
        ),
    }
    print(f"Classes: {len(names)} (Clock present: {'clock' in lower}; empty in some domain: {empty or 'none'})")

    counts_out = per_dom_class.copy()
    counts_out.index.name = "class"
    counts_out.to_csv(audit_dir / "class_counts.csv")
    pc = per_dom_class[src_dom]
    summary["product_class_counts"] = {
        "min": int(pc.min()), "min_class": str(pc.idxmin()),
        "max": int(pc.max()), "max_class": str(pc.idxmax()),
        "median": float(pc.median()),
    }
    # Markdown table to paste into the dataset card
    cols = [c for c in [src_dom, tgt_dom, *dcfg["extra_domains"]] if c in per_dom_class.columns]
    md = ["| Class | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for cname, row in per_dom_class[cols].iterrows():
        md.append(f"| {cname} | " + " | ".join(str(int(row[c])) for c in cols) + " |")
    md.append("| **Total** | " + " | ".join(str(int(per_dom_class[c].sum())) for c in cols) + " |")
    (audit_dir / "class_counts.md").write_text("\n".join(md) + "\n")

    # ---- 3. decode every image (raw bytes, so one bad file cannot crash the loop)
    from datasets import Image as HFImage

    raw = ds.cast_column("image", HFImage(decode=False))
    n = len(raw)
    widths = np.zeros(n, dtype=np.int32)
    heights = np.zeros(n, dtype=np.int32)
    modes = np.empty(n, dtype=object)
    readable = np.ones(n, dtype=bool)
    errors = {}
    hash_rows = set(np.flatnonzero((domains == src_dom) | (domains == tgt_dom)).tolist())
    hashes = {}
    print(f"Decoding {n} images (and hashing {len(hash_rows)}) ...")
    for i, ex in enumerate(raw):
        try:
            info = ex["image"]
            data = info.get("bytes")
            if data is None and info.get("path"):
                data = Path(info["path"]).read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                widths[i], heights[i] = im.size
                modes[i] = im.mode
                if i in hash_rows:
                    hashes[i] = phash_u64(im.convert("RGB"))
        except Exception as e:  # noqa: BLE001 - we want to log any failure
            readable[i] = False
            modes[i] = None
            errors[i] = f"{type(e).__name__}: {e}"
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}")
    summary["unreadable_images"] = {"count": int((~readable).sum()), "row_ids": sorted(errors)[:50]}
    print(f"Unreadable images: {int((~readable).sum())}")

    # image-size statistics per domain
    size_rows = []
    for dom in sorted(set(domains.tolist())):
        m = (domains == dom) & readable
        size_rows.append({
            "domain": dom,
            "n": int(m.sum()),
            "width_min": int(widths[m].min()), "width_median": float(np.median(widths[m])), "width_max": int(widths[m].max()),
            "height_min": int(heights[m].min()), "height_median": float(np.median(heights[m])), "height_max": int(heights[m].max()),
            "non_rgb": int(sum(1 for x in modes[m] if x != "RGB")),
        })
    pd.DataFrame(size_rows).to_csv(audit_dir / "image_size_stats.csv", index=False)

    # ---- 4. near-duplicates inside Product ---------------------------------
    src_rows = np.array([r for r in np.flatnonzero(domains == src_dom) if r in hashes])
    tgt_rows = np.array([r for r in np.flatnonzero(domains == tgt_dom) if r in hashes])
    h_src = np.array([hashes[r] for r in src_rows], dtype=np.uint64)
    h_tgt = np.array([hashes[r] for r in tgt_rows], dtype=np.uint64)

    pairs = near_pairs(h_src, None, max_dist)
    pair_df = pd.DataFrame(
        [(int(src_rows[i]), int(src_rows[j]), d) for i, j, d in pairs],
        columns=["row_a", "row_b", "hamming"],
    )
    pair_df["label_a"] = [names[labels[r]] for r in pair_df.row_a]
    pair_df["label_b"] = [names[labels[r]] for r in pair_df.row_b]
    pair_df["same_label"] = pair_df.label_a == pair_df.label_b
    pair_df.sort_values(["hamming", "row_a"]).to_csv(audit_dir / "product_near_dup_pairs.csv", index=False)

    group = union_find_groups(len(src_rows), pairs)
    sizes = np.bincount(group)
    grp_df = pd.DataFrame({
        "row_id": src_rows.astype(int),
        "label": labels[src_rows].astype(int),
        "class": [names[labels[r]] for r in src_rows],
        "dup_group": group.astype(int),
        "group_size": sizes[group].astype(int),
        "phash": [f"{int(h):016x}" for h in h_src],
    })
    # unreadable Product images are left out of the groups file and therefore out of the split
    grp_df.to_csv(audit_dir / "product_dup_groups.csv", index=False)
    multi = sizes[sizes > 1]
    mixed_label_groups = int(
        grp_df[grp_df.group_size > 1].groupby("dup_group")["label"].nunique().gt(1).sum()
    )
    summary["product_near_duplicates"] = {
        "hamming_threshold": max_dist,
        "pairs": len(pair_df),
        "exact_hash_pairs": int((pair_df.hamming == 0).sum()) if len(pair_df) else 0,
        "groups_with_2plus_images": int(len(multi)),
        "images_in_such_groups": int(multi.sum()),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "groups_with_conflicting_labels": mixed_label_groups,
    }
    print(f"Product near-duplicate pairs: {len(pair_df)} -> {len(multi)} groups of 2+ images")

    # ---- 5. Product vs Real World -----------------------------------------
    xpairs = near_pairs(h_src, h_tgt, max_dist)
    x_df = pd.DataFrame(
        [(int(src_rows[i]), int(tgt_rows[j]), d) for i, j, d in xpairs],
        columns=["product_row", "realworld_row", "hamming"],
    )
    x_df["product_class"] = [names[labels[r]] for r in x_df.product_row]
    x_df["realworld_class"] = [names[labels[r]] for r in x_df.realworld_row]
    x_df.sort_values("hamming").to_csv(audit_dir / "cross_domain_near_dup_pairs.csv", index=False)
    summary["cross_domain_near_duplicates"] = {
        "pairs": len(x_df),
        "note": "Pairs here mean a Real World test image may look almost identical to a Product image. "
                "Review them by eye; report any true duplicates as a leakage limitation.",
    }
    print(f"Product vs Real World near-duplicate pairs: {len(x_df)}")

    # ---- 6. flagged samples -----------------------------------------------
    flags = []
    for r in range(n):
        reasons = []
        if not readable[r]:
            reasons.append("unreadable: " + errors[r])
        else:
            if modes[r] != "RGB":
                reasons.append(f"mode {modes[r]} (converted to RGB at load time)")
            if min(widths[r], heights[r]) < TINY_SIDE:
                reasons.append(f"tiny {widths[r]}x{heights[r]}")
            ar = max(widths[r], heights[r]) / max(1, min(widths[r], heights[r]))
            if ar > MAX_ASPECT:
                reasons.append(f"aspect ratio {ar:.1f}")
        if reasons:
            flags.append({"row_id": r, "domain": domains[r], "class": names[labels[r]], "reasons": "; ".join(reasons)})
    for _, p in pair_df[~pair_df.same_label].iterrows():
        flags.append({"row_id": int(p.row_a), "domain": src_dom, "class": p.label_a,
                      "reasons": f"near-duplicate of row {int(p.row_b)} labelled {p.label_b}"})
    flag_df = pd.DataFrame(flags, columns=["row_id", "domain", "class", "reasons"])
    flag_df.to_csv(audit_dir / "flagged_samples.csv", index=False)
    summary["flagged_samples"] = {
        "count": len(flag_df),
        "by_domain": flag_df.domain.value_counts().to_dict() if len(flag_df) else {},
        "manual_review_todo": "Look at Real World and Product examples for identifiable people, "
                              "watermarks and wrong labels; note them in the dataset card.",
    }

    # ---- 7. figures and image dumps ---------------------------------------
    _figures(ds, names, labels, domains, src_dom, tgt_dom, per_dom_class, fig_dir)
    _dump_images(ds, names, labels, domains, src_dom, tgt_dom, pair_df, x_df, dump_dir, args.max_dup_dumps)

    summary["seconds"] = round(time.time() - t0, 1)
    with open(audit_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Done in {summary['seconds']} s. Summary: {audit_dir / 'summary.json'}")


def _figures(ds, names, labels, domains, src_dom, tgt_dom, per_dom_class, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = per_dom_class[src_dom].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(counts)), counts.values, color="#4C72B0")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=7)
    ax.set_ylabel("images")
    ax.set_title(f"{src_dom}: images per class (n={int(counts.sum())})")
    fig.tight_layout()
    fig.savefig(fig_dir / "product_class_distribution.png", dpi=150)
    plt.close(fig)

    # Same 6 classes in both domains, chosen with a fixed seed
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(names), size=min(6, len(names)), replace=False).tolist())
    fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.6))
    axes = np.atleast_2d(axes)
    for col, c in enumerate(pick):
        for row, dom in enumerate([src_dom, tgt_dom]):
            ax = axes[row, col]
            idx = np.flatnonzero((labels == c) & (domains == dom))
            ax.axis("off")
            if len(idx):
                ax.imshow(ds[int(idx[0])]["image"].convert("RGB"))
            ax.set_title(f"{dom}\n{names[c]}", fontsize=9)
    fig.suptitle("Source (Product) vs target (Real World): same classes", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "examples_product_vs_realworld.png", dpi=150)
    plt.close(fig)


def _dump_images(ds, names, labels, domains, src_dom, tgt_dom, pair_df, x_df, dump_dir, max_pairs):
    """Save images for manual review: one per class per domain, and near-duplicate pairs."""
    ex_dir = dump_dir / "examples"
    for dom in (src_dom, tgt_dom):
        d = ex_dir / dom.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        for c in range(len(names)):
            idx = np.flatnonzero((labels == c) & (domains == dom))
            if len(idx):
                r = int(idx[0])
                ds[r]["image"].convert("RGB").save(d / f"{names[c]}_row{r}.jpg", quality=90)

    for name, df, a, b in (
        ("product_near_dups", pair_df, "row_a", "row_b"),
        ("cross_domain_near_dups", x_df, "product_row", "realworld_row"),
    ):
        d = dump_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for _, p in df.sort_values("hamming").head(max_pairs).iterrows():
            ia = ds[int(p[a])]["image"].convert("RGB").resize((224, 224))
            ib = ds[int(p[b])]["image"].convert("RGB").resize((224, 224))
            canvas = Image.new("RGB", (452, 224), "white")
            canvas.paste(ia, (0, 0))
            canvas.paste(ib, (228, 0))
            canvas.save(d / f"d{int(p.hamming)}_{int(p[a])}_{int(p[b])}.jpg", quality=90)


if __name__ == "__main__":
    main()
