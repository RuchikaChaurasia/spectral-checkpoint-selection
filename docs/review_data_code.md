# Code review: data pipeline (`src/data/`)

Files reviewed: `officehome.py`, `audit.py`, `make_splits.py`, `datasets.py`, `scripts/01_data.sh`.
Method: read the code, reran it on a different platform (see `docs/reproduction.md`), and checked the outputs directly.

## What works well
1. **Leakage guard.** `datasets.py` refuses to load Real World (`product_rows` only accepts train/val/test), so the
   training code cannot touch the target domain by accident.
2. **Pinned data version.** The Hugging Face revision is pinned in `configs/base.yaml`, and `OfficeHomeRows` checks
   that manifest labels match the loaded dataset, so a wrong revision fails loudly instead of silently.
3. **Near-duplicates kept together.** Whole duplicate groups go into one split; verified: 0 groups cross splits.
4. **Split cannot be changed by accident.** `make_splits.py` stops if a new split differs from the saved one, unless `--force` is given.
5. **Nested budgets.** Budgets are built by ranking each class once per seed and taking the first k, so 10% is inside
   25% is inside 50% is inside 100%; verified for all three seeds.
6. **Reproducible.** Rerun from scratch on Kaggle gives the same split for every image.

## Issues and suggestions
1. **Images with two labels (decision needed).** 16 near-duplicate groups contain more than one label
   (`results/data_audit/conflicting_label_review.csv`). Some are the same picture under two classes
   (e.g. Computer/Laptop, Calendar/Notebook). Most sit in **train**, so the model sees the same image with two
   different labels (13 groups); 3 groups sit in **val** and none in test, which slightly affects validation loss and checkpoint selection. In total 38 images are involved.
   The team should decide: keep them (and say so in the report), or drop them before training.
   Since all 24 runs are finished, keeping them and documenting it is the practical choice.
2. **File hash depends on the platform.** `split_summary.json` stores a SHA-256 of `product_split.csv`, but that file
   includes `phash` values, and 22 of them differ on Kaggle versus Colab. The split itself is identical.
   Suggestion: pin Pillow/imagehash versions in `requirements.txt`, or verify only `row_id` + `split`.
3. **Cross-domain duplicates: 14 pairs vs 2 copies.** The audit finds 14 Product-Real World near-duplicate pairs;
   the dataset card reports 2 copies. Please make sure the card explains that 2 of the 14 were confirmed by manual review.
4. **Revision check is partial.** `OfficeHomeRows` checks only the first 50 rows against the dataset. Probably
   enough, but checking all rows' labels (no image decoding needed) would be cheap.
5. **Group label rule.** A duplicate group with mixed labels is counted under its majority label when computing
   per-class quotas. This only affects how many images each class gets per split (counts still match 3,551/444/444),
   not any image's label. Fine, but worth one sentence in the dataset card.

