# Reproduction log

## 1. Clean-checkout check of the data split (Sadaf, Nov 7 2026)

**What was checked:** Disha's split pipeline (`scripts/01_data.sh` -> `src/data/audit.py`, `src/data/make_splits.py`),
re-run by a member who did not write it, on a different platform.

**Setup**
- Platform: Kaggle notebook `sadaf-clean-checkout-check` (Disha's original: Google Colab)
- Python 3.13.16 | torch 2.11.0+cu130 | datasets 4.8.5 | pandas 2.3.3
- Code: copied from the team Drive folder `group9/` on Nov 7 2026 (configs, docs, scripts, src, results, notebooks;
  no checkpoints/outputs). Not yet from GitHub, because the code was not on GitHub at the time.
- Dataset: `flwrlabs/office-home`, revision `d430d44779e929b335be368ab3456484e5de36bb` (from `configs/base.yaml`)
- Disha's `configs/splits/` was moved aside before running, so every split file was regenerated from scratch.

**Commands**
```bash
pip install -q imagehash pyyaml datasets
bash scripts/01_data.sh 2>&1 | tee my_01_data_log.txt
```
Then compared each regenerated file in `configs/splits/` with Disha's (SHA-256), and compared `product_split.csv` column by column.

**Results**
| Check | Expected | Got | OK |
|---|---|---|---|
| Total / Product / Real World / Art / Clipart | 15,588 / 4,439 / 4,357 / 2,427 / 4,365 | same | yes |
| Classes ("Clock" absent) | 65 | 65, Clock absent | yes |
| Unreadable images | 0 | 0 | yes |
| Train / val / test | 3,551 / 444 / 444 | 3,551 / 444 / 444 | yes |
| Duplicate groups crossing splits | 0 | 0 | yes |
| Images in more than one split | 0 | 0 | yes |
| Classes present in each split | 65 | 65 / 65 / 65 | yes |
| Budgets (seeds 42, 123, 999) | 355 / 888 / 1,776 / 3,551, nested | same, nested | yes |
| `budgets_seed{42,123,999}.csv` | identical to Disha's | identical (SHA-256) | yes |
| `target_{real_world,art,clipart}.csv` | identical to Disha's | identical (SHA-256) | yes |
| `product_split.csv` split assignment | identical | **identical for all 4,439 images** | yes |
| `product_split.csv` file hash | identical | **differs**, see below | no (harmless) |

Audit runtime: 145.8 s for decoding 15,588 images.

**Discrepancy:** `product_split.csv` is not byte-identical because **22 of 4,439 `phash` values differ**.
`row_id`, `label`, `class`, `split`, `dup_group` and `group_size` match on every row. The difference most likely comes
from different image-library versions between Kaggle and Disha's Colab, which change a few perceptual-hash bits.
It changed no near-duplicate grouping and no split decision.
**Implication:** the split is reproducible, but the file hash in `split_summary.json` is platform-dependent.
Suggested fix: pin Pillow/imagehash versions in `requirements.txt`, or check the `row_id`+`split` columns instead of the whole-file hash.

**Cross-domain duplicates:** the audit reports 14 Product-Real World near-duplicate candidate pairs; the dataset card's
manual review confirms 2 of them are the same photo (`results/data_audit/cross_domain_review.csv`). Consistent.

**Fixes needed to run:** none (only `pip install imagehash pyyaml datasets` on Kaggle).

**Verdict: PASS.** The Product split, label budgets and target lists are reproduced from code and config alone, on a different platform.

## 2. Linear-probe baseline result (Nov 7 2026)

**What was checked:** the reported Product-validation score of the linear-probe reference baseline
(`lp_b100_s42`: frozen ImageNet ResNet-18, 100% label budget, seed 42), re-scored by a member who did not train it.

**Setup:** Kaggle notebook, same environment as section 1. Saved checkpoints (`checkpoints/lp_b100_s42/`, 31 epochs
plus `metrics.csv` and `run_info.json`) copied from the team Drive; original split files from `configs/splits/`.

**Command**
```bash
python -m src.train.predict --run-dir checkpoints/lp_b100_s42 --split val
```
(The script picks the run's lowest-validation-loss epoch from `run_info.json`: epoch 26.)

**Result**
| | Reported (`metrics.csv`, epoch 26) | Reproduced | Match |
|---|---|---|---|
| Product val accuracy | 0.8784 | 0.8784 | yes |
| Product val macro-F1 | 0.8660 | 0.8660 | yes |

**Fixes needed:** none. **Discrepancies:** none.
**Not covered:** retraining the probe from scratch (not required); Real World scores (scored only after the freeze).

**Verdict: PASS.**

## 3. Ablation 2 (trigger patience) table (Nov 26 2026)

**What was checked:** `results/tables/ablation_patience.csv`, recomputed with `src/extras/ablation2_patience.py`.

**Setup:** local Mac, Python 3.13.9, pandas 3.1.1, scikit-learn 1.9.1 (no GPU or images needed; only CSV files).
Repo at commit `2e4b18b` (after tag `selections-frozen`), with the scope fix below applied to the script.

**Commands**
```bash
python -m src.extras.ablation2_patience selftest
python -m src.extras.ablation2_patience select --alpha-log results/alpha/mean_alpha.csv --out ablation2_selections.csv
python -m src.extras.ablation2_patience score --selections ablation2_selections.csv --pred-dir results/predictions/real_world
```

**Result**
| Check | Result |
|---|---|
| Self-test (8 rule edge cases) | pass |
| Selections vs `results/selections/frozen_selections.json` (12 runs x 3 patiences) | 36 / 36 match |
| Mean and SD of Real World macro-F1 per budget and patience (12 cells) | match, differences below 0.0001 (rounding to 4 decimals) |
| Selected epoch and firing rate | identical |

**Fix needed:** the script first mixed the `all` and `last_stage` alpha rows; it now uses only `scope == all`.

**Verdict: PASS.**
