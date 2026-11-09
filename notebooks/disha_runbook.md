# Disha's runbook: data and training

Everything here can be done without waiting for anyone. Run all commands from the repo root.

## 1. Setup
```bash
pip install torch torchvision datasets imagehash pyyaml numpy pandas scikit-learn matplotlib pillow
```
(Send this list to Sadaf for `requirements.txt`.) Add the lines in `.gitignore` to the team's `.gitignore`: checkpoints and image dumps must never go to GitHub.

## 2. Order of work
| # | Task | Command | Output | Commit? |
|---|---|---|---|---|
| 1 | Data audit | `python -m src.data.audit` | `results/data_audit/*`, two figures, `outputs/audit/` | yes, except `outputs/` |
| 2 | Split + budgets + target lists | `python -m src.data.make_splits` | `configs/splits/*` | **yes, by Nov 7** (Sadaf checks it) |
| 3 | Dataset card | fill `docs/dataset_card.md` | | yes |
| 4 | Baseline (linear probe) | `bash scripts/02_baseline.sh` | `checkpoints/lp_b100_s42/`, predictions for Shruti, loss curve | figure only |
| 5 | Progress-report loss curve | one fine-tuning run (see Colab notebook, cell 6) | `results/figures/loss_curve_*.png` | figure only |
| 6 | LR pilot | `bash scripts/03_lr_pilot.sh` | `configs/lr_choice.json` | yes |
| 7 | All 24 runs | `bash scripts/04_train_all.sh` | `checkpoints/<run>/`, `results/training_time_summary.csv` | summary only |

Steps 1 and 2 together are `bash scripts/01_data.sh`.

## 3. What each file does
| File | Purpose |
|---|---|
| `configs/base.yaml` | Every setting: pinned dataset revision, split seed 0, training seeds, budgets, image size, optimizer, epochs, paths |
| `src/data/officehome.py` | Loads the dataset at the pinned revision; config and hashing helpers |
| `src/data/audit.py` | Counts, class check, unreadable files, image sizes, pHash near-duplicates (inside Product and Product vs Real World), flagged samples, figures |
| `src/data/make_splits.py` | Fixed 80/10/10 Product split (duplicates kept together, exact 3,551/444/444), nested budgets per seed, target lists; refuses to overwrite a different split |
| `src/data/datasets.py` | PyTorch dataset and transforms; **cannot load Real World** by design |
| `src/train/train.py` | One run: linear probe or fine-tuning, checkpoint every epoch (epoch 0 too), metrics, run info, resume, optional per-epoch hook |
| `src/train/utils.py` | Seeding, model, checkpoint saving/loading, environment info |
| `src/train/predict.py` | Predictions of a checkpoint on Product val/test (for Shruti's harness check) |
| `src/train/run_all.py` | Runs the pilot or all 24 runs in the order 100 → 10 → 25 → 50; skips finished runs |
| `src/train/select_lr.py` | Picks the LR with the lowest Product validation loss |
| `src/train/report_runs.py` | Loss-curve figures and the training-time table |
| `src/data/make_fake_dataset.py` | Tiny fake dataset to test everything offline |
| `notebooks/colab_runner.ipynb` | Runs the steps on Colab, with checkpoints saved to Drive |

## 4. Decisions built into the code (tell the team)
- **Image ID = row number** in the pinned Hugging Face revision (the dataset has no file names).
- **Split seed 0**; training seeds 42, 123, 999; the split is made once.
- **Near-duplicate threshold:** pHash distance ≤ 4 out of 64.
- **Augmentation:** train RandomResizedCrop(224, scale 0.6–1.0) + flip; eval Resize(256) + CenterCrop(224).
- **AdamW, weight decay 0.01, constant LR, batch 32, 30 epochs, mixed precision on GPU.**
- **Linear-probe LR 1e-3** (not covered by the pilot): confirm with the team.
- **Validation-loss selection** (written to each `run_info.json` as `lowest_val_loss_epoch`): lowest Product val loss over epochs 1–30, earliest epoch on ties. Ruchika reproduces this with her function; agree the tie rule with her.
- **Checkpoints:** weights only, fp32; whole model for fine-tuning (45 MB), head only for the probe (135 KB).

## 5. Things to agree with teammates
- **Ruchika:** run WeightWatcher on saved checkpoints, or call it inside training with `--epoch-hook module:function` (called after each checkpoint with `model, epoch, run_dir, checkpoint_path`)?
- **Shruti:** predictions are CSVs with `row_id, label, pred, confidence`; checkpoints are `checkpoints/<run>/epoch_XX.pt`; the target list is `configs/splits/target_real_world.csv`.
- **Sadaf:** clean-checkout check = `bash scripts/01_data.sh` must print 3,551 / 444 / 444 PASS and 0 unreadable images, and the file hashes must match `split_summary.json`.

## 6. Tested
Tested offline on a fake dataset (CPU, no pretrained weights): audit finds planted duplicates, a cross-domain copy and a corrupt file; the split hits 3,551/444/444 exactly at full size; budgets are 355/888/1,776/3,551 and nested; a run killed mid-way and resumed gives bit-identical weights. **Not yet run on the real dataset or a GPU**: the first real run is yours.

## 7. AI-assistance note
This code was drafted with Claude (Anthropic) and reviewed by Disha. The course requires student-owned code: read every file, make sure you can explain it, and list it in the README and the AI disclosure.

## Data-access section for the README (for Sadaf to paste)
> The project uses the Hugging Face copy of Office-Home, `flwrlabs/office-home`, pinned to revision `d430d44779e929b335be368ab3456484e5de36bb`. It downloads automatically (about 1.2 GB) the first time any script runs. Image IDs in `configs/splits/` are row numbers in that revision. License: fair-use notice of the original Office-Home website; research use only; images are not redistributed.
