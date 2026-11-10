# WeightWatcher for Checkpoint Selection Under Visual Domain Shift

**Question.** When fine-tuning an ImageNet-pretrained ResNet-18 on Office-Home *Product* images, can a pre-specified
WeightWatcher rule (Rule W) pick a checkpoint with higher *Real World* macro-F1 than standard validation-loss
selection, without using any Real World images to choose?

| Role | Method |
|---|---|
| Reference baseline | Linear probe on a frozen ImageNet-pretrained ResNet-18 |
| Strong baseline | Full ResNet-18 fine-tuning, checkpoint with the lowest Product validation loss |
| Team investigation | Same fine-tuning runs, checkpoint chosen by Rule W: the last checkpoint before mean WeightWatcher alpha stays below 2.0 for two consecutive epochs (epoch 0 excluded; final epoch if it never triggers) |

**Primary metric:** Real World macro-F1 over all 65 classes. Secondary: Product test macro-F1, accuracy, per-class
recall, confusion matrices, selected epoch, training and WeightWatcher time.
**Domains:** Product = source (train/val/test); Real World = held-out target; Art and Clipart = optional extra targets.

## 1. Setup
```bash
git clone https://github.com/RuchikaChaurasia/spectral-checkpoint-selection.git
cd spectral-checkpoint-selection
pip install -r requirements.txt
```
Tested with Python 3.13 on Google Colab and Kaggle. Run every command from the repository root.

## 2. Data access
The project uses the Hugging Face copy of Office-Home, `flwrlabs/office-home`, pinned to revision
`d430d44779e929b335be368ab3456484e5de36bb`. It downloads automatically (about 1.2 GB) the first time any script runs;
no manual download is needed. Image IDs in `configs/splits/` are row numbers in that revision.
License: fair-use notice of the original Office-Home website; research use only; images are not redistributed.
Dataset details: `notebooks/dataset_card.md`.

## 3. Reproduce the results
| Step | Command | Output |
|---|---|---|
| 1. Data audit + fixed split + label budgets | `bash scripts/01_data.sh` | `configs/splits/`, `results/data_audit/` (prints PASS) |
| 2. Linear-probe baseline (100%, seed 42) | `bash scripts/02_baseline.sh` | `checkpoints/lp_b100_s42/`, predictions, loss curve |
| 3. Learning-rate pilot (1e-4 vs 3e-4) | `bash scripts/03_lr_pilot.sh` | `configs/lr_choice.json` |
| 4. All 24 runs (12 fine-tuning + 12 linear probe) | `bash scripts/04_train_all.sh` | `checkpoints/<run>/epoch_00.pt ... epoch_30.pt`, `run_info.json`, loss curves |
| 5. WeightWatcher alpha logs + frozen selections | _to be added_ | _to be added_ |
| 6. Real World evaluation, main table | _to be added_ | _to be added_ |
| 7. Ablations 1 and 2 | _to be added_ | _to be added_ |

**Fixed settings** (all in `configs/base.yaml`): split seed 0; training seeds 42, 123, 999; label budgets
10/25/50/100% (nested); 224x224 input; AdamW, weight decay 0.01, batch 32, 30 epochs, checkpoint every epoch;
fine-tuning LR 1e-4 (chosen on Product validation loss only; see `configs/lr_choice.json`).

**Test-set rule:** Real World images are never used for training, validation, thresholds or choosing a checkpoint.
All selections are frozen before the first Real World evaluation.

## 4. Checkpoints
Checkpoints (about 17 GB) are not in this repository. Download them from the team's Google Drive folder:
https://drive.google.com/drive/folders/112iJXRXYml8QiohpJtn1PVJxiLnEVVDh?usp=sharing
Each run has `epoch_00.pt ... epoch_30.pt` plus `run_info.json` and `metrics.csv`. Place them in `checkpoints/<run>/`.
To regenerate them instead, run steps 1-4 above (about 1 GPU-hour in total on a Colab A100).

## 5. Repository layout
| Folder | Contents |
|---|---|
| `configs/` | `base.yaml` (all settings), split manifests, LR choice |
| `src/data/` | dataset loading, audit, split generation |
| `src/train/` | linear probe and fine-tuning, predictions, run reports |
| `src/spectral/` | WeightWatcher logging and selection rules |
| `src/eval/` | metrics, uncertainty, error analysis |
| `src/extras/` | ablation 2 and optional Art evaluation |
| `scripts/` | entry points listed above |
| `results/` | metric summaries and figures only |
| `docs/` | reproduction log, code review, responsible-use notes |
| `notebooks/` | Colab notebooks, dataset card, data-and-training runbook |

## 6. Reproduction log
See `docs/reproduction.md` for who reproduced which result, the commands, results and any fixes.

## 7. Attribution
- PyTorch and torchvision; ImageNet-pretrained ResNet-18 weights (torchvision `IMAGENET1K_V1`)
- WeightWatcher (Apache-2.0), https://github.com/CalculatedContent/WeightWatcher
- Office-Home dataset: H. Venkateswara et al., "Deep hashing network for unsupervised domain adaptation", CVPR 2017
- Hugging Face `datasets`, `imagehash`, NumPy, pandas, scikit-learn, Matplotlib, Pillow
