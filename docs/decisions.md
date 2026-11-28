# Decisions and deviations log

Add an entry whenever a setting or rule changes, or when we learn something that changes the plan. Newest at the bottom.

| Date | Decision or deviation | Why | Real-World results seen first? | Who |
| --- | --- | --- | --- | --- |
| 2026-11-07 | Training seeds are 42, 123, 999 (as in the submitted proposal); the Product split uses separate seed 0, used once | Keep the proposal and the repo consistent | No | Team |
| 2026-11-07 | Learning rate 1e-4 (3e-4 chosen worse on Product validation loss: 0.437 vs 0.579) | LR pilot, Disha | No | Disha |
| 2026-11-07 | The proposal's note about an empty "Clock" class is wrong: the dataset has 65 classes and no Clock class | Audit result | No | Disha |
| 2026-11-07 | At the 10% budget the 444-image validation set is larger than the roughly 355 training images; one fixed validation set is kept and stated in the report | Favours the validation-loss baseline, so a Rule W win would be conservative | No | Team |
| 2026-11-07 | Linear-probe learning rate is 1e-3 and was not tuned; the fine-tuning rate came from a two-value pilot. Reported as a limitation | Probe is a secondary reference | No | Team to confirm |
| 2026-11-07 | Image ID is the row number in the pinned Hugging Face revision; near-duplicates (pHash distance <= 4) stay in one split | Reproducibility and no leakage | No | Disha |
| 2026-11-07 | Validation-loss baseline: lowest Product validation loss over epochs 1-30, earliest epoch on ties | Written to each run_info.json as lowest_val_loss_epoch | No | Disha, Ruchika |
| 2026-11-26 | Rule W is analysed on the 12 fine-tuning runs only, with WeightWatcher library defaults, plain mean alpha over layers whose fit succeeded, epoch 0 excluded. Linear-probe runs are not analysed | They train only the head | No | Ruchika |
| 2026-11-26 | Result: mean alpha is 3.33-4.30 at every epoch of all 12 runs (threshold 2.0). Rule W fires in 0 of 12 runs for all six variants, so it selects epoch 30 everywhere; ablations 1 and 2 give identical selections. Rule W stays the primary rule as pre-specified; no new rule is added | Reported as a negative result; changing the rule after seeing alpha curves would be tuning | No | Ruchika |
| 2026-11-26 | Threshold sensitivity (2.0 to 4.0) is reported descriptively from alpha only; it is not used to choose a rule | No stable operating point exists | No | Ruchika |
| 2026-11-26 | Linear-probe reference baseline is scored at its lowest Product validation-loss epoch, frozen in the same file | Same selection protocol as fine-tuning | No | Ruchika |
| 2026-11-26 |Hardware: all 24 runs were trained on Google Colab with one NVIDIA A100-SXM4-40GB, not the RTX 4070 assumed in the work plan. Training took 41.8 minutes in total (52 to 173 s per run, peak GPU memory 0.8 GB), far below the planned 6 to 10 GPU-hours | Colab A100 was available and fast enough; closes the work plan's open compute question. Times from results/training_time_summary.csv | No | Disha |
| 2026-11-26 |Labels of the 10 Product near-duplicate groups that show the same picture under two classes (22 images; e.g. Computer/Laptop, File_Cabinet/Folder) are left unchanged; 8 groups are in train, 2 in val, none in test | Stay comparable with published Office-Home results; the label noise affects every method equally. Details in notebooks/dataset_card.md | No (labels were never changed; recorded here to close the open point in docs/review_data_code.md) | Disha |

## Open questions
- Does alpha carry any signal at all about Real-World F1? Answered after the freeze by the descriptive alpha-versus-F1analysis over all epochs (`src/eval/aggregate.py`); not used to select anything.
