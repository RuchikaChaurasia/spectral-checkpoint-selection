# Spectral analysis: settings, pipeline check, how to run (Ruchika)

## Settings (fixed before any Real-World image is scored)
- WeightWatcher 0.7.7, library defaults, `analyze()` with no extra arguments. Layers analysed: the 20 Conv2d layers and the final Linear layer of ResNet-18 (21 layers).
- Per-epoch statistic: plain mean of alpha over the layers whose fit succeeded. A failed fit (status not "success", or alpha missing or not finite) is logged and left out of the mean.
- Rule W: first epoch from 1 on where the mean alpha is below 2.0 for `patience` consecutive epochs, select the epoch just before (epoch 1 if the streak starts at epoch 1 or 2); never fires, select epoch 30. Primary: all layers, patience 2.
- Ablation 1 (layer scope): all layers vs `layer4.*` + `fc`. Ablation 2 (patience): 1, 2, 3.
- Only the 12 fine-tuning runs are analysed. Linear-probe runs train only the 65-way head on a frozen backbone, so there is nothing to track.
- The code reads only checkpoints, `metrics.csv` and the alpha log. It never touches Real-World data.

## Pipeline check on one run (ft_b100_s42, three checkpoints)
| Epoch | Mean alpha, all 21 layers | Mean alpha, layer4 + fc | Layers fitted | Time for one checkpoint (CPU) |
| --- | --- | --- | --- | --- |
| 0 (new random head) | 3.637 | 4.427 | 21 of 21 | about 2 s |
| 12 (lowest validation loss) | 3.602 | 4.281 | 21 of 21 | about 2 s |
| 30 (last) | 3.506 | 4.415 | 21 of 21 | about 2 s |

- The new, random 65-way head (`fc`) has alpha 7.87 at epoch 0 and 5.74 at epoch 30: alpha falls as the layer is trained, as the theory expects. This is the sanity check that the numbers respond to training.
- Most backbone layers barely move in 30 epochs at learning rate 1e-4 (for example `layer1.0.conv1` 1.96, 1.95, 1.96), because they start from ImageNet weights. The largest change is `layer3.0.conv1` (6.74 to 3.91).
- No single layer has alpha below 1.95 at any of the three epochs, and the mean over layers stays between 3.5 and 3.6. **With threshold 2.0 the pre-specified Rule W did not come close to firing on this run.** This is one run and three checkpoints; the check on all 12 runs (`scripts/05_spectral.sh`, about 12 minutes) decides whether Rule W ever fires.
- Raw per-layer values: `results/alpha/pipeline_check_ft_b100_s42_epochs_00_12_30.csv`.

## How to run (Colab, CPU is enough)
```
!git clone https://github.com/RuchikaChaurasia/spectral-checkpoint-selection.git
%cd spectral-checkpoint-selection
!pip install -q weightwatcher pandas pyyaml pytest
from google.colab import drive; drive.mount('/content/drive')
!python -m pytest -q                               # 13 tests, a few seconds
!bash scripts/05_spectral.sh /content/drive/MyDrive/DeepLearning/group9/checkpoints
```
Then look at `results/alpha/mean_alpha.csv` before running `scripts/06_select_and_freeze.sh`.

## Verification done
- 13 unit tests (`python -m pytest -q`): the rule's edge cases (never fires, streak at epoch 1, failed fit, epoch 0, strict threshold, end of training), the validation-loss epoch against `run_info.json` (12 for ft_b100_s42), and an end-to-end freeze on synthetic alpha logs.
- `rule_w` was compared with the independently written `rule_w_epoch` in `src/extras/ablation2_patience.py` on 60,000 random alpha series (with missing and NaN values): no disagreements.
