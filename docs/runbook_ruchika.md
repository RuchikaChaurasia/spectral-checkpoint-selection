# Runbook: from Disha's commit to the frozen selections (Ruchika)

Order matters. Steps marked (Mac) run in Terminal, (Colab) in Google Colab.

## 0. One-time setup (Mac)
```bash
git config --global user.name  "Ruchika Chaurasia"
git config --global user.email "chaurasiaruchikaa@gmail.com"
brew install gh          # skip if already installed
gh auth login            # GitHub.com -> HTTPS -> login with browser
```

## 1. Optional but recommended: remove image-bearing notebooks from history
The repo is public. Two notebooks contain images (possibly people). Do this once, before others clone.
```bash
brew install git-filter-repo
git clone https://github.com/RuchikaChaurasia/spectral-checkpoint-selection.git fresh && cd fresh
git filter-repo --invert-paths \
  --path notebooks/Group9_Dataset_Card.ipynb --path notebooks/Group9_Colab.ipynb
git remote add origin https://github.com/RuchikaChaurasia/spectral-checkpoint-selection.git
git push --force origin main
```
Then tell everyone to delete their local clone and clone again. Alternative: make the repo private.
Disha can re-add those notebooks after clearing outputs (Kernel/Runtime -> Clear all outputs).

## 2. Clone (Mac), after Disha has pushed her code
```bash
cd ~/Documents && git clone https://github.com/RuchikaChaurasia/spectral-checkpoint-selection.git
cd spectral-checkpoint-selection
ls src            # must show data/ and train/ (Disha's code). If not, wait for Disha.
```

## 3. Add the spectral part and commit it (Mac)
```bash
git checkout -b ruchika/spectral
unzip -o ~/Downloads/ruchika_full_package.zip -d /tmp/pkg
cp -R /tmp/pkg/spectral-part/. .
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt weightwatcher pytest
python -m pytest -q                      # expect 20 passed
git add configs scripts src tests docs notebooks results CONTRIBUTING.md
git status                               # no .pt, .venv, outputs/, images, __pycache__
git commit -m "Add spectral analysis (Rule W), freeze gate, Real-World evaluation and aggregation"
git push -u origin ruchika/spectral
```
Open a pull request on GitHub, ask Shruti to review `src/eval/`, then merge. Merge before the freeze step.
(Make sure `.gitignore` lists `.venv/`, `__pycache__/`, `*.pt`, `outputs/`.)

## 4. Get the 24 run logs (Colab Part A of notebooks/ruchika_colab.ipynb)
Run Part A, download `runs_logs.zip`, then (Mac):
```bash
git checkout main && git pull
mkdir -p results/runs && unzip -o ~/Downloads/runs_logs.zip -d results/runs
ls results/runs | wc -l                  # 24
```

## 5. Freeze (Mac)
Needs `results/alpha/mean_alpha.csv` from the alpha backup (copy `alpha_results_backup/*` into `results/alpha/`).
```bash
bash scripts/06_select_and_freeze.sh results/runs
head -40 results/selections/frozen_selections.json
git add results/selections results/alpha results/runs
git commit -m "Freeze all checkpoint selections before Real-World scoring"
git tag -a selections-frozen -m "all selections fixed before Real-World scoring"
git push origin main --tags
```
After this, never edit `frozen_selections.json`. If a bug is found, create a new file and say so in the report.

## 6. Evaluate (Colab Part B, GPU)
Run Part B: it clones the repo (with tag), runs the sanity check (must PASS), scores the selected checkpoints on Real World,
optionally `--all-epochs`, builds tables, and zips `eval_results.zip`.

## 7. Commit results (Mac)
```bash
unzip -o ~/Downloads/eval_results.zip -d .
git add results/predictions results/tables results/figures
git commit -m "Add Real-World predictions, tables and figures"
git push origin main
```
Not committed: checkpoints, `outputs/cache`, `outputs/predictions_all_epochs`.
