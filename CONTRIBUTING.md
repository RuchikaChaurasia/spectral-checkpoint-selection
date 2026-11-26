# How we work in this repository

## Branches and reviews
- `main` always runs. Work on a branch named `<name>/<topic>` (for example `ruchika/rule-w`) and open a pull request, or, for the first uploads, commit to `main` and say so in the message.
- Every pull request is read by its reviewer before merge: Disha's code by Ruchika and Sadaf, Ruchika's code by Shruti, Shruti's code by Disha, Sadaf's code by Disha.
- Say in the pull request what changed, how you checked it, and which config and seed produced any number.
- Commit under your own name. Small commits with clear messages. Never commit work as someone else.

## What never goes in git
- Images or dataset files, checkpoints (`*.pt`), notebooks that still contain outputs (clear them first), Drive links with sharing keys.

## Seeds, configs and results
- Training seeds are 42, 123 and 999. The Product split seed is 0 and is used once. Settings live in `configs/`, never hard-coded.
- Only small outputs go in `results/` (metric summaries, selection files, predictions as label/prediction CSVs, tables, figures).

## The freeze rule (this is what protects the test set)
1. Ruchika runs `scripts/06_select_and_freeze.sh results/runs`. It writes the selected epoch for both rules, every ablation variant (two layer scopes; patience 1, 2, 3) and the linear-probe reference to `results/selections/frozen_selections.json`.
2. She commits the file and creates an annotated tag: `git tag -a selections-frozen -m "all selections fixed before Real-World scoring"`, pushes the tag, and tells the group.
3. Only after the tag exists does anyone run `scripts/07_evaluate.sh` (it checks the tag and that the file is unchanged since it).
4. If a rule or setting must change after the tag, do not edit history. Add an entry to `docs/decisions.md` saying what changed, why, and that Real-World results had been seen; report the original rule as the primary result.
5. The alpha-versus-F1 plot over all checkpoints is made after the tag and is descriptive only.

## Code ownership and AI use
- Write or review code you can explain line by line. If an AI tool helped, note the file and what it did in `docs/ai_disclosure.md`.
- Attribute reused code and models in the README.

## Figures and privacy
- No images that show people in figures, slides or the report.

## Your own records
- Keep your section of the contribution statement up to date as you work: files you wrote, one review you did, one experiment you can explain in depth.
