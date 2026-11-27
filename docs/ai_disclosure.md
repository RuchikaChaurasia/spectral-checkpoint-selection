# AI-assistance disclosure

List every AI tool each member used, what for, and how the output was checked. Keep this accurate; it goes into the final report and the README.

| Member | Tool | Used for | How we checked it |
| --- | --- | --- | --- |
| Disha Rawat | Claude (Anthropic) | Drafted the data and training code in src/data and src/train, and scripts/01-04 (stated in her runbook); helped write the Week 7 reproduction check (Colab notebook and section 4 of docs/reproduction.md), clear dataset photos from notebook outputs and fix references in the dataset card | Read every file; tested offline on a fake dataset (planted duplicates, corrupt file, exact 3,551/444/444 split, bit-identical resume); ran all 24 runs on the real data; split reproduced independently by Sadaf on Kaggle; Shruti's harness re-scored saved checkpoints and matched the training log; reproduction results checked by running the notebook in Colab and comparing files |
| Ruchika Chaurasia | Claude (Anthropic) | Helped write src/spectral (WeightWatcher logging, Rule W, freeze), the evaluation harness in src/eval, their tests, and the run order | Ruchika to complete: read every file; 20 unit tests; Rule W compared with Sadaf's independent implementation on 60,000 random series; harness checked against the training log; alpha results rechecked from the raw files |
| Shruti Naik | TODO | TODO | TODO |
| Sadaf Fatima Syeda | Claude (Anthropic) | Planning tasks; drafting the README, reproduction log, code review, responsible-use and ablation-2 text; writing src/extras/ablation2_patience.py and the Kaggle check commands | Ran every check on Kaggle or locally; 8 rule self-tests; ablation-2 selections matched the frozen file (36/36) and scores matched the team table; read and edited all text |
