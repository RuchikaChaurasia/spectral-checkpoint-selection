# AI-assistance disclosure

List every AI tool each member used, what for, and how the output was checked. Keep this accurate; it goes into the final report and the README.

| Member | Tool | Used for | How we checked it |
| --- | --- | --- | --- |
| Disha Rawat | Claude (Anthropic) | Drafted the data and training code in src/data and src/train (stated in her runbook) | Disha to complete: reviewed every file, tested offline, ran on the real data |
| Ruchika Chaurasia | Claude (Anthropic) | Helped write src/spectral (WeightWatcher logging, Rule W, freeze), the evaluation harness in src/eval, their tests, and the run order | Ruchika to complete: read every file; 20 unit tests; Rule W compared with Sadaf's independent implementation on 60,000 random series; harness checked against the training log; alpha results rechecked from the raw files |
| Shruti Naik | TODO | TODO | TODO |
| Sadaf Fatima Syeda | Claude (Anthropic) | Planning tasks; drafting the README, reproduction log, code review, responsible-use and ablation-2 text; writing src/extras/ablation2_patience.py and the Kaggle check commands | Ran every check on Kaggle or locally; 8 rule self-tests; ablation-2 selections matched the frozen file (36/36) and scores matched the team table; read and edited all text |
