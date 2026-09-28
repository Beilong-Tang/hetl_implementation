# Deep-learning reproduction rules

The deep-learning results use the same experimental protocol as the first HeTL reproduction. The original baseline and HeTL files in the parent experiment remain unchanged.

## Data and splits

- Dataset: NSL-KDD `KDDTrain+` processed by the parent experiment.
- Tasks: DoS → R2L, DoS → Probe, and Probe → R2L.
- Each domain uses 995 attack records and 995 normal records.
- Categorical features use the shared one-hot vocabulary.
- `difficulty` is removed.
- Source and target domains are scaled separately with Min-Max scaling.
- The same ten seeds (`42`–`51`) and the same saved NPZ splits are used.
- The target domain has 500 validation records and 1490 final test records.

## Training and evaluation

- Models train on the labeled source domain only.
- Target validation labels are used only for early stopping/model selection.
- Target test labels are used only once for final evaluation.
- The reported positive class is `attack=1`.
- Every model reports Accuracy, Precision, Recall, attack F1, and ROC-AUC.
- Summary values are mean ± standard deviation over the ten seeds.

## Comparison summary

`outputs/tables/deep_learning_summary.csv` contains the unchanged parent `No TL` and `HeTL` summaries together with the new deep-learning rows. `deep_learning_only_summary.csv` contains only the deep-learning rows.

The deep-learning models are source-only classifiers. They are therefore comparable to `No TL`; they do not claim to reproduce the HeTL representation-learning algorithm. The combined plots label the three approaches explicitly.
