# Deep Learning Experiments

This directory is reserved for deep-learning baselines on the same NSL-KDD transfer-learning tasks used in the parent experiment:

- DoS → R2L
- DoS → Probe
- Probe → R2L

The existing `first_experiment_review` directory is not modified by this experiment. Results should be written only to `outputs/` and model weights to `checkpoints/`.

## Candidate methods

1. **MLP / tabular ResNet** — the main ResNet-style baseline. Residual fully connected blocks are more natural for tabular features than an image ResNet.
2. **1D CNN** — treats the fixed feature vector as a one-dimensional sequence.
3. **GRU or LSTM** — tests whether sequential modeling helps, although the tabular feature order has no intrinsic temporal meaning.
4. **Tabular Transformer** — attention-based model for feature interactions; use a small version to control overfitting.

The first model to implement should be the tabular ResNet, followed by the 1D CNN. LSTM/GRU and Transformer are useful comparison models, but their inductive biases are less well matched to this dataset.

## Fair comparison protocol

- Reuse the parent preprocessing and task splits whenever possible.
- Keep the target validation/test separation: use the 500 validation rows for early stopping and model selection, and report final metrics only on the 1490 test rows.
- Use the same ten seeds (`42`–`51`) as the first experiment.
- Report Accuracy, attack Precision, attack Recall, attack F1, ROC-AUC, mean ± standard deviation, and per-seed results.
- Do not select checkpoints using the target test labels.
- Clearly label whether an experiment is transductive (target features available during training) or inductive.

## Suggested initial settings

- Optimizer: AdamW
- Learning rate: `1e-3` with ReduceLROnPlateau or cosine decay
- Batch size: `128`
- Maximum epochs: `100`
- Early stopping patience: `15`
- Loss: weighted binary cross-entropy or focal loss, with class weights computed from the source training split only
- Regularization: dropout `0.2`, weight decay `1e-4`

See `configs/experiment.json` for the proposed experiment matrix.
