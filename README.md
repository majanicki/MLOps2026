# CIFAR-100 classification

Homework 1: train ResNet-18 with PyTorch Lightning, log experiments in MLflow, and tune hyperparameters with Optuna.

## Setup and training

Tested with Python 3.13, CUDA 12.4 and an RTX 3070.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py all --epochs 20 --trials 12 --minutes 52
python summarize.py
```

`all` trains the baseline, runs Optuna, and evaluates the best checkpoint on the test set. The `baseline`, `tune`, and `final` commands run these stages separately. `smoke` checks training and evaluation on two batches. Tuning resumes the existing study up to the requested trial count. Use a new `EXPERIMENT` name when changing the experiment settings.

The time limit reserves two minutes for evaluation. Runs interrupted by the time limit are excluded from selection; runs that stop on a validation plateau are valid completed trials.

To view the logged curves:

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

Open <http://127.0.0.1:5000> and select `cifar100-resnet18-adam-es20`. Databases and checkpoints are local. The exported CSVs and figures are included; regenerating them with `summarize.py` requires the local training records and MLflow database produced by training.

## Experiment

ResNet-18 is trained from scratch with a 3×3 stride-1 first convolution and no initial max pool. The CIFAR-100 training set is split into 45,000 training and 5,000 validation images, stratified by class with seed 42. Test images are used only for evaluation.

Both baseline and trials use Adam, batch size 256, mixed precision, no scheduler, and at most 20 epochs. Adam betas are `(0.9, 0.999)` and epsilon is `1e-8`. Early stopping monitors validation accuracy with patience 4 and `min_delta=0.002` (0.2 percentage points). Each run retains its best validation checkpoint.

| Parameter | Baseline | Optuna search |
| --- | --- | --- |
| Learning rate | `1e-3` | `1e-4–3e-3`, logarithmic |
| Weight decay | `1e-4` | `1e-6–1e-2`, logarithmic |
| Label smoothing | `0.1` | `0–0.2`, uniform |
| Augmentation | None | None, crop/flip, erasing, RandAugment |

The first four trials hold the numerical parameters at their baseline values and vary only augmentation. The remaining eight use TPE to explore the joint search space. All runs share the same early-stopping rule; there is no additional performance-based Optuna pruning.

Augmentation applies only to training images:

- Crop/flip: random crop with padding 4 and horizontal flip.
- Erasing: crop/flip plus random erasing with probability 0.25, area 2–15%, and normalized-zero fill.
- RandAugment: crop/flip plus two RandAugment operations at magnitude 5.

All splits use the same channel normalization. Logged training and validation losses are ordinary cross-entropy; the label-smoothed training objective is logged separately as `train_objective`.

## Results

| Metric | Result |
| --- | --- |
| Baseline validation accuracy | 51.56% |
| Best validation accuracy, trial 9 | 65.12% |
| Improvement | +13.56 percentage points (+26.30% relative) |
| Test accuracy | 64.01% |
| Test macro-F1 | 0.6371 |

The selected configuration uses erasing, learning rate `0.001567778`, weight decay `1.867588e-6`, and label smoothing `0.058188874`. All 12 trials completed. The baseline and trial 0 stopped at epoch 14, trial 7 at epoch 13, and the other trials at epoch 20.

At the fixed baseline parameters:

| Augmentation | Best validation accuracy | Epochs |
| --- | --- | --- |
| None | 51.56% | 14 |
| Crop/flip | 64.90% | 20 |
| Crop/flip + erasing | 63.58% | 20 |
| Crop/flip + RandAugment | 62.98% | 20 |

Crop/flip was best in the controlled comparison. The joint search selected erasing with different numerical settings. These are single-seed results; validation was used for selection and the test score is a separate evaluation.

[Learning curves](results/adam20/learning_curves.png) · [Optuna trials](results/adam20/tuning_trials.png) · [Augmentation comparison](results/adam20/augmentation_comparison.png) · [Trial parameters](results/adam20/trials.csv) · [Results and configuration](results/adam20/comparison.json)
