# MLflow screenshots

Captured from the local MLflow 3.17.2 interface for `cifar100-resnet18-adam-es20`.

- [Experiment runs](01-experiment-runs.png): baseline and all 12 Optuna trials.
- [Baseline learning curves](02-baseline-learning-curves.png): training and validation metrics.
- [Selected trial learning curves](03-selected-trial-learning-curves.png): trial 9, including test metrics.
- [Selected parameters and test results](04-selected-trial-parameters-and-test.png).
- [Baseline versus selected parameters](05-baseline-selected-comparison.png).

MLflow tables show the last logged validation accuracy: 50.32% for the baseline and 63.40% for trial 9. Their best checkpoint accuracies are 51.56% and 65.12%, respectively. Test accuracy is 64.01% and comes from the selected best checkpoint. Curve x-axes show optimizer steps.
