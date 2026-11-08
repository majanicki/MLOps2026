"""Export metrics and figures from the local MLflow experiment."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mlflow.tracking import MlflowClient


def augmentation_comparison(folder, client):
    path = folder / "augmentation_results.json"
    if not path.exists():
        return []
    runs = [r for r in json.loads(path.read_text()) if r["completed"]]
    for run in runs:
        assert run["config"] == runs[0]["config"], (
            "Controlled runs must differ only in augmentation"
        )
        for key in ("epochs", "seed", "split_hash", "precision"):
            assert run["protocol"][key] == runs[0]["protocol"][key]
    reference = {r["protocol"]["augmentation"]: r["best_val_acc"] for r in runs}
    rows = [dict(augmentation=r["protocol"]["augmentation"], best_val_acc=r["best_val_acc"],
                 gain_over_none_pp=100 * (r["best_val_acc"] - reference["none"]),
                 gain_over_basic_pp=100 * (
                     r["best_val_acc"] - reference.get("basic", r["best_val_acc"])
                 ),
                 epochs_run=r["epochs_run"],
                 stop_reason=r["stop_reason"], run_id=r["run_id"]) for r in runs]
    (folder / "augmentation_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (folder / "augmentation_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = {"none": "None", "basic": "Crop/flip", "erasing": "Crop/flip +\nerasing",
              "randaugment": "Crop/flip +\nRandAugment"}
    for i, result in enumerate(runs):
        augmentation = result["protocol"]["augmentation"]
        color = plt.get_cmap("tab10")(i)
        bar = axes[0].bar(labels[augmentation], 100 * result["best_val_acc"], color=color)
        axes[0].bar_label(bar, labels=[f"{result['best_val_acc']:.2%}"], padding=3, fontsize=9)
        epochs = {
            m.step: int(m.value) + 1
            for m in client.get_metric_history(result["run_id"], "epoch")
        }
        history = client.get_metric_history(result["run_id"], "val_acc")
        axes[1].plot([epochs[m.step] for m in history], [100*m.value for m in history],
                     label=labels[augmentation].replace("\n", " "), color=color)
    axes[0].set(ylabel="Best validation accuracy (%)", ylim=(0, 100))
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Augmentation comparison")
    fig.tight_layout()
    fig.savefig(folder / "augmentation_comparison.png", dpi=180)
    plt.close(fig)
    return rows


def summarize(folder, tracking_uri):
    folder = Path(folder)
    baseline = json.loads((folder / "baseline.json").read_text())
    tuned = json.loads((folder / "tuned.json").read_text())
    shared_settings = (
        "architecture", "epochs", "seed", "optimizer", "batch_size", "scheduler",
        "precision", "split_hash", "train_size", "val_size",
        "early_stopping_patience", "early_stopping_min_delta", "early_stopping_monitor",
    )
    for key in shared_settings:
        if baseline["protocol"][key] != tuned["protocol"][key]:
            raise ValueError(f"Baseline/tuned protocol mismatch: {key}")
    best_params = json.loads((folder / "best_params.json").read_text())
    if tuned["config"] != best_params:
        raise ValueError(
            "Run train.py final: selected parameters changed after the saved final run"
        )
    with (folder / "trials.csv").open(newline="") as handle:
        trials = list(csv.DictReader(handle))
    best_trial = max(
        (r for r in trials if r["state"] == "COMPLETE"),
        key=lambda r: float(r["value"]),
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    histories = {}
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey="row")
    for column, result in enumerate((baseline, tuned)):
        run_id = result["run_id"]
        epochs = {m.step: int(m.value) + 1 for m in client.get_metric_history(run_id, "epoch")}
        histories[result["name"]] = {}
        for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
            history = client.get_metric_history(run_id, key)
            values = [(epochs[m.step], m.value) for m in history]
            histories[result["name"]][key] = values
            rows.extend(dict(run=result["name"], epoch=e, metric=key, value=v) for e, v in values)
            axis = axes[0 if key.endswith("loss") else 1, column]
            axis.plot([e for e, _ in values], [v for _, v in values],
                      label="Training" if key.startswith("train") else "Validation")
        axes[0, column].set_title(
            result["name"].capitalize() + " · " + result["protocol"]["augmentation"]
        )
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend()
        axes[1, column].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[1, 0].set_ylabel("Top-1 accuracy")
    fig.suptitle("CIFAR-100: baseline and tuned model")
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=180)
    fig.savefig(folder / "learning_curves.pdf")
    plt.close(fig)
    with (folder / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "epoch", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    fig, axis = plt.subplots(figsize=(10, 4))
    plotted = [r for r in trials if r["value"]]
    axis.bar([int(r["number"]) for r in plotted], [float(r["value"]) for r in plotted],
             color=["tab:blue" if r["state"] == "COMPLETE" else "0.65" for r in plotted])
    base_line = axis.axhline(
        baseline["best_val_acc"], color="black", linestyle="--", label="Baseline"
    )
    target_line = axis.axhline(baseline["best_val_acc"] + 0.03, color="tab:green", linestyle=":",
                               label="Baseline + 3 percentage points")
    legend = [Patch(color="tab:blue", label="Best validation accuracy")]
    if any(row["state"] != "COMPLETE" for row in plotted):
        legend.append(Patch(color="0.65", label="Pruned: intermediate accuracy"))
    axis.legend(
        handles=legend + [base_line, target_line],
        loc="upper left", bbox_to_anchor=(1.01, 1),
    )
    axis.set(xlabel="Trial number", ylabel="Validation top-1 accuracy", ylim=(0, 0.85),
             title="Optuna search results")
    axis.set_xticks([int(r["number"]) for r in plotted])
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "tuning_trials.png", dpi=180)
    plt.close(fig)
    delta = tuned["best_val_acc"] - baseline["best_val_acc"]
    summary = dict(baseline_val_acc=baseline["best_val_acc"], tuned_val_acc=tuned["best_val_acc"],
                   improvement_percentage_points=100 * delta,
                   improvement_relative_percent=100 * delta / baseline["best_val_acc"],
                   meets_conservative_3pp_target=delta >= 0.03,
                   meets_relative_3percent_target=delta / baseline["best_val_acc"] >= 0.03,
                   best_trial=int(best_trial["number"]),
                   trial_counts=dict(Counter(r["state"] for r in trials)),
                   test=tuned["test"],
                   baseline_run_id=baseline["run_id"], tuned_run_id=tuned["run_id"],
                   baseline_epochs_run=baseline["epochs_run"],
                   tuned_epochs_run=tuned["epochs_run"],
                   baseline_stop_reason=baseline["stop_reason"],
                   tuned_stop_reason=tuned["stop_reason"],
                   baseline_config=baseline["config"], best_config=tuned["config"],
                   protocol={key: value for key, value in tuned["protocol"].items()
                             if key not in ("source_sha256", "augment")},
                   augmentation_comparison=augmentation_comparison(folder, client))
    (folder / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Validation: {baseline['best_val_acc']:.2%} → {tuned['best_val_acc']:.2%} "
        f"({100 * delta:+.2f} pp); test: {tuned['test']['test_acc']:.2%}"
    )
    for result in (baseline, tuned):
        logged = histories[result["name"]]
        assert all(len(logged[key]) == result["epochs_run"] for key in logged)
        assert max(v for _, v in logged["val_acc"]) == result["best_val_acc"]
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/adam20")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    summarize(args.results, args.tracking_uri)
