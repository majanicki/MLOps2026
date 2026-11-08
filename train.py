"""Train and tune a CIFAR-100 classifier with Lightning and Optuna."""

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning as L
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.models import resnet18

SEED = 42
EXPERIMENT = "cifar100-resnet18-adam-es20"
RESULTS = Path("results/adam20")
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
BASELINE = dict(
    optimizer="adam", lr=1e-3, weight_decay=1e-4,
    label_smoothing=0.1, batch_size=256,
)
EARLY_STOPPING = dict(monitor="val_acc", mode="max", min_delta=0.002, patience=4)
AUGMENTATIONS = ["none", "basic", "erasing", "randaugment"]


class CIFAR100Dataset(Dataset):
    def __init__(self, data, indices, transform):
        self.data, self.indices, self.transform = data, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.data[self.indices[index]]
        return self.transform(image), label


class CIFAR100DataModule(L.LightningDataModule):
    def __init__(self, root="data", batch_size=256, num_workers=4, augmentation="basic"):
        super().__init__()
        self.save_hyperparameters()
        operations = []
        if augmentation != "none":
            operations = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        if augmentation == "randaugment":
            operations.append(transforms.RandAugment(num_ops=2, magnitude=5))
        operations += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        if augmentation == "erasing":
            operations.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0))
        self.train_tf = transforms.Compose(operations)
        self.eval_tf = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])

    def prepare_data(self):
        for train in (True, False):
            CIFAR100(self.hparams.root, train=train, download=True)

    def setup(self, stage=None):
        if hasattr(self, "train_ds"):
            return
        data = CIFAR100(self.hparams.root, train=True, download=False)
        generator = torch.Generator().manual_seed(SEED)
        labels = torch.tensor(data.targets)
        train_idx, val_idx = [], []
        for label in range(100):
            indices = torch.where(labels == label)[0]
            indices = indices[torch.randperm(len(indices), generator=generator)].tolist()
            val_idx.extend(indices[:50])
            train_idx.extend(indices[50:])
        self.split_hash = hashlib.sha256(json.dumps([train_idx, val_idx]).encode()).hexdigest()
        self.train_ds = CIFAR100Dataset(data, train_idx, self.train_tf)
        self.val_ds = CIFAR100Dataset(data, val_idx, self.eval_tf)
        test = CIFAR100(self.hparams.root, train=False, download=False)
        self.test_ds = CIFAR100Dataset(test, range(len(test)), self.eval_tf)

    def loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.hparams.num_workers > 0,
            generator=torch.Generator().manual_seed(SEED),
        )

    def train_dataloader(self):
        return self.loader(self.train_ds, True)

    def val_dataloader(self):
        return self.loader(self.val_ds, False)

    def test_dataloader(self):
        return self.loader(self.test_ds, False)


class LitResnet18(L.LightningModule):
    def __init__(self, optimizer="adam", lr=1e-3, weight_decay=1e-4, label_smoothing=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(weights=None, num_classes=100)
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.to(memory_format=torch.channels_last)
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_acc", torchmetrics.classification.MulticlassAccuracy(
                num_classes=100, average="micro"))
        self.test_f1 = torchmetrics.classification.MulticlassF1Score(
            num_classes=100, average="macro",
        )

    def forward(self, x):
        return self.model(x.contiguous(memory_format=torch.channels_last))

    def step(self, batch, split):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        metric = getattr(self, f"{split}_acc")
        metric.update(logits, y)
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        self.log(f"{split}_acc", metric, on_step=False, on_epoch=True)
        if split == "test":
            self.test_f1.update(logits, y)
            self.log("test_macro_f1", self.test_f1, on_step=False, on_epoch=True)
        if split == "train":
            objective = F.cross_entropy(
                logits, y, label_smoothing=self.hparams.label_smoothing,
            )
            self.log("train_objective", objective, on_step=False, on_epoch=True, batch_size=len(y))
            return objective
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        if self.hparams.optimizer != "adam":
            raise ValueError("This experiment uses Adam for both baseline and trials")
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class Progress(L.Callback):
    def __init__(self, trial=None):
        self.trial = trial
        self.epochs_run = 0

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.epochs_run += 1
        accuracy = float(trainer.callback_metrics["val_acc"])
        print(f"epoch={trainer.current_epoch + 1} val_acc={accuracy:.4f}", flush=True)
        if self.trial is not None:
            self.trial.report(accuracy, trainer.current_epoch)


def run(
    config, name, epochs, trial=None, test=False, smoke=False,
    augmentation="basic", max_seconds=None,
):
    if not 1 <= epochs <= 20:
        raise ValueError("Every training run must use at most 20 epochs")
    gc.collect()  # Release the previous trial and its workers.
    L.seed_everything(SEED, workers=True)
    torch.set_num_threads(4)
    dm = CIFAR100DataModule(batch_size=config["batch_size"], augmentation=augmentation)
    dm.prepare_data()
    dm.setup()
    model = LitResnet18(**{
        key: value for key, value in config.items() if key != "batch_size"
    })
    output = Path("results/smoke") if smoke else RESULTS
    output.mkdir(parents=True, exist_ok=True)
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"
    protocol = dict(
        config,
        epochs=epochs,
        seed=SEED,
        architecture="resnet18",
        precision=precision,
        augmentation=augmentation,
        split_hash=dm.split_hash,
        train_size=45000,
        val_size=5000,
        scheduler=False,
        early_stopping_patience=EARLY_STOPPING["patience"],
        early_stopping_min_delta=EARLY_STOPPING["min_delta"],
        early_stopping_monitor=EARLY_STOPPING["monitor"],
    )
    logger = MLFlowLogger(
        experiment_name=EXPERIMENT + ("-smoke" if smoke else ""),
        run_name=name,
        tracking_uri="sqlite:///mlflow.db",
    )
    logger.log_hyperparams(protocol)
    if trial is not None:
        trial.set_user_attr("mlflow_run_id", logger.run_id)
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints" / name),
        monitor="val_acc", mode="max", save_top_k=1, save_last=True,
    )
    progress = Progress(trial)
    early_stopping = EarlyStopping(
        **EARLY_STOPPING, check_on_train_epoch_end=False, verbose=True,
    )
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        deterministic=True,
        logger=logger,
        callbacks=[progress, early_stopping, checkpoint],
        max_time=timedelta(seconds=max_seconds) if max_seconds is not None else None,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=2 if smoke else 1.0,
        limit_val_batches=2 if smoke else 1.0,
        limit_test_batches=2 if smoke else 1.0,
    )
    trainer.fit(model, datamodule=dm)
    if trainer.interrupted:
        raise KeyboardInterrupt("Training interrupted")

    reached_max = trainer.global_step >= epochs * int(trainer.num_training_batches)
    plateau = early_stopping.stopped_epoch > 0
    if plateau:
        stop_reason = "plateau"
    elif reached_max:
        stop_reason = "max_epochs"
    else:
        stop_reason = "time_limit"
    result = dict(
        name=name,
        run_id=logger.run_id,
        config=config,
        protocol=protocol,
        best_val_acc=float(checkpoint.best_model_score),
        last_val_acc=float(trainer.callback_metrics["val_acc"]),
        checkpoint=checkpoint.best_model_path,
        epochs_run=progress.epochs_run,
        stop_reason=stop_reason,
        completed=reached_max or plateau,
    )
    if trial is not None:
        trial.set_user_attr("epochs_run", result["epochs_run"])
        trial.set_user_attr("stop_reason", stop_reason)
    if test:
        result["test"] = trainer.test(model, datamodule=dm, ckpt_path="best")[0]
    logger.finalize("success" if result["completed"] else "KILLED")
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def get_study(epochs):
    study = optuna.create_study(
        study_name=EXPERIMENT,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4),
        pruner=optuna.pruners.NopPruner(),
        storage="sqlite:///optuna.db",
        load_if_exists=True,
    )
    protocol = dict(
        epochs=epochs, seed=SEED, architecture="resnet18", augmentations=AUGMENTATIONS,
        optimizer="adam", batch_size=BASELINE["batch_size"],
        early_stopping=EARLY_STOPPING,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
    )
    if study.trials and study.user_attrs.get("protocol") != protocol:
        raise ValueError("Existing study uses a different protocol; choose another EXPERIMENT name")
    study.set_user_attr("protocol", protocol)
    if not study.trials:
        baseline_params = {
            name: BASELINE[name] for name in ("lr", "weight_decay", "label_smoothing")
        }
        for augmentation in AUGMENTATIONS:
            study.enqueue_trial(dict(baseline_params, augmentation=augmentation))
    return study


def tune(epochs, trials, deadline):
    study = get_study(epochs)

    def objective(trial):
        augmentation = trial.suggest_categorical("augmentation", AUGMENTATIONS)
        config = dict(
            optimizer="adam", batch_size=BASELINE["batch_size"],
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
        )
        if deadline - time.monotonic() < 180:
            raise optuna.TrialPruned("Time budget: no new training run started")
        result = run(
            config, f"trial_{trial.number}", epochs, trial=trial,
            augmentation=augmentation,
            max_seconds=max(1, deadline - time.monotonic() - 120),
        )
        if not result["completed"]:
            raise optuna.TrialPruned("Time cap: incomplete run excluded from selection")
        return result["best_val_acc"]

    try:
        finished = sum(trial.state.is_finished() for trial in study.trials)
        study.optimize(objective, n_trials=max(0, trials - finished))
    finally:
        study.trials_dataframe().to_csv(RESULTS / "trials.csv", index=False)
    return study


def evaluate_best(epochs):
    study = get_study(epochs)
    winner = json.loads((RESULTS / f"trial_{study.best_trial.number}.json").read_text())
    assert winner["completed"]
    gc.collect()
    torch.set_num_threads(4)
    dm = CIFAR100DataModule(batch_size=winner["config"]["batch_size"],
                            augmentation=winner["protocol"]["augmentation"])
    model = LitResnet18.load_from_checkpoint(winner["checkpoint"], map_location="cpu")
    logger = MLFlowLogger(experiment_name=EXPERIMENT, tracking_uri="sqlite:///mlflow.db",
                          run_id=winner["run_id"])
    trainer = L.Trainer(accelerator="auto", devices=1, precision=winner["protocol"]["precision"],
                        logger=logger, enable_progress_bar=False, enable_checkpointing=False)
    winner = dict(winner, name="tuned")
    winner["test"] = trainer.test(model, datamodule=dm)[0]
    logger.finalize("success")
    (RESULTS / "tuned.json").write_text(json.dumps(winner, indent=2) + "\n")
    (RESULTS / "best_params.json").write_text(json.dumps(winner["config"], indent=2) + "\n")
    controlled = [json.loads((RESULTS / f"trial_{number}.json").read_text())
                  for number in range(4) if (RESULTS / f"trial_{number}.json").exists()]
    (RESULTS / "augmentation_results.json").write_text(json.dumps(controlled, indent=2) + "\n")
    print("Selected parameters:", study.best_params, flush=True)
    print("Selected validation accuracy:", winner["best_val_acc"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["baseline", "tune", "final", "all", "smoke"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument(
        "--minutes", type=float, default=52,
        help="Time cap, including two minutes reserved for evaluation",
    )
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or args.trials < 1 or args.minutes <= 0:
        parser.error("epochs must be 1–20; trials and minutes must be positive")
    RESULTS.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.minutes * 60
    if args.command == "smoke":
        run(BASELINE, "smoke", 1, test=True, smoke=True, augmentation="none")
    if args.command in ("baseline", "all"):
        result = run(BASELINE, "baseline", args.epochs, augmentation="none",
                     max_seconds=max(1, deadline - time.monotonic() - 120))
        if not result["completed"]:
            raise RuntimeError("Baseline did not finish the requested epoch budget")
    if args.command in ("tune", "all"):
        tune(args.epochs, args.trials, deadline)
    if args.command in ("final", "all"):
        evaluate_best(args.epochs)


if __name__ == "__main__":
    main()
