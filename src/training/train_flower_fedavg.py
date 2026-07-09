from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import parameters_to_ndarrays

from src.data.datasets import (
    VinDrBinaryDataset,
    get_train_transform,
    get_val_transform,
)
from src.models.cnn import create_model
from src.training.evaluate import train_one_epoch, evaluate_model


def get_model_parameters(model: nn.Module):
    return [
        value.cpu().numpy()
        for _, value in model.state_dict().items()
    ]


def set_model_parameters(model: nn.Module, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)

    state_dict = OrderedDict({
        key: torch.tensor(value)
        for key, value in params_dict
    })

    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)

    aggregated = {}

    for key in metrics[0][1].keys():
        aggregated[key] = sum(
            num_examples * metric_dict[key]
            for num_examples, metric_dict in metrics
        ) / total_examples

    return aggregated


def make_loader_from_df(config: Dict[str, Any], df: pd.DataFrame, shuffle: bool):
    image_size = int(config["image_size"])
    transform = get_train_transform(image_size) if shuffle else get_val_transform(image_size)

    dataset = VinDrBinaryDataset(
        df=df,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )

    return loader


class VinDrFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        client_train_df: pd.DataFrame,
        shared_val_df: pd.DataFrame,
        config: Dict[str, Any],
    ):
        self.client_id = client_id
        self.config = config

        self.model = create_model(
            model_name=config.get("model_name", "resnet18"),
            num_classes=int(config["num_classes"]),
        )
        self.device = torch.device("cpu")

        self.train_loader = make_loader_from_df(
            config=config,
            df=client_train_df,
            shuffle=True,
        )

        self.val_loader = make_loader_from_df(
            config=config,
            df=shared_val_df,
            shuffle=False,
        )

        local_class_counts = (
            client_train_df["binary_label"]
            .value_counts()
            .sort_index()
            .reindex([0, 1], fill_value=1)
        )

        local_class_weights = len(client_train_df) / (2.0 * local_class_counts)

        self.class_weights = torch.tensor(
            local_class_weights.values,
            dtype=torch.float32,
        ).to(self.device)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        learning_rate = float(self.config.get("learning_rate", self.config.get("fl_lr", 1e-4)))

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        local_epochs = int(
            config.get(
                "local_epochs",
                self.config.get("fl_local_epochs", 1),
            )
        )

        final_train_metrics = None

        for _ in range(local_epochs):
            final_train_metrics = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
            )

        return (
            get_model_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "train_loss": float(final_train_metrics["loss"]),
                "train_accuracy": float(final_train_metrics["accuracy"]),
            },
        )

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        metrics = evaluate_model(
            model=self.model,
            loader=self.val_loader,
            criterion=criterion,
            device=self.device,
        )

        auc_value = metrics["auc"]
        if np.isnan(auc_value):
            auc_value = 0.0

        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "auc": float(auc_value),
            },
        )


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, global_model, checkpoint_dir, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_model = global_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            set_model_parameters(self.global_model, ndarrays)

            checkpoint_path = (
                self.checkpoint_dir /
                f"flower_fedavg_round_{server_round}.pt"
            )

            torch.save(
                {
                    "model_state_dict": self.global_model.state_dict(),
                    "config": self.config,
                    "round": server_round,
                    "aggregated_metrics": aggregated_metrics,
                },
                checkpoint_path,
            )

            print(f"Saved FL checkpoint: {checkpoint_path}")

        return aggregated_parameters, aggregated_metrics


def history_to_dataframe(history) -> pd.DataFrame:
    fl_metrics_df = pd.DataFrame()

    if len(history.losses_distributed) > 0:
        fl_metrics_df = pd.DataFrame(
            history.losses_distributed,
            columns=["round", "distributed_loss"],
        )

    if hasattr(history, "metrics_distributed"):
        for metric_name, values in history.metrics_distributed.items():
            metric_df = pd.DataFrame(
                values,
                columns=["round", metric_name],
            )

            if fl_metrics_df.empty:
                fl_metrics_df = metric_df
            else:
                fl_metrics_df = fl_metrics_df.merge(
                    metric_df,
                    on="round",
                    how="outer",
                )

    if hasattr(history, "metrics_distributed_fit"):
        for metric_name, values in history.metrics_distributed_fit.items():
            metric_df = pd.DataFrame(
                values,
                columns=["round", f"fit_{metric_name}"],
            )

            if fl_metrics_df.empty:
                fl_metrics_df = metric_df
            else:
                fl_metrics_df = fl_metrics_df.merge(
                    metric_df,
                    on="round",
                    how="outer",
                )

    return fl_metrics_df


def run_fedavg_debug(
    config: Dict[str, Any],
    client_dfs: List[pd.DataFrame],
    val_df: pd.DataFrame,
    checkpoint_dir: Path,
    metrics_dir: Path,
) -> Tuple[object, Path]:
    num_clients = int(config.get("fl_num_clients", config.get("num_clients", 3)))
    fl_rounds = int(config.get("fl_num_rounds", config.get("fl_rounds", 3)))
    local_epochs = int(config.get("fl_local_epochs", 1))

    initial_global_model = create_model(
        model_name=config.get("model_name", "resnet18"),
        num_classes=int(config["num_classes"]),
    )

    initial_parameters = fl.common.ndarrays_to_parameters(
        get_model_parameters(initial_global_model)
    )

    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
        }

    def client_fn(cid: str):
        client_id = int(cid)

        return VinDrFlowerClient(
            client_id=client_id,
            client_train_df=client_dfs[client_id],
            shared_val_df=val_df,
            config=config,
        ).to_client()

    strategy = SaveModelFedAvg(
        global_model=initial_global_model,
        checkpoint_dir=checkpoint_dir,
        config=config,

        fraction_fit=1.0,
        fraction_evaluate=1.0,

        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,

        initial_parameters=initial_parameters,

        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=fl_rounds
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": 0.0,
        },
    )

    fl_metrics_df = history_to_dataframe(history)

    fl_metrics_path = metrics_dir / "flower_fedavg_debug_metrics.csv"
    fl_metrics_df.to_csv(fl_metrics_path, index=False)

    print("Saved Flower metrics to:", fl_metrics_path)
    print(fl_metrics_df)

    return history, fl_metrics_path