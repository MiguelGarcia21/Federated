import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader

from src.data.datasets import (
    VinDrBinaryDataset,
    get_train_transform,
    get_val_transform,
    validate_image_cache,
)
from src.models.cnn import create_model
from src.training.evaluate import evaluate_model, train_one_epoch


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Convert the model state dictionary to Flower-compatible arrays."""
    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_model_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Load Flower parameters into a PyTorch model."""
    model_keys = list(model.state_dict().keys())

    if len(model_keys) != len(parameters):
        raise ValueError(
            "Parameter count mismatch between the model and "
            "the received Flower parameters."
        )

    state_dict = OrderedDict(
        (
            key,
            torch.as_tensor(value),
        )
        for key, value in zip(model_keys, parameters)
    )

    model.load_state_dict(state_dict, strict=True)


def weighted_average(
    metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    """Aggregate numeric client metrics weighted by sample count."""
    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples <= 0:
        raise ValueError(
            "Cannot aggregate Flower metrics with zero examples."
        )

    common_keys = set(metrics[0][1].keys())

    for _, metric_dict in metrics[1:]:
        common_keys.intersection_update(metric_dict.keys())

    aggregated = {}

    for key in sorted(common_keys):
        aggregated[key] = float(
            sum(
                num_examples * float(metric_dict[key])
                for num_examples, metric_dict in metrics
            )
            / total_examples
        )

    return aggregated

def resolve_image_source(
    config: Dict[str, Any],
) -> Tuple[str, Path | None, str]:
    """Resolve and validate the configured image source."""
    image_source = str(
        config.get(
            "image_source",
            "dicom",
        )
    ).strip().lower()

    if image_source not in {"dicom", "cache"}:
        raise ValueError(
            "image_source must be either 'dicom' or 'cache'."
        )

    cache_extension = str(
        config.get(
            "image_cache_extension",
            ".png",
        )
    ).strip().lower()

    cache_dir = None

    if image_source == "cache":
        configured_cache_dir = config.get(
            "image_cache_dir"
        )

        if not configured_cache_dir:
            raise ValueError(
                "image_cache_dir is required when "
                "image_source='cache'."
            )

        cache_dir = Path(
            configured_cache_dir
        )

        if not cache_dir.is_dir():
            raise FileNotFoundError(
                "Configured image cache directory does not "
                f"exist: {cache_dir}"
            )

    return image_source, cache_dir, cache_extension

def make_loader_from_df(
    config: Dict[str, Any],
    df: pd.DataFrame,
    shuffle: bool,
) -> DataLoader:
    """Create a local Flower client DataLoader."""
    image_size = int(config["image_size"])
    image_source, cache_dir, cache_extension = (
        resolve_image_source(config)
    )

    if shuffle:
        transform = get_train_transform(
            image_size=image_size,
            horizontal_flip=bool(
                config.get("horizontal_flip", True)
            ),
            rotation_degrees=float(
                config.get("rotation_degrees", 0.0)
            ),
        )
    else:
        transform = get_val_transform(image_size)

    dataset = VinDrBinaryDataset(
        df=df,
        transform=transform,
        image_source=image_source,
        cache_dir=cache_dir,
        cache_extension=cache_extension,
    )

    num_workers = int(
        config.get("fl_num_workers", 0)
    )

    client_device = str(
        config.get("fl_client_device", "cpu")
    ).lower()

    pin_memory = (
        client_device == "cuda"
        and torch.cuda.is_available()
    )

    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def resolve_client_device(
    config: Dict[str, Any],
) -> torch.device:
    """Resolve the configured Flower client device safely."""
    requested_device = str(
        config.get("fl_client_device", "cpu")
    ).lower()

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "fl_client_device='cuda' was requested, "
                "but CUDA is not available."
            )

        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        "fl_client_device must be either 'cpu' or 'cuda'."
    )

def compute_binary_class_weights(
    df: pd.DataFrame,
    source_name: str,
) -> np.ndarray:
    """Calculate inverse-frequency weights for binary labels."""
    class_counts = (
        df["binary_label"]
        .value_counts()
        .sort_index()
        .reindex([0, 1], fill_value=0)
    )

    if (class_counts <= 0).any():
        raise ValueError(
            f"{source_name} must contain samples from both "
            "binary classes."
        )

    class_weights = (
        len(df)
        / (2.0 * class_counts)
    )

    return class_weights.to_numpy(
        dtype=np.float32
    )

class VinDrFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        client_train_df: pd.DataFrame,
        config: Dict[str, Any],
        global_class_weights: np.ndarray | None = None,
    ):
        self.client_id = client_id
        self.config = config
        self.device = resolve_client_device(config)

        self.model = create_model(
            model_name=config.get("model_name", "resnet18"),
            num_classes=int(config["num_classes"]),
            pretrained=bool(
                config.get("pretrained", False)
            ),
        )

        self.train_loader = make_loader_from_df(
            config=config,
            df=client_train_df,
            shuffle=True,
        )

        local_class_counts = (
            client_train_df["binary_label"]
            .value_counts()
            .sort_index()
            .reindex([0, 1], fill_value=0)
        )

        class_weight_scope = str(
            config.get(
                "class_weight_scope",
                "local",
            )
        ).strip().lower()

        if class_weight_scope == "local":
            selected_class_weights = (
                compute_binary_class_weights(
                    df=client_train_df,
                    source_name=f"Client {client_id}",
                )
            )

        elif class_weight_scope == "global":
            if global_class_weights is None:
                raise ValueError(
                    "global_class_weights must be provided when "
                    "class_weight_scope='global'."
                )

            selected_class_weights = np.asarray(
                global_class_weights,
                dtype=np.float32,
            )

            if selected_class_weights.shape != (2,):
                raise ValueError(
                    "Global class weights must contain exactly two "
                    "values ordered as labels [0, 1]."
                )

            if (
                not np.isfinite(selected_class_weights).all()
                or (selected_class_weights <= 0).any()
            ):
                raise ValueError(
                    "Global class weights must be finite and positive."
                )

        else:
            raise ValueError(
                "class_weight_scope must be either "
                "'local' or 'global'."
            )

        self.class_weight_scope = class_weight_scope

        self.class_weights = torch.tensor(
            selected_class_weights,
            dtype=torch.float32,
            device=self.device,
        )

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(
            self.model,
            parameters,
        )

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights
        )

        learning_rate = float(
            self.config.get(
                "learning_rate",
                self.config.get("fl_lr", 1e-4),
            )
        )

        weight_decay = float(
            self.config.get("weight_decay", 0.01)
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        local_epochs = int(
            config.get(
                "local_epochs",
                self.config.get("fl_local_epochs", 1),
            )
        )

        fit_start_time = time.perf_counter()
        final_train_metrics = None

        for _ in range(local_epochs):
            final_train_metrics = train_one_epoch(
                model=self.model,
                loader=self.train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
            )

        fit_runtime_seconds = (
            time.perf_counter() - fit_start_time
        )

        if final_train_metrics is None:
            raise RuntimeError(
                f"Client {self.client_id} completed no local epochs."
            )

        return (
            get_model_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "train_loss": float(
                    final_train_metrics["loss"]
                ),
                "train_accuracy": float(
                    final_train_metrics["accuracy"]
                ),
                "fit_runtime_seconds": float(
                    fit_runtime_seconds
                ),
            },
        )

def make_centralized_evaluate_fn(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    """Create one server-side evaluation function for the global model."""
    evaluation_device = resolve_client_device(config)

    evaluation_model = create_model(
        model_name=config.get(
            "model_name",
            "resnet18",
        ),
        num_classes=int(config["num_classes"]),
        pretrained=bool(
            config.get("pretrained", False)
        ),
    ).to(evaluation_device)

    val_loader = make_loader_from_df(
        config=config,
        df=val_df,
        shuffle=False,
    )

    global_class_weights = compute_binary_class_weights(
        df=train_df,
        source_name="Complete federated training data",
    )

    class_weights_tensor = torch.tensor(
        global_class_weights,
        dtype=torch.float32,
        device=evaluation_device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "auc",
        "average_precision",
        "balanced_accuracy",
        "negative_predictive_value",
        "false_positive_rate",
        "false_negative_rate",
        "mcc",
        "brier_score",
        "mean_positive_probability",
        "mean_prediction_confidence",
        "positive_prediction_rate",
    ]

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        evaluate_config: Dict[str, Any],
    ):
        del evaluate_config

        # Preserve the previous experiment behavior: validation metrics begin after Round 1 rather than evaluating the untrained model.
        if server_round == 0:
            return None

        set_model_parameters(
            evaluation_model,
            parameters,
        )

        evaluation_model.to(evaluation_device)

        evaluation_start_time = time.perf_counter()

        metrics = evaluate_model(
            model=evaluation_model,
            loader=val_loader,
            criterion=criterion,
            device=evaluation_device,
        )

        evaluation_runtime_seconds = (
            time.perf_counter() - evaluation_start_time
        )

        flower_metrics: Dict[str, float | int] = {}

        for metric_name in metric_names:
            metric_value = float(metrics[metric_name])

            flower_metrics[metric_name] = (
                0.0
                if np.isnan(metric_value)
                else metric_value
            )

        flower_metrics["evaluation_runtime_seconds"] = float(
            evaluation_runtime_seconds
        )

        flower_metrics["num_validation_samples"] = int(
            len(val_loader.dataset)
        )

        validation_loss = float(metrics["loss"])

        print(
            "Centralized evaluation "
            f"Round {server_round}: "
            f"loss={validation_loss:.6f}, "
            f"accuracy={flower_metrics['accuracy']:.6f}, "
            f"runtime={evaluation_runtime_seconds:.2f}s"
        )

        return validation_loss, flower_metrics

    return evaluate_fn


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves selected global checkpoints."""

    def __init__(
        self,
        global_model: nn.Module,
        checkpoint_dir: Path,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.global_model = global_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.save_every = int(
            config.get("save_checkpoints_every", 1)
        )

        self.total_rounds = int(
            config.get(
                "fl_num_rounds",
                config.get("fl_rounds", 3),
            )
        )

        if self.save_every <= 0:
            raise ValueError(
                "save_checkpoints_every must be greater than zero."
            )

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics

        should_save = (
            server_round % self.save_every == 0
            or server_round == self.total_rounds
        )

        if should_save:
            ndarrays = parameters_to_ndarrays(
                aggregated_parameters
            )

            set_model_parameters(
                self.global_model,
                ndarrays,
            )

            checkpoint_path = (
                self.checkpoint_dir
                / f"flower_fedavg_round_{server_round}.pt"
            )

            torch.save(
                {
                    "model_state_dict": (
                        self.global_model.state_dict()
                    ),
                    "config": self.config,
                    "round": server_round,
                    "aggregated_fit_metrics": (
                        aggregated_metrics
                    ),
                },
                checkpoint_path,
            )

            print(
                f"Saved FL checkpoint: {checkpoint_path}"
            )

        return aggregated_parameters, aggregated_metrics


def history_to_dataframe(history) -> pd.DataFrame:
    """Convert Flower history into one row per completed round."""
    fl_metrics_df = pd.DataFrame()

    centralized_losses = getattr(
        history,
        "losses_centralized",
        [],
    )

    distributed_losses = getattr(
        history,
        "losses_distributed",
        [],
    )

    if centralized_losses:
        fl_metrics_df = pd.DataFrame(
            centralized_losses,
            columns=[
                "round",
                "centralized_loss",
            ],
        )
    elif distributed_losses:
        fl_metrics_df = pd.DataFrame(
            distributed_losses,
            columns=[
                "round",
                "distributed_loss",
            ],
        )

    centralized_metrics = getattr(
        history,
        "metrics_centralized",
        {},
    )

    distributed_metrics = getattr(
        history,
        "metrics_distributed",
        {},
    )

    evaluation_metrics = (
        centralized_metrics
        if centralized_metrics
        else distributed_metrics
    )

    for metric_name, values in evaluation_metrics.items():
        metric_df = pd.DataFrame(
            values,
            columns=[
                "round",
                metric_name,
            ],
        )

        fl_metrics_df = (
            metric_df
            if fl_metrics_df.empty
            else fl_metrics_df.merge(
                metric_df,
                on="round",
                how="outer",
            )
        )

    distributed_fit_metrics = getattr(
        history,
        "metrics_distributed_fit",
        {},
    )

    for metric_name, values in distributed_fit_metrics.items():
        metric_df = pd.DataFrame(
            values,
            columns=[
                "round",
                f"fit_{metric_name}",
            ],
        )

        fl_metrics_df = (
            metric_df
            if fl_metrics_df.empty
            else fl_metrics_df.merge(
                metric_df,
                on="round",
                how="outer",
            )
        )

    if not fl_metrics_df.empty:
        fl_metrics_df = (
            fl_metrics_df
            .sort_values("round")
            .reset_index(drop=True)
        )

    return fl_metrics_df


def run_fedavg_debug(
    config: Dict[str, Any],
    client_dfs: List[pd.DataFrame],
    val_df: pd.DataFrame,
    checkpoint_dir: Path,
    metrics_dir: Path,
) -> Tuple[object, Path]:
    """Run the reusable Phase 5/Phase 6 FedAvg simulation.

    The function name remains unchanged for Phase 5 compatibility.
    """
    num_clients = int(
        config.get(
            "fl_num_clients",
            config.get("num_clients", 3),
        )
    )

    fl_rounds = int(
        config.get(
            "fl_num_rounds",
            config.get("fl_rounds", 3),
        )
    )

    local_epochs = int(
        config.get("fl_local_epochs", 1)
    )

    if len(client_dfs) != num_clients:
        raise ValueError(
            f"Expected {num_clients} client DataFrames, "
            f"received {len(client_dfs)}."
        )

    image_source, cache_dir, cache_extension = (
        resolve_image_source(config)
    )

    if image_source == "cache":
        complete_image_df = pd.concat(
            [
                *client_dfs,
                val_df,
            ],
            ignore_index=True,
        )

        if complete_image_df["image_id"].duplicated().any():
            duplicate_ids = (
                complete_image_df.loc[
                    complete_image_df["image_id"].duplicated(
                        keep=False
                    ),
                    "image_id",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicate image IDs were found while validating "
                f"the image cache. Examples: {duplicate_ids[:5]}"
            )

        if cache_dir is None:
            raise RuntimeError(
                "Cache mode was resolved without a cache directory."
            )

        validate_image_cache(
            df=complete_image_df,
            cache_dir=cache_dir,
            cache_extension=cache_extension,
        )

        print("Flower image source: cached PNG")
        print("Flower image cache:", cache_dir)

    else:
        print("Flower image source: original DICOM")

    initial_global_model = create_model(
        model_name=config.get(
            "model_name",
            "resnet18",
        ),
        num_classes=int(config["num_classes"]),
        pretrained=bool(
            config.get("pretrained", False)
        ),
    )

    save_initial_checkpoint = bool(
        config.get("save_initial_checkpoint", False)
    )

    if save_initial_checkpoint:
        initial_checkpoint_path = (
            Path(checkpoint_dir)
            / "flower_fedavg_round_0.pt"
        )

        torch.save(
            {
                "model_state_dict": initial_global_model.state_dict(),
                "config": config,
                "round": 0,
                "aggregated_fit_metrics": {},
            },
            initial_checkpoint_path,
        )

        print(
            "Saved initial FL checkpoint:",
            initial_checkpoint_path,
        )

    initial_parameters = (
        fl.common.ndarrays_to_parameters(
            get_model_parameters(
                initial_global_model
            )
        )
    )

    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "local_epochs": local_epochs,
        }
    
    combined_train_df = pd.concat(
        client_dfs,
        ignore_index=True,
    )

    expected_training_samples = sum(
        len(client_df)
        for client_df in client_dfs
    )

    if len(combined_train_df) != expected_training_samples:
        raise RuntimeError(
            "Could not reconstruct the complete federated "
            "training DataFrame."
        )

    if combined_train_df["image_id"].duplicated().any():
        duplicate_ids = (
            combined_train_df.loc[
                combined_train_df["image_id"].duplicated(
                    keep=False
                ),
                "image_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate image IDs were found in the combined "
            f"federated training data. Examples: {duplicate_ids[:5]}"
        )

    class_weight_scope = str(
        config.get(
            "class_weight_scope",
            "local",
        )
    ).strip().lower()

    if class_weight_scope not in {"local", "global"}:
        raise ValueError(
            "class_weight_scope must be either "
            "'local' or 'global'."
        )

    global_client_class_weights = None

    if class_weight_scope == "global":
        global_client_class_weights = (
            compute_binary_class_weights(
                df=combined_train_df,
                source_name="Complete federated training data",
            )
        )

    print("Client class-weight scope:", class_weight_scope)

    if global_client_class_weights is not None:
        print(
            "Global client class weights:",
            global_client_class_weights.tolist(),
        )

    def client_fn(cid: str):
        client_id = int(cid)

        return VinDrFlowerClient(
            client_id=client_id,
            client_train_df=client_dfs[client_id],
            config=config,
            global_class_weights=global_client_class_weights,
        ).to_client()

    centralized_evaluate_fn = make_centralized_evaluate_fn(
        config=config,
        train_df=combined_train_df,
        val_df=val_df,
    )

    strategy = SaveModelFedAvg(
        global_model=initial_global_model,
        checkpoint_dir=checkpoint_dir,
        config=config,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=centralized_evaluate_fn,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_average,
    )

    client_resources = {
        "num_cpus": float(
            config.get(
                "fl_client_num_cpus",
                1,
            )
        ),
        "num_gpus": float(
            config.get(
                "fl_client_num_gpus",
                0.0,
            )
        ),
    }

    simulation_start_time = time.perf_counter()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(
            num_rounds=fl_rounds
        ),
        strategy=strategy,
        client_resources=client_resources,
    )

    total_runtime_seconds = (
        time.perf_counter() - simulation_start_time
    )

    fl_metrics_df = history_to_dataframe(history)

    if fl_metrics_df.empty:
        raise RuntimeError(
            "Flower completed without producing metric history."
        )

    experiment_id = config["experiment_id"]
    run_type = str(
        config.get(
            "run_type",
            "flower_fedavg_debug",
        )
    )

    fl_metrics_df.insert(
        0,
        "experiment_id",
        experiment_id,
    )

    fl_metrics_df["run_type"] = run_type
    fl_metrics_df["num_clients"] = num_clients
    fl_metrics_df["split_type"] = config.get(
        "split_type",
        "iid",
    )
    fl_metrics_df["seed"] = config["seed"]
    fl_metrics_df["model_name"] = config.get(
        "model_name",
        "resnet18",
    )
    fl_metrics_df["class_weight_scope"] = str(
        config.get(
            "class_weight_scope",
            "local",
        )
    )

    fl_metrics_df["non_iid_skew_strength"] = (
        float(config["non_iid_skew_strength"])
        if str(
            config.get(
                "split_type",
                "iid",
            )
        ).lower() == "non_iid_label_skew"
        else np.nan
    )

    fl_metrics_df["image_source"] = str(
        config.get(
            "image_source",
            "dicom",
        )
    )
    fl_metrics_df["dataset"] = config.get(
        "dataset",
        "vindr",
    )
    fl_metrics_df["task"] = config.get(
        "task",
        "binary_abnormal",
    )
    fl_metrics_df["fl_algorithm"] = config.get(
        "fl_algorithm",
        "fedavg",
    )
    fl_metrics_df["local_epochs"] = local_epochs
    fl_metrics_df["fl_rounds"] = fl_rounds
    fl_metrics_df["batch_size"] = int(
        config["batch_size"]
    )
    fl_metrics_df["learning_rate"] = float(
        config.get(
            "learning_rate",
            config.get("fl_lr", 1e-4),
        )
    )
    fl_metrics_df["total_runtime_seconds"] = (
        total_runtime_seconds
    )
    fl_metrics_df["average_runtime_per_round_seconds"] = (
        total_runtime_seconds / fl_rounds
    )

    metrics_filename = (
        f"{experiment_id}_flower_fedavg_metrics.csv"
    )

    fl_metrics_path = (
        metrics_dir / metrics_filename
    )

    fl_metrics_df.to_csv(
        fl_metrics_path,
        index=False,
    )

    print(
        "Saved Flower metrics to:",
        fl_metrics_path,
    )
    print(
        "Total Flower runtime:",
        f"{total_runtime_seconds:.2f} seconds",
    )
    print(fl_metrics_df)

    return history, fl_metrics_path