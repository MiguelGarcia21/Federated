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
from flwr.common import FitIns, parameters_to_ndarrays
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
        if key == "sup007_logical_client_id":
            # Identifier metadata are categorical, not a metric to average.
            continue
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

        # SUP-003 opt-in only. No change for historical configurations.
        gamma_profile = config.get("sup003_gamma_by_client")
        if gamma_profile is not None:
            from torchvision import transforms as T
            from src.data.sup003_heterogeneity import GammaTransform

            if len(gamma_profile) != int(config["fl_num_clients"]):
                raise ValueError("SUP003 gamma profile/client count mismatch")

            original = self.train_loader.dataset.transform
            if not isinstance(original, T.Compose) or not original.transforms:
                raise TypeError("SUP003 expected a Compose training transform")

            # Resize first, then gamma (PIL), then the original augmentations,
            # ToTensor, and ImageNet Normalize. Validation remains unchanged.
            self.train_loader.dataset.transform = T.Compose([
                original.transforms[0],
                GammaTransform(float(gamma_profile[self.client_id])),
                *original.transforms[1:],
            ])

        local_class_counts = (
            client_train_df["binary_label"]
            .value_counts()
            .sort_index()
            .reindex([0, 1], fill_value=0)
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

    def get_properties(self, config):
        # SUP007: identify logical data partitions independently of Flower node IDs.
        if self.config.get("sup007_condition") is None:
            return {}
        return {
            "sup007_logical_client_id": int(self.client_id),
            "sup007_train_samples": int(len(self.train_loader.dataset)),
        }

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
                **(
                    {"sup007_logical_client_id": int(self.client_id)}
                    if self.config.get("sup007_condition") is not None
                    else {}
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

        # SUP-007 audit state: no effect unless sup007_condition is provided.
        self._sup007_expected_by_round = {}
        self._sup007_expected_logical_by_round = {}
        self._sup007_logical_by_proxy = {}
        self._sup007_proxy_by_logical = {}

        if self.save_every <= 0:
            raise ValueError(
                "save_checkpoints_every must be greater than zero."
            )

    
    def configure_fit(self, server_round, parameters, client_manager):
        condition = self.config.get("sup007_condition")

        if condition in {"fedavg_partial", "fedavgm_full"}:
            # The five Flower transport node IDs are NOT the five logical
            # VinDr client partition IDs. Resolve the mapping once via
            # each client's explicit properties, then validate each round.
            if not client_manager.wait_for(5, timeout=300):
                raise RuntimeError("SUP007: five eligible clients unavailable")
            available = client_manager.all()
            if len(available) != 5:
                raise RuntimeError(
                    f"SUP007 expected five Flower nodes, got {len(available)}"
                )

            if not self._sup007_proxy_by_logical:
                from flwr.common import Code, GetPropertiesIns

                by_logical = {}
                by_proxy = {}
                for proxy in available.values():
                    response = proxy.get_properties(
                        GetPropertiesIns(config={}),
                        timeout=90,
                        group_id=server_round,
                    )
                    if response.status.code != Code.OK:
                        raise RuntimeError(
                            f"SUP007 node {proxy.cid} did not return properties"
                        )
                    props = response.properties
                    try:
                        logical_id = str(int(props["sup007_logical_client_id"]))
                        num_samples = int(props["sup007_train_samples"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise RuntimeError(
                            f"SUP007 node {proxy.cid} missing valid partition identity"
                        ) from exc
                    if logical_id not in {"0", "1", "2", "3", "4"}:
                        raise RuntimeError(f"SUP007 invalid logical ID {logical_id}")
                    if num_samples != 2400:
                        raise RuntimeError(
                            f"SUP007 logical client {logical_id}: {num_samples} != 2400"
                        )
                    if logical_id in by_logical or str(proxy.cid) in by_proxy:
                        raise RuntimeError("SUP007 duplicated logical/node identity")
                    by_logical[logical_id] = proxy
                    by_proxy[str(proxy.cid)] = logical_id

                if set(by_logical) != {"0", "1", "2", "3", "4"}:
                    raise RuntimeError("SUP007 cannot map all five data partitions")
                self._sup007_proxy_by_logical = by_logical
                self._sup007_logical_by_proxy = by_proxy
                print("SUP007 verified node->partition map:", by_proxy)

            # A node replacement/remapping invalidates the established schedule.
            if set(available) != set(self._sup007_logical_by_proxy):
                raise RuntimeError("SUP007 Flower node membership changed mid-run")

        if condition == "fedavg_partial":
            seed = int(self.config["seed"])
            order = np.random.default_rng(seed).permutation(5).tolist()
            # In 50 rounds, each logical partition is chosen 30 times.
            selected = [
                str(order[((server_round - 1) * 3 + j) % 5])
                for j in range(3)
            ]
            fit_settings = (
                self.on_fit_config_fn(server_round)
                if self.on_fit_config_fn is not None else {}
            )
            configured = [
                (
                    self._sup007_proxy_by_logical[logical_id],
                    FitIns(parameters, dict(fit_settings)),
                )
                for logical_id in selected
            ]
        else:
            # Historical runs and SUP007 full-client FedAvgM use Flower sampler.
            configured = super().configure_fit(
                server_round, parameters, client_manager
            )

        if condition is not None:
            self._sup007_expected_by_round[int(server_round)] = sorted(
                str(proxy.cid) for proxy, _ in configured
            )
            self._sup007_expected_logical_by_round[int(server_round)] = sorted(
                self._sup007_logical_by_proxy[str(proxy.cid)]
                for proxy, _ in configured
            )
        return configured


    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        # SUP-003 fail-closed participation audit, bypassed for original runs.
        if "sup003_condition" in self.config:
            import json
            from src.data.sup003_heterogeneity import QUANTITY_SIZES

            expected_n = int(self.config["fl_num_clients"])
            if failures or len(results) != expected_n:
                raise RuntimeError(
                    f"SUP003 round {server_round}: incomplete client participation"
                )

            sample_sizes = sorted(int(fit.num_examples) for _, fit in results)
            expected_sizes = sorted(
                [2400] * 5 if self.config["sup003_condition"] == "feature_only"
                else QUANTITY_SIZES
            )

            if sample_sizes != expected_sizes:
                raise RuntimeError(
                    f"SUP003 round {server_round}: sample sizes {sample_sizes}"
                    f" != {expected_sizes}"
                )

            audit_path = self.checkpoint_dir / "sup003_fit_participation.jsonl"
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "round": int(server_round),
                    "samples": sample_sizes,
                    "client_cids": [str(proxy.cid) for proxy, _ in results],
                }) + "\n")

        if self.config.get("sup007_condition") is not None:
            import json

            condition = str(self.config["sup007_condition"])
            n_expected = 3 if condition == "fedavg_partial" else 5
            planned = self._sup007_expected_by_round.get(int(server_round))
            planned_logical = self._sup007_expected_logical_by_round.get(
                int(server_round)
            )
            actual = sorted(str(proxy.cid) for proxy, _ in results)
            samples_by_cid = {
                str(proxy.cid): int(fit.num_examples)
                for proxy, fit in results
            }
            try:
                logical_by_cid = {
                    str(proxy.cid): str(int(fit.metrics["sup007_logical_client_id"]))
                    for proxy, fit in results
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"SUP007 round {server_round}: missing actual logical client IDs"
                ) from exc
            actual_logical = sorted(logical_by_cid.values())
            mapping_ok = all(
                self._sup007_logical_by_proxy.get(cid) == logical_id
                for cid, logical_id in logical_by_cid.items()
            )
            if (
                failures or planned is None or planned_logical is None
                or len(results) != n_expected or actual != planned
                or actual_logical != planned_logical
                or len(set(actual_logical)) != n_expected
                or not mapping_ok or len(samples_by_cid) != n_expected
                or any(n != 2400 for n in samples_by_cid.values())
            ):
                raise RuntimeError(
                    f"SUP007 round {server_round} participation failure: "
                    f"planned_nodes={planned}, actual_nodes={actual}, "
                    f"planned_partitions={planned_logical}, "
                    f"actual_partitions={actual_logical}, "
                    f"samples={samples_by_cid}, failures={len(failures)}"
                )
            total = sum(samples_by_cid.values())
            audit_path = self.checkpoint_dir / "sup007_fit_participation.jsonl"
            with audit_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({
                    "round": int(server_round),
                    "condition": condition,
                    "planned_cids": planned,
                    "actual_cids": actual,
                    "planned_logical_client_ids": planned_logical,
                    "actual_logical_client_ids": actual_logical,
                    "logical_client_by_proxy": logical_by_cid,
                    "samples_by_cid": samples_by_cid,
                    "sample_total": total,
                    "aggregation_weight_by_cid": {
                        cid: n / total for cid, n in samples_by_cid.items()
                    },
                    "num_failures": 0,
                }, sort_keys=True) + "\n")


        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_parameters is None:
            return aggregated_parameters, aggregated_metrics

        # SUP-007 FedAvgM only: keep server momentum on trainable weights,
        # but use ordinary weighted aggregation for non-trainable state.
        if self.config.get("sup007_condition") == "fedavgm_full":
            aggregated_parameters = self._sup007_restore_nontrainable_state(
                server_round=server_round,
                results=results,
                aggregated_parameters=aggregated_parameters,
            )

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


class SaveModelFedAvgM(SaveModelFedAvg, fl.server.strategy.FedAvgM):
    """Opt-in FedAvgM with ordinary aggregation of non-trainable buffers.

    MRO: SaveModelFedAvgM -> SaveModelFedAvg -> FedAvgM -> FedAvg.
    SaveModelFedAvg calls Flower's FedAvgM first. Before saving or
    returning the global parameters, the SUP-007 hook corrects buffers.
    """

    def _sup007_restore_nontrainable_state(
        self,
        server_round,
        results,
        aggregated_parameters,
    ):
        from flwr.common import bytes_to_ndarray, ndarrays_to_parameters

        state_dict = self.global_model.state_dict()
        names = list(state_dict.keys())
        trainable = {
            name for name, parameter in self.global_model.named_parameters()
            if parameter.requires_grad
        }
        arrays = parameters_to_ndarrays(aggregated_parameters)

        if len(arrays) != len(names):
            raise RuntimeError("SUP007 FedAvgM: model state length mismatch")
        if self.momentum_vector is None or len(self.momentum_vector) != len(names):
            raise RuntimeError("SUP007 FedAvgM: missing server momentum state")

        total_examples = sum(int(fit.num_examples) for _, fit in results)
        if total_examples <= 0:
            raise RuntimeError("SUP007 FedAvgM: no valid client samples")
        for _, fit in results:
            if int(fit.num_examples) <= 0 or len(fit.parameters.tensors) != len(names):
                raise RuntimeError("SUP007 FedAvgM: invalid client state")

        for idx, name in enumerate(names):
            reference = state_dict[name].detach().cpu().numpy()
            if arrays[idx].shape != reference.shape:
                raise RuntimeError(f"SUP007 FedAvgM: shape mismatch for {name}")

            if name in trainable:
                if not np.isfinite(arrays[idx]).all():
                    raise RuntimeError(f"SUP007 FedAvgM: nonfinite weight {name}")
                continue

            # Weighted averaging follows the original FedAvg state-dict
            # convention; only trainable weights receive server momentum.
            mean_value = np.zeros(reference.shape, dtype=np.float64)
            for _, fit in results:
                client_value = bytes_to_ndarray(fit.parameters.tensors[idx])
                if client_value.shape != reference.shape:
                    raise RuntimeError(f"SUP007 FedAvgM: client shape mismatch for {name}")
                if not np.isfinite(client_value).all():
                    raise RuntimeError(f"SUP007 FedAvgM: nonfinite client buffer {name}")
                if name.endswith("running_var") and np.any(client_value < 0):
                    raise RuntimeError(f"SUP007 FedAvgM: negative client variance {name}")
                mean_value += (
                    client_value.astype(np.float64)
                    * (int(fit.num_examples) / total_examples)
                )

            if np.issubdtype(reference.dtype, np.integer):
                mean_value = np.rint(mean_value)
            corrected = mean_value.astype(reference.dtype)
            if not np.isfinite(corrected).all():
                raise RuntimeError(f"SUP007 FedAvgM: nonfinite averaged buffer {name}")
            if name.endswith("running_var") and np.any(corrected < 0):
                raise RuntimeError(f"SUP007 FedAvgM: negative averaged variance {name}")

            arrays[idx] = corrected
            # Eliminate momentum history for buffers as well as the
            # current buffer update; preserve trainable momentum unchanged.
            self.momentum_vector[idx] = np.zeros_like(self.momentum_vector[idx])

        # Flower's FedAvgM stores current server state in initial_parameters.
        # Update it before the existing checkpoint logic and next round.
        self.initial_parameters = ndarrays_to_parameters(arrays)
        print(
            f"SUP007 FedAvgM Round {server_round}: "
            "non-trainable buffers averaged without momentum"
        )
        return self.initial_parameters



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

    sup007_condition = config.get("sup007_condition")
    if sup007_condition is None:
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
    else:
        if sup007_condition not in {"fedavgm_full", "fedavg_partial"}:
            raise ValueError("Unknown SUP007 condition")
        if num_clients != 5:
            raise ValueError("SUP007 requires exactly 5 eligible clients")
        partial = sup007_condition == "fedavg_partial"
        strategy_type = SaveModelFedAvg if partial else SaveModelFedAvgM
        strategy_options = dict(
            global_model=initial_global_model,
            checkpoint_dir=checkpoint_dir,
            config=config,
            fraction_fit=0.6 if partial else 1.0,
            fraction_evaluate=0.0,
            min_fit_clients=3 if partial else 5,
            min_available_clients=5,
            accept_failures=False,
            initial_parameters=initial_parameters,
            evaluate_fn=centralized_evaluate_fn,
            on_fit_config_fn=fit_config,
            fit_metrics_aggregation_fn=weighted_average,
        )
        if not partial:
            strategy_options.update(
                server_learning_rate=float(
                    config["sup007_server_learning_rate"]
                ),
                server_momentum=float(config["sup007_server_momentum"]),
            )
        strategy = strategy_type(**strategy_options)
        print(
            "SUP007 strategy:", strategy_type.__name__,
            "active clients:", 3 if partial else 5,
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
    if config.get("sup007_condition") is not None:
        fl_metrics_df["sup007_condition"] = config["sup007_condition"]
        fl_metrics_df["sup007_active_clients"] = int(
            config["sup007_active_clients"]
        )
        fl_metrics_df["sup007_server_momentum"] = float(
            config.get("sup007_server_momentum", 0.0)
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