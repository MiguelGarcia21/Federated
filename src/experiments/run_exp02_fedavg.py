import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.datasets import normalize_cache_extension
from src.data.preprocess_vindr import prepare_vindr_full_data
from src.data.split_clients import (
    create_iid_client_splits,
    create_label_skew_client_splits,
)
from src.training.train_flower_fedavg import run_fedavg_debug
from src.utils.paths import (
    create_experiment_dirs,
    print_output_check,
    save_config_snapshot,
)
from src.utils.seed import set_seed


SUPPORTED_SPLIT_TYPES = {
    "iid",
    "non_iid_label_skew",
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the top-level YAML structure."""
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the required settings for full FedAvg experiments."""
    required_keys = [
        "experiment_id",
        "project_root",
        "output_dir",
        "train_csv",
        "image_source",
        "train_image_dir",
        "dataset",
        "task",
        "image_size",
        "num_classes",
        "batch_size",
        "learning_rate",
        "seed",
        "val_fraction",
        "fl_algorithm",
        "fl_num_clients",
        "fl_num_rounds",
        "fl_local_epochs",
        "split_type",
        "save_checkpoints_every",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Missing required configuration keys: "
            + ", ".join(missing_keys)
        )

    experiment_id = str(config["experiment_id"]).strip()

    if not experiment_id:
        raise ValueError("experiment_id cannot be empty.")

    if Path(experiment_id).name != experiment_id:
        raise ValueError(
            "experiment_id must be a plain identifier without "
            "directory components."
        )

    train_csv = Path(config["train_csv"])
    train_image_dir = Path(config["train_image_dir"])

    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Configured train_csv does not exist: {train_csv}"
        )

    if not train_image_dir.is_dir():
        raise FileNotFoundError(
            "Configured train_image_dir does not exist or is not "
            f"a directory: {train_image_dir}"
        )
    
    image_source = str(
        config["image_source"]
    ).strip().lower()

    if image_source not in {"dicom", "cache"}:
        raise ValueError(
            "image_source must be either 'dicom' or 'cache'."
        )

    if image_source == "cache":
        configured_cache_dir = config.get(
            "image_cache_dir"
        )

        if (
            configured_cache_dir is None
            or not str(configured_cache_dir).strip()
        ):
            raise ValueError(
                "image_cache_dir is required when "
                "image_source='cache'."
            )

        cache_dir = Path(
            configured_cache_dir
        )

        if not cache_dir.is_dir():
            raise FileNotFoundError(
                "Configured image cache directory does not exist: "
                f"{cache_dir}"
            )

        cache_extension = normalize_cache_extension(
            config.get(
                "image_cache_extension",
                ".png",
            )
        )

        cache_num_workers = int(
            config.get(
                "cache_num_workers",
                8,
            )
        )

        if cache_num_workers <= 0:
            raise ValueError(
                "cache_num_workers must be greater than zero."
            )

    else:
        cache_dir = None

        cache_extension = normalize_cache_extension(
            config.get(
                "image_cache_extension",
                ".png",
            )
        )

    if str(config["dataset"]).lower() != "vindr":
        raise ValueError(
            "The current federated runner supports only dataset='vindr'."
        )

    if str(config["task"]).lower() != "binary_abnormal":
        raise ValueError(
            "The current federated runner supports only "
            "task='binary_abnormal'."
        )

    if str(config.get("model_name", "resnet18")).lower() != "resnet18":
        raise ValueError(
            "The current federated runner supports only ResNet-18."
        )

    if int(config["num_classes"]) != 2:
        raise ValueError(
            "Binary VinDr classification requires num_classes=2."
        )

    if str(config["fl_algorithm"]).lower() != "fedavg":
        raise ValueError(
            "The current federated runner supports only FedAvg."
        )

    split_type = str(config["split_type"]).lower()

    if split_type not in SUPPORTED_SPLIT_TYPES:
        raise ValueError(
            f"Unsupported split_type={split_type!r}. "
            f"Supported values: {sorted(SUPPORTED_SPLIT_TYPES)}"
        )
    
    class_weight_scope = str(
        config.get(
            "class_weight_scope",
            "local",
        )
    ).strip().lower()

    if class_weight_scope not in {
        "local",
        "global",
    }:
        raise ValueError(
            "class_weight_scope must be either "
            "'local' or 'global'."
        )

    if int(config["fl_num_clients"]) < 2:
        raise ValueError(
            "fl_num_clients must be at least 2."
        )

    if int(config["fl_num_rounds"]) <= 0:
        raise ValueError(
            "fl_num_rounds must be greater than zero."
        )

    if int(config["fl_local_epochs"]) <= 0:
        raise ValueError(
            "fl_local_epochs must be greater than zero."
        )

    if int(config["save_checkpoints_every"]) <= 0:
        raise ValueError(
            "save_checkpoints_every must be greater than zero."
        )

    if int(config["batch_size"]) <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if float(config["learning_rate"]) <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    val_fraction = float(config["val_fraction"])

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be between 0 and 1, exclusive."
        )

    requested_device = str(
        config.get("fl_client_device", "cpu")
    ).lower()

    if requested_device not in {"cpu", "cuda"}:
        raise ValueError(
            "fl_client_device must be either 'cpu' or 'cuda'."
        )

    configured_gpu_resource = float(
        config.get("fl_client_num_gpus", 0.0)
    )

    if requested_device == "cuda" and configured_gpu_resource <= 0.0:
        raise ValueError(
            "fl_client_device='cuda' requires "
            "fl_client_num_gpus greater than zero."
        )

    if requested_device == "cpu" and configured_gpu_resource > 0.0:
        raise ValueError(
            "fl_client_num_gpus is greater than zero while "
            "fl_client_device='cpu'."
        )

    if split_type == "non_iid_label_skew":
        if "non_iid_skew_strength" not in config:
            raise KeyError(
                "non_iid_label_skew requires "
                "non_iid_skew_strength."
            )

        skew_strength = float(
            config["non_iid_skew_strength"]
        )

        if skew_strength < 1.0:
            raise ValueError(
                "non_iid_skew_strength must be at least 1.0."
            )

        if "class_weight_scope" not in config:
            raise KeyError(
                "non_iid_label_skew experiments must explicitly "
                "define class_weight_scope."
            )

    print("Federated experiment config validation passed.")
    print("Configured split type:", split_type)
    print(
        "Configured class-weight scope:",
        class_weight_scope,
    )

    if split_type == "non_iid_label_skew":
        print(
            "Configured non-IID skew strength:",
            skew_strength,
        )

    if image_source == "cache":
        print("Configured image cache:", cache_dir)
        print("Configured cache extension:", cache_extension)


def create_client_splits(
    config: Dict[str, Any],
    train_df,
    interim_dir: Path,
):
    """Create client DataFrames according to the configured split."""
    split_type = str(config["split_type"]).lower()
    num_clients = int(config["fl_num_clients"])
    seed = int(config["seed"])
    experiment_id = str(config["experiment_id"])

    if split_type == "iid":
        return create_iid_client_splits(
            train_df=train_df,
            num_clients=num_clients,
            seed=seed,
            interim_dir=interim_dir,
            experiment_id=experiment_id,
        )

    if split_type == "non_iid_label_skew":
        return create_label_skew_client_splits(
            train_df=train_df,
            num_clients=num_clients,
            seed=seed,
            interim_dir=interim_dir,
            experiment_id=experiment_id,
            skew_strength=float(
                config.get("non_iid_skew_strength", 4.0)
            ),
        )

    raise RuntimeError(
        f"Unhandled split type after validation: {split_type}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full VinDr FedAvg experiment using IID or "
            "label-skew client splitting."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a federated YAML config, for example "
            "configs/exp02_fedavg_iid_5clients.yaml"
        ),
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)

    dirs = create_experiment_dirs(config)

    config_snapshot_path = save_config_snapshot(
        config=config,
        dirs=dirs,
    )

    print("Saved config snapshot:", config_snapshot_path)

    set_seed(int(config["seed"]))

    print("\nPrepare full VinDr train/validation split:")

    train_df, val_df, xai_ids = prepare_vindr_full_data(
        config=config,
        interim_dir=dirs["interim_dir"],
    )

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Fixed XAI image IDs:", len(xai_ids))

    print("\nCreate federated client splits:")

    client_dfs = create_client_splits(
        config=config,
        train_df=train_df,
        interim_dir=dirs["interim_dir"],
    )

    print("Created clients:", len(client_dfs))

    print("\nRun Flower FedAvg simulation:")

    _, metrics_path = run_fedavg_debug(
        config=config,
        client_dfs=client_dfs,
        val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"],
        metrics_dir=dirs["metrics_dir"],
    )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Flower completed without creating the expected "
            f"metrics CSV: {metrics_path}"
        )

    final_round = int(config["fl_num_rounds"])

    final_checkpoint_path = (
        dirs["checkpoint_dir"]
        / f"flower_fedavg_round_{final_round}.pt"
    )

    if not final_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Flower completed without creating the expected final "
            f"checkpoint: {final_checkpoint_path}"
        )

    print("\nFederated experiment completed:")
    print("Metrics CSV:", metrics_path)
    print("Final checkpoint:", final_checkpoint_path)

    print("\nFinal output check:")
    print_output_check(dirs)


if __name__ == "__main__":
    main()