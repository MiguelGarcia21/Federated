import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Allows this script to be run from the repository root as:
# python src/experiments/run_exp01_centralized.py \
#   --config configs/vindr_binary_full.yaml
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess_vindr import prepare_vindr_full_data
from src.training.train_centralized import train_centralized_debug
from src.utils.paths import (
    create_experiment_dirs,
    print_output_check,
    save_config_snapshot,
)
from src.utils.seed import set_seed


SUPPORTED_CHECKPOINT_METRICS = {
    "f1",
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the minimum requirements for Experiment 1."""
    required_keys = [
        "experiment_id",
        "project_root",
        "output_dir",
        "train_csv",
        "train_image_dir",
        "dataset",
        "task",
        "image_size",
        "num_classes",
        "batch_size",
        "learning_rate",
        "epochs",
        "seed",
        "val_fraction",
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

    # experiment_id must remain a plain identifier, not a path.
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

    if str(config.get("dataset", "")).lower() != "vindr":
        raise ValueError(
            "Experiment 1 currently supports only dataset='vindr'."
        )

    if str(config.get("task", "")).lower() != "binary_abnormal":
        raise ValueError(
            "Experiment 1 currently supports only "
            "task='binary_abnormal'."
        )

    if str(config.get("model_name", "resnet18")).lower() != "resnet18":
        raise ValueError(
            "Experiment 1 currently supports only model_name='resnet18'."
        )

    if int(config["num_classes"]) != 2:
        raise ValueError(
            "Binary VinDr classification requires num_classes=2."
        )

    if int(config["image_size"]) <= 0:
        raise ValueError("image_size must be greater than zero.")

    if int(config["batch_size"]) <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if int(config["epochs"]) <= 0:
        raise ValueError("epochs must be greater than zero.")

    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    val_fraction = float(config["val_fraction"])

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be between 0 and 1, exclusive."
        )

    checkpoint_metric = str(
        config.get("checkpoint_metric", "f1")
    ).lower()

    if checkpoint_metric not in SUPPORTED_CHECKPOINT_METRICS:
        raise ValueError(
            f"Unsupported checkpoint_metric={checkpoint_metric!r}. "
            f"Currently supported: "
            f"{sorted(SUPPORTED_CHECKPOINT_METRICS)}"
        )

    print("Experiment 1 config validation passed.")


def print_device_info(device: torch.device) -> None:
    """Print the compute device and perform a small CUDA check."""
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print(
            "GPU capability:",
            torch.cuda.get_device_capability(0),
        )

        test_tensor = torch.tensor(
            [1.0, 2.0, 3.0],
            device=device,
        )

        print("CUDA test tensor:", test_tensor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment 1: full centralized VinDr binary baseline."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the Experiment 1 YAML config, for example "
            "configs/vindr_binary_full.yaml"
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

    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_device_info(device)

    print("\nExperiment 1: Prepare full VinDr data:")

    train_df, val_df, xai_ids = prepare_vindr_full_data(
        config=config,
        interim_dir=dirs["interim_dir"],
    )

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Fixed XAI image IDs:", len(xai_ids))

    print("\nExperiment 1: Centralized training:")

    best_checkpoint_path, metrics_path = train_centralized_debug(
        config=config,
        train_df=train_df,
        val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"],
        metrics_dir=dirs["metrics_dir"],
        device=device,
    )

    if not best_checkpoint_path.is_file():
        raise FileNotFoundError(
            "Centralized training completed without creating the "
            f"expected checkpoint: {best_checkpoint_path}"
        )

    if not metrics_path.is_file():
        raise FileNotFoundError(
            "Centralized training completed without creating the "
            f"expected metrics CSV: {metrics_path}"
        )

    print("\nExperiment 1 completed:")
    print("Best checkpoint:", best_checkpoint_path)
    print("Metrics CSV:", metrics_path)

    print("\nFinal output check")
    print_output_check(dirs)


if __name__ == "__main__":
    main()