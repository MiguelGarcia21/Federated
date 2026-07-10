import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allows this file to be run exactly as:
# python src/experiments/run_debug_reproduce_kaggle.py --config configs/vindr_binary_debug.yaml
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.seed import set_seed
from src.utils.paths import (
    create_experiment_dirs,
    save_config_snapshot,
    print_output_check,
    zip_experiment_outputs,
)
from src.data.preprocess_vindr import prepare_vindr_debug_data
from src.data.split_clients import create_iid_client_splits
from src.training.train_centralized import train_centralized_debug
from src.training.train_flower_fedavg import run_fedavg_debug
from src.xai.gradcam import generate_gradcam_outputs


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def validate_debug_config(config):
    required_keys = [
        "experiment_id",
        "project_root",
        "output_dir",
        "train_csv",
        "train_image_dir",
        "image_size",
        "batch_size",
        "num_classes",
        "seed",
    ]

    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise KeyError(
            "Missing required config keys: "
            + ", ".join(missing_keys)
        )

    train_csv = Path(config["train_csv"])
    train_image_dir = Path(config["train_image_dir"])

    if not train_csv.exists():
        raise FileNotFoundError(
            f"Configured train_csv does not exist: {train_csv}"
        )

    if not train_image_dir.exists():
        raise FileNotFoundError(
            f"Configured train_image_dir does not exist: {train_image_dir}"
        )

    if not train_image_dir.is_dir():
        raise NotADirectoryError(
            f"Configured train_image_dir is not a directory: {train_image_dir}"
        )

    print("Config validation passed.")


def print_device_info(device):
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))

        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print("CUDA test tensor:", test_tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config, for example configs/vindr_binary_debug.yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_debug_config(config)

    dirs = create_experiment_dirs(config)
    save_config_snapshot(config, dirs)

    set_seed(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_device_info(device)

    print("\n=== Step 3-6: Prepare VinDr debug data ===")
    train_df, val_df, xai_ids = prepare_vindr_debug_data(
        config=config,
        interim_dir=dirs["interim_dir"],
    )

    print("\n=== Step 7-12: Centralized ResNet-18 debug training ===")
    best_checkpoint_path, _ = train_centralized_debug(
        config=config,
        train_df=train_df,
        val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"],
        metrics_dir=dirs["metrics_dir"],
        device=device,
    )

    print("\n=== Step 13-14: Centralized Grad-CAM ===")
    generate_gradcam_outputs(
        config=config,
        checkpoint_path=best_checkpoint_path,
        val_df=val_df,
        xai_ids=xai_ids,
        heatmap_dir=dirs["heatmap_dir"],
        metrics_dir=dirs["metrics_dir"],
        run_type="centralized_debug",
        device=device,
    )

    print("\n=== Step 15: Create IID client splits ===")
    num_clients = int(config.get("fl_num_clients", config.get("num_clients", 3)))
    client_dfs = create_iid_client_splits(
        train_df=train_df,
        num_clients=num_clients,
        seed=int(config["seed"]),
        interim_dir=dirs["interim_dir"],
    )

    print("\n=== Step 16-20: Flower FedAvg debug simulation ===")
    run_fedavg_debug(
        config=config,
        client_dfs=client_dfs,
        val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"],
        metrics_dir=dirs["metrics_dir"],
    )

    print("\n=== Step 21: Grad-CAM from final FL model ===")
    final_round = int(config.get("fl_num_rounds", config.get("fl_rounds", 3)))
    final_fl_checkpoint_path = (
        dirs["checkpoint_dir"] / f"flower_fedavg_round_{final_round}.pt"
    )

    print("Final FL checkpoint exists:", final_fl_checkpoint_path.exists())
    print("Path:", final_fl_checkpoint_path)

    if not final_fl_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected final FL checkpoint not found: {final_fl_checkpoint_path}. "
            "Check whether Flower completed all rounds and saved checkpoints correctly."
        )

    generate_gradcam_outputs(
        config=config,
        checkpoint_path=final_fl_checkpoint_path,
        val_df=val_df,
        xai_ids=xai_ids,
        heatmap_dir=dirs["heatmap_dir"],
        metrics_dir=dirs["metrics_dir"],
        run_type="flower_fedavg_debug",
        device=device,
        round_num=final_round,
    )

    print("\n=== Step 22: Final output check ===")
    print_output_check(dirs)

    if bool(config.get("create_output_zip", False)):
        print("\n=== Step 23: Zip outputs for download/export ===")
        zip_experiment_outputs(
            dirs=dirs,
            experiment_id=config["experiment_id"],
        )

    print("\nDebug RunPod reproduction script completed.")


if __name__ == "__main__":
    main()