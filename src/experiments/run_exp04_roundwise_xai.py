import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocess_vindr import prepare_vindr_full_data
from src.utils.paths import (
    create_experiment_dirs,
    print_output_check,
    save_config_snapshot,
)
from src.utils.seed import set_seed
from src.xai.gradcam import generate_gradcam_outputs
from src.xai.xai_metrics import compare_heatmap_files


def load_config(config_path: Path) -> Dict[str, Any]:
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
    required_keys = [
        "experiment_id",
        "project_root",
        "output_dir",
        "train_csv",
        "train_image_dir",
        "dataset",
        "task",
        "model_name",
        "num_classes",
        "image_size",
        "seed",
        "val_fraction",
        "xai_subset_size",
        "source_experiment_id",
        "source_checkpoint_dir",
        "checkpoint_rounds",
        "final_round",
        "source_split_type",
        "heatmap_threshold",
    ]

    missing_keys = [
        key for key in required_keys
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
            "experiment_id must be a plain identifier."
        )

    checkpoint_rounds = config["checkpoint_rounds"]

    if not isinstance(checkpoint_rounds, list):
        raise ValueError(
            "checkpoint_rounds must be a YAML list."
        )

    checkpoint_rounds = [
        int(round_number)
        for round_number in checkpoint_rounds
    ]

    if not checkpoint_rounds:
        raise ValueError(
            "checkpoint_rounds cannot be empty."
        )

    if any(round_number < 0 for round_number in checkpoint_rounds):
        raise ValueError(
            "Checkpoint rounds cannot be negative."
        )

    if len(checkpoint_rounds) != len(set(checkpoint_rounds)):
        raise ValueError(
            "checkpoint_rounds contains duplicate values."
        )

    final_round = int(config["final_round"])

    if final_round not in checkpoint_rounds:
        raise ValueError(
            "final_round must be included in checkpoint_rounds."
        )

    threshold = float(config["heatmap_threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "heatmap_threshold must be between 0 and 1."
        )

    source_checkpoint_dir = Path(
        config["source_checkpoint_dir"]
    )

    if not source_checkpoint_dir.is_dir():
        raise FileNotFoundError(
            "Source checkpoint directory does not exist: "
            f"{source_checkpoint_dir}"
        )

    print("Round-wise XAI config validation passed.")


def get_checkpoint_path(
    source_checkpoint_dir: Path,
    round_number: int,
) -> Path:
    return (
        source_checkpoint_dir
        / f"flower_fedavg_round_{round_number}.pt"
    )


def validate_checkpoints(
    source_checkpoint_dir: Path,
    checkpoint_rounds: List[int],
) -> None:
    missing_checkpoints = []

    for round_number in checkpoint_rounds:
        checkpoint_path = get_checkpoint_path(
            source_checkpoint_dir,
            round_number,
        )

        if not checkpoint_path.is_file():
            missing_checkpoints.append(
                str(checkpoint_path)
            )

    if missing_checkpoints:
        raise FileNotFoundError(
            "One or more requested checkpoints are missing:\n"
            + "\n".join(missing_checkpoints)
        )

    print(
        f"Validated {len(checkpoint_rounds)} checkpoints."
    )


def generate_roundwise_gradcam(
    config: Dict[str, Any],
    val_df: pd.DataFrame,
    xai_ids: List[str],
    heatmap_dir: Path,
    metrics_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    source_checkpoint_dir = Path(
        config["source_checkpoint_dir"]
    )

    checkpoint_rounds = sorted(
        int(round_number)
        for round_number in config["checkpoint_rounds"]
    )

    metadata_frames = []

    for round_number in checkpoint_rounds:
        checkpoint_path = get_checkpoint_path(
            source_checkpoint_dir,
            round_number,
        )

        print(
            f"\nGenerating Grad-CAM for round {round_number}"
        )

        metadata_path = generate_gradcam_outputs(
            config=config,
            checkpoint_path=checkpoint_path,
            val_df=val_df,
            xai_ids=xai_ids,
            heatmap_dir=heatmap_dir,
            metrics_dir=metrics_dir,
            run_type=str(
                config.get(
                    "source_split_type",
                    "federated",
                )
            ),
            device=device,
            round_num=round_number,
        )

        round_df = pd.read_csv(metadata_path)
        metadata_frames.append(round_df)

    combined_df = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    combined_metadata_path = (
        metrics_dir
        / f"{config['experiment_id']}_roundwise_gradcam_metadata.csv"
    )

    combined_df.to_csv(
        combined_metadata_path,
        index=False,
    )

    print(
        "Saved combined round-wise Grad-CAM metadata:",
        combined_metadata_path,
    )

    return combined_df


def build_similarity_results(
    metadata_df: pd.DataFrame,
    final_round: int,
    threshold: float,
) -> pd.DataFrame:
    rows = []

    required_columns = {
        "image_id",
        "round",
        "predicted_label",
        "true_label",
        "confidence",
        "raw_heatmap_path",
    }

    missing_columns = required_columns.difference(
        metadata_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Grad-CAM metadata is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    for image_id, image_df in metadata_df.groupby("image_id"):
        image_df = (
            image_df
            .sort_values("round")
            .reset_index(drop=True)
        )

        final_rows = image_df[
            image_df["round"] == final_round
        ]

        if len(final_rows) != 1:
            raise ValueError(
                f"Expected one final-round heatmap for image "
                f"{image_id}, found {len(final_rows)}."
            )

        final_row = final_rows.iloc[0]
        final_heatmap_path = final_row["raw_heatmap_path"]
        final_predicted_label = int(
            final_row["predicted_label"]
        )

        for index, current_row in image_df.iterrows():
            current_round = int(current_row["round"])
            current_predicted_label = int(
                current_row["predicted_label"]
            )

            previous_row = (
                image_df.iloc[index - 1]
                if index > 0
                else None
            )

            row = {
                "experiment_id": current_row[
                    "experiment_id"
                ],
                "source_experiment_id": current_row.get(
                    "source_experiment_id",
                    None,
                ),
                "image_id": str(image_id),
                "true_label": int(
                    current_row["true_label"]
                ),
                "round": current_round,
                "predicted_label": current_predicted_label,
                "confidence": float(
                    current_row["confidence"]
                ),
                "final_round": final_round,
                "final_predicted_label": (
                    final_predicted_label
                ),
                "same_target_as_final": int(
                    current_predicted_label
                    == final_predicted_label
                ),
                "previous_round": (
                    int(previous_row["round"])
                    if previous_row is not None
                    else None
                ),
                "same_target_as_previous": (
                    int(
                        current_predicted_label
                        == int(previous_row["predicted_label"])
                    )
                    if previous_row is not None
                    else None
                ),
            }

            if current_round == final_round:
                final_metrics = compare_heatmap_files(
                    current_row["raw_heatmap_path"],
                    final_heatmap_path,
                    threshold=threshold,
                )

                for metric_name, metric_value in final_metrics.items():
                    row[
                        f"{metric_name}_to_final"
                    ] = metric_value

            elif current_predicted_label == final_predicted_label:
                final_metrics = compare_heatmap_files(
                    current_row["raw_heatmap_path"],
                    final_heatmap_path,
                    threshold=threshold,
                )

                for metric_name, metric_value in final_metrics.items():
                    row[
                        f"{metric_name}_to_final"
                    ] = metric_value

            else:
                for metric_name in [
                    "pearson_correlation",
                    "cosine_similarity",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "intersection_over_union",
                    "dice_score",
                    "threshold",
                ]:
                    row[
                        f"{metric_name}_to_final"
                    ] = None

            if previous_row is None:
                for metric_name in [
                    "pearson_correlation",
                    "cosine_similarity",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "intersection_over_union",
                    "dice_score",
                    "threshold",
                ]:
                    row[
                        f"{metric_name}_to_previous"
                    ] = None

            elif (
                current_predicted_label
                == int(previous_row["predicted_label"])
            ):
                previous_metrics = compare_heatmap_files(
                    previous_row["raw_heatmap_path"],
                    current_row["raw_heatmap_path"],
                    threshold=threshold,
                )

                for metric_name, metric_value in previous_metrics.items():
                    row[
                        f"{metric_name}_to_previous"
                    ] = metric_value

            else:
                for metric_name in [
                    "pearson_correlation",
                    "cosine_similarity",
                    "mean_absolute_error",
                    "mean_squared_error",
                    "intersection_over_union",
                    "dice_score",
                    "threshold",
                ]:
                    row[
                        f"{metric_name}_to_previous"
                    ] = None

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and compare round-wise Grad-CAM heatmaps "
            "from saved federated checkpoints."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a round-wise XAI YAML config.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)

    checkpoint_rounds = sorted(
        int(round_number)
        for round_number in config["checkpoint_rounds"]
    )

    source_checkpoint_dir = Path(
        config["source_checkpoint_dir"]
    )

    validate_checkpoints(
        source_checkpoint_dir=source_checkpoint_dir,
        checkpoint_rounds=checkpoint_rounds,
    )

    dirs = create_experiment_dirs(config)
    save_config_snapshot(
        config=config,
        dirs=dirs,
    )

    set_seed(int(config["seed"]))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    print("\nPrepare full VinDr validation split:")

    _, val_df, xai_ids = prepare_vindr_full_data(
        config=config,
        interim_dir=dirs["interim_dir"],
    )

    print("Validation samples:", len(val_df))
    print("XAI images:", len(xai_ids))

    print("\nGenerate round-wise Grad-CAM:")

    metadata_df = generate_roundwise_gradcam(
        config=config,
        val_df=val_df,
        xai_ids=xai_ids,
        heatmap_dir=dirs["heatmap_dir"],
        metrics_dir=dirs["metrics_dir"],
        device=device,
    )

    print("\nCompute explanation stability:")

    similarity_df = build_similarity_results(
        metadata_df=metadata_df,
        final_round=int(config["final_round"]),
        threshold=float(
            config["heatmap_threshold"]
        ),
    )

    similarity_path = (
        dirs["metrics_dir"]
        / f"{config['experiment_id']}_roundwise_xai_similarity.csv"
    )

    similarity_df.to_csv(
        similarity_path,
        index=False,
    )

    print(
        "Saved round-wise XAI similarity metrics:",
        similarity_path,
    )

    print("\nFinal output check:")
    print_output_check(dirs)


if __name__ == "__main__":
    main()