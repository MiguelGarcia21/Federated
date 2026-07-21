import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

HEATMAP_COMPARISON_FIELDS = [
    "pearson_correlation",
    "cosine_similarity",
    "mean_absolute_error",
    "mean_squared_error",
    "intersection_over_union",
    "dice_score",
    "threshold",
    "heatmap_a_mean",
    "heatmap_b_mean",
    "heatmap_a_std",
    "heatmap_b_std",
    "heatmap_a_max",
    "heatmap_b_max",
    "is_degenerate_pair",
]

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

def validate_roundwise_metadata(
    metadata_df: pd.DataFrame,
    expected_image_ids: List[str],
    checkpoint_rounds: List[int],
) -> None:
    """Validate one Grad-CAM record per image and checkpoint round."""
    required_columns = {
        "image_id",
        "round",
        "true_label",
        "predicted_label",
        "raw_heatmap_path",
        "overlay_path",
    }

    missing_columns = required_columns.difference(
        metadata_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Round-wise Grad-CAM metadata is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    expected_ids = {
        str(image_id)
        for image_id in expected_image_ids
    }

    actual_ids = set(
        metadata_df["image_id"].astype(str)
    )

    missing_ids = sorted(
        expected_ids.difference(actual_ids)
    )

    unexpected_ids = sorted(
        actual_ids.difference(expected_ids)
    )

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} expected XAI images are missing "
            f"from the combined metadata. Examples: "
            f"{missing_ids[:5]}"
        )

    if unexpected_ids:
        raise ValueError(
            f"{len(unexpected_ids)} unexpected image IDs were found "
            f"in the combined metadata. Examples: "
            f"{unexpected_ids[:5]}"
        )

    expected_rounds = {
        int(round_number)
        for round_number in checkpoint_rounds
    }

    actual_rounds = {
        int(round_number)
        for round_number in metadata_df["round"].unique()
    }

    if actual_rounds != expected_rounds:
        missing_rounds = sorted(
            expected_rounds.difference(actual_rounds)
        )

        unexpected_rounds = sorted(
            actual_rounds.difference(expected_rounds)
        )

        raise ValueError(
            "Round-wise metadata does not contain the expected rounds. "
            f"Missing: {missing_rounds}; "
            f"unexpected: {unexpected_rounds}."
        )

    duplicate_mask = metadata_df.duplicated(
        subset=[
            "image_id",
            "round",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = (
            metadata_df.loc[
                duplicate_mask,
                [
                    "image_id",
                    "round",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate image-round Grad-CAM records were found. "
            f"Examples: {duplicate_rows}"
        )

    expected_rows = (
        len(expected_ids)
        * len(expected_rounds)
    )

    if len(metadata_df) != expected_rows:
        raise ValueError(
            "Unexpected number of Grad-CAM metadata rows. "
            f"Expected {expected_rows}, found {len(metadata_df)}."
        )

    counts_per_image = (
        metadata_df
        .groupby("image_id")["round"]
        .nunique()
    )

    invalid_counts = counts_per_image[
        counts_per_image != len(expected_rounds)
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "One or more images do not contain exactly one record "
            "for every requested checkpoint round. Examples: "
            f"{invalid_counts.head().to_dict()}"
        )

    true_label_counts = (
        metadata_df
        .groupby("image_id")["true_label"]
        .nunique()
    )

    inconsistent_labels = true_label_counts[
        true_label_counts != 1
    ]

    if not inconsistent_labels.empty:
        raise ValueError(
            "True labels changed between rounds for one or more images. "
            f"Examples: {inconsistent_labels.head().to_dict()}"
        )

    missing_files = []

    for column_name in [
        "raw_heatmap_path",
        "overlay_path",
    ]:
        for file_path in metadata_df[column_name]:
            path = Path(file_path)

            if not path.is_file():
                missing_files.append(str(path))

                if len(missing_files) >= 10:
                    break

        if len(missing_files) >= 10:
            break

    if missing_files:
        raise FileNotFoundError(
            "One or more Grad-CAM output files referenced by the "
            "metadata do not exist. Examples:\n"
            + "\n".join(missing_files)
        )

    print(
        "Round-wise Grad-CAM metadata validation passed:",
        f"{len(expected_ids)} images × "
        f"{len(expected_rounds)} rounds = "
        f"{expected_rows} records.",
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

    validate_roundwise_metadata(
        metadata_df=combined_df,
        expected_image_ids=xai_ids,
        checkpoint_rounds=checkpoint_rounds,
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

def add_comparison_fields(
    row: Dict[str, Any],
    suffix: str,
    metrics: Dict[str, Any] | None,
) -> None:
    """Add a complete set of heatmap-comparison fields to one row."""
    if metrics is not None:
        missing_metrics = set(
            HEATMAP_COMPARISON_FIELDS
        ).difference(metrics.keys())

        if missing_metrics:
            raise ValueError(
                "Heatmap comparison did not return all expected "
                "metrics: "
                + ", ".join(sorted(missing_metrics))
            )

    for metric_name in HEATMAP_COMPARISON_FIELDS:
        row[f"{metric_name}_{suffix}"] = (
            metrics[metric_name]
            if metrics is not None
            else None
        )

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

                add_comparison_fields(
                    row=row,
                    suffix="to_final",
                    metrics=final_metrics,
                )

                row["to_final_comparison_status"] = (
                    "self_comparison"
                )

            elif current_predicted_label == final_predicted_label:
                final_metrics = compare_heatmap_files(
                    current_row["raw_heatmap_path"],
                    final_heatmap_path,
                    threshold=threshold,
                )

                add_comparison_fields(
                    row=row,
                    suffix="to_final",
                    metrics=final_metrics,
                )

                row["to_final_comparison_status"] = "computed"

            else:
                add_comparison_fields(
                    row=row,
                    suffix="to_final",
                    metrics=None,
                )

                row["to_final_comparison_status"] = (
                    "skipped_target_changed"
                )

            if previous_row is None:
                add_comparison_fields(
                    row=row,
                    suffix="to_previous",
                    metrics=None,
                )

                row["to_previous_comparison_status"] = (
                    "not_applicable_first_round"
                )

            elif (
                current_predicted_label
                == int(previous_row["predicted_label"])
            ):
                previous_metrics = compare_heatmap_files(
                    previous_row["raw_heatmap_path"],
                    current_row["raw_heatmap_path"],
                    threshold=threshold,
                )

                add_comparison_fields(
                    row=row,
                    suffix="to_previous",
                    metrics=previous_metrics,
                )

                row["to_previous_comparison_status"] = "computed"

            else:
                add_comparison_fields(
                    row=row,
                    suffix="to_previous",
                    metrics=None,
                )

                row["to_previous_comparison_status"] = (
                    "skipped_target_changed"
                )

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