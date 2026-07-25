import argparse
import sys
from itertools import combinations
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

SOURCE_KEYS = [
    "centralized_source",
    "iid_source",
    "non_iid_source",
]

EXPECTED_CONDITIONS = {
    "centralized_source": "centralized",
    "iid_source": "iid",
    "non_iid_source": "non_iid_label_skew",
}


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


def validate_plain_identifier(
    value: Any,
    field_name: str,
) -> str:
    identifier = str(value).strip()

    if not identifier:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    if Path(identifier).name != identifier:
        raise ValueError(
            f"{field_name} must be a plain identifier."
        )

    return identifier


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
        "use_full_dataset",
        "reuse_split_files",
        "val_fraction",
        "xai_subset_size",
        "canonical_train_split_path",
        "canonical_val_split_path",
        "canonical_xai_ids_path",
        "image_source",
        "heatmap_threshold",
        "xai_method",
        *SOURCE_KEYS,
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

    validate_plain_identifier(
        config["experiment_id"],
        "experiment_id",
    )

    if config["use_full_dataset"] is not True:
        raise ValueError(
            "Experiment 5 must use the full canonical dataset."
        )

    if config["reuse_split_files"] is not True:
        raise ValueError(
            "Experiment 5 must reuse the canonical split files."
        )

    if str(config["xai_method"]).strip().lower() != "gradcam":
        raise ValueError(
            "Experiment 5 currently supports only Grad-CAM."
        )

    xai_subset_size = int(config["xai_subset_size"])

    if xai_subset_size <= 0:
        raise ValueError(
            "xai_subset_size must be greater than zero."
        )

    threshold = float(config["heatmap_threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "heatmap_threshold must be between 0 and 1."
        )

    canonical_paths = [
        Path(config["canonical_train_split_path"]),
        Path(config["canonical_val_split_path"]),
        Path(config["canonical_xai_ids_path"]),
    ]

    for path in canonical_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Canonical file does not exist: {path}"
            )

    image_source = str(
        config["image_source"]
    ).strip().lower()

    if image_source not in {
        "dicom",
        "cache",
    }:
        raise ValueError(
            "image_source must be either 'dicom' or 'cache'."
        )

    if image_source == "cache":
        configured_cache_dir = config.get(
            "image_cache_dir"
        )

        if not configured_cache_dir:
            raise KeyError(
                "image_cache_dir is required when "
                "image_source='cache'."
            )

        cache_dir = Path(configured_cache_dir)

        if not cache_dir.is_dir():
            raise FileNotFoundError(
                f"Image cache directory does not exist: {cache_dir}"
            )

    observed_conditions = []

    for source_key in SOURCE_KEYS:
        source = config[source_key]

        if not isinstance(source, dict):
            raise ValueError(
                f"{source_key} must contain a YAML mapping."
            )

        required_source_keys = [
            "experiment_id",
            "condition",
            "split_type",
            "checkpoint_path",
        ]

        if source_key != "centralized_source":
            required_source_keys.extend([
                "checkpoint_round",
                "metadata_path",
            ])

        missing_source_keys = [
            key
            for key in required_source_keys
            if key not in source
        ]

        if missing_source_keys:
            raise KeyError(
                f"{source_key} is missing keys: "
                + ", ".join(missing_source_keys)
            )

        validate_plain_identifier(
            source["experiment_id"],
            f"{source_key}.experiment_id",
        )

        condition = str(
            source["condition"]
        ).strip().lower()

        expected_condition = EXPECTED_CONDITIONS[
            source_key
        ]

        if condition != expected_condition:
            raise ValueError(
                f"{source_key}.condition must be "
                f"{expected_condition!r}, found {condition!r}."
            )

        observed_conditions.append(condition)

        checkpoint_path = Path(
            source["checkpoint_path"]
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        if source_key != "centralized_source":
            checkpoint_round = int(
                source["checkpoint_round"]
            )

            if checkpoint_round < 0:
                raise ValueError(
                    f"{source_key}.checkpoint_round cannot be negative."
                )

            metadata_path = Path(
                source["metadata_path"]
            )

            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"Metadata file does not exist: {metadata_path}"
                )

    if len(observed_conditions) != len(
        set(observed_conditions)
    ):
        raise ValueError(
            "Each Experiment 5 source must have a unique condition."
        )

    print("Experiment 5 config validation passed.")
    print("Experiment:", config["experiment_id"])
    print("XAI images:", xai_subset_size)
    print("Heatmap threshold:", threshold)
    print("Image source:", image_source)


def validate_source_metadata(
    metadata_df: pd.DataFrame,
    source: Dict[str, Any],
    expected_image_ids: List[str],
    expected_round: int | None,
    expected_image_source: str,
) -> pd.DataFrame:
    required_columns = {
        "source_experiment_id",
        "training_type",
        "split_type",
        "image_source",
        "round",
        "checkpoint_path",
        "image_id",
        "true_label",
        "predicted_label",
        "correct",
        "confidence",
        "xai_method",
        "target_class",
        "target_layer",
        "raw_heatmap_path",
        "overlay_path",
    }

    missing_columns = required_columns.difference(
        metadata_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Metadata for {source['condition']} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    metadata_df = metadata_df.copy()
    metadata_df["image_id"] = (
        metadata_df["image_id"].astype(str)
    )

    expected_ids = set(expected_image_ids)
    actual_ids = set(metadata_df["image_id"])

    if actual_ids != expected_ids:
        missing_ids = sorted(
            expected_ids.difference(actual_ids)
        )
        unexpected_ids = sorted(
            actual_ids.difference(expected_ids)
        )

        raise ValueError(
            f"Image-ID mismatch for {source['condition']}. "
            f"Missing examples: {missing_ids[:5]}; "
            f"unexpected examples: {unexpected_ids[:5]}."
        )

    duplicate_mask = metadata_df.duplicated(
        subset=["image_id"],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            metadata_df.loc[
                duplicate_mask,
                ["image_id"],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"Duplicate image records found for "
            f"{source['condition']}: {examples}"
        )

    if len(metadata_df) != len(expected_ids):
        raise ValueError(
            f"Expected {len(expected_ids)} metadata rows for "
            f"{source['condition']}, found {len(metadata_df)}."
        )

    source_experiment_values = set(
        metadata_df["source_experiment_id"]
        .dropna()
        .astype(str)
    )

    if source_experiment_values != {
        str(source["experiment_id"])
    }:
        raise ValueError(
            f"Unexpected source_experiment_id values for "
            f"{source['condition']}: "
            f"{sorted(source_experiment_values)}"
        )

    training_type_values = set(
        metadata_df["training_type"]
        .dropna()
        .astype(str)
    )

    if training_type_values != {
        str(source["condition"])
    }:
        raise ValueError(
            f"Unexpected training_type values for "
            f"{source['condition']}: "
            f"{sorted(training_type_values)}"
        )

    split_type_values = set(
        metadata_df["split_type"]
        .dropna()
        .astype(str)
    )

    if split_type_values != {
        str(source["split_type"])
    }:
        raise ValueError(
            f"Unexpected split_type values for "
            f"{source['condition']}: "
            f"{sorted(split_type_values)}"
        )

    image_source_values = set(
        metadata_df["image_source"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    if image_source_values != {
        expected_image_source
    }:
        raise ValueError(
            f"Unexpected image_source values for "
            f"{source['condition']}: "
            f"{sorted(image_source_values)}"
        )

    xai_method_values = set(
        metadata_df["xai_method"]
        .dropna()
        .astype(str)
        .str.lower()
    )

    if xai_method_values != {"gradcam"}:
        raise ValueError(
            f"Unexpected XAI methods for {source['condition']}: "
            f"{sorted(xai_method_values)}"
        )

    target_layer_values = set(
        metadata_df["target_layer"]
        .dropna()
        .astype(str)
    )

    if target_layer_values != {"layer4[-1]"}:
        raise ValueError(
            f"Unexpected Grad-CAM target layer for "
            f"{source['condition']}: "
            f"{sorted(target_layer_values)}"
        )

    expected_checkpoint_path = Path(
        source["checkpoint_path"]
    ).resolve()

    observed_checkpoint_paths = {
        Path(path).resolve()
        for path in metadata_df["checkpoint_path"].astype(str)
    }

    if observed_checkpoint_paths != {
        expected_checkpoint_path
    }:
        raise ValueError(
            f"Checkpoint provenance mismatch for "
            f"{source['condition']}: "
            f"{sorted(str(path) for path in observed_checkpoint_paths)}"
        )

    if expected_round is None:
        if metadata_df["round"].notna().any():
            raise ValueError(
                "Centralized metadata unexpectedly contains "
                "a federated round."
            )
    else:
        observed_rounds = set(
            pd.to_numeric(
                metadata_df["round"],
                errors="raise",
            ).astype(int)
        )

        if observed_rounds != {expected_round}:
            raise ValueError(
                f"Expected round {expected_round} for "
                f"{source['condition']}, found "
                f"{sorted(observed_rounds)}."
            )

    target_mismatch = (
        pd.to_numeric(
            metadata_df["target_class"],
            errors="raise",
        ).astype(int)
        !=
        pd.to_numeric(
            metadata_df["predicted_label"],
            errors="raise",
        ).astype(int)
    )

    if target_mismatch.any():
        raise ValueError(
            f"Grad-CAM target class differs from the predicted "
            f"class for {source['condition']}."
        )

    missing_files = []

    for column_name in [
        "raw_heatmap_path",
        "overlay_path",
    ]:
        for value in metadata_df[column_name]:
            path = Path(value)

            if not path.is_file():
                missing_files.append(str(path))

                if len(missing_files) >= 10:
                    break

        if len(missing_files) >= 10:
            break

    if missing_files:
        raise FileNotFoundError(
            f"Missing Grad-CAM files for {source['condition']}:\n"
            + "\n".join(missing_files)
        )

    metadata_df["comparison_experiment_id"] = None
    metadata_df["condition"] = str(
        source["condition"]
    )
    metadata_df["model_experiment_id"] = str(
        source["experiment_id"]
    )

    print(
        f"Validated {source['condition']} metadata:",
        f"{len(metadata_df)} images.",
    )

    return metadata_df


def generate_centralized_metadata(
    config: Dict[str, Any],
    source: Dict[str, Any],
    val_df: pd.DataFrame,
    xai_ids: List[str],
    heatmap_dir: Path,
    metrics_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    centralized_config = dict(config)

    centralized_config["source_experiment_id"] = (
        source["experiment_id"]
    )
    centralized_config["source_split_type"] = (
        source["split_type"]
    )

    metadata_path = generate_gradcam_outputs(
        config=centralized_config,
        checkpoint_path=Path(
            source["checkpoint_path"]
        ),
        val_df=val_df,
        xai_ids=xai_ids,
        heatmap_dir=heatmap_dir,
        metrics_dir=metrics_dir,
        run_type=str(source["condition"]),
        device=device,
        round_num=None,
    )

    return pd.read_csv(metadata_path)


def add_comparison_fields(
    row: Dict[str, Any],
    metrics: Dict[str, Any] | None,
) -> None:
    if metrics is not None:
        missing_metrics = set(
            HEATMAP_COMPARISON_FIELDS
        ).difference(metrics.keys())

        if missing_metrics:
            raise ValueError(
                "Heatmap comparison did not return all expected "
                "fields: "
                + ", ".join(sorted(missing_metrics))
            )

    for field_name in HEATMAP_COMPARISON_FIELDS:
        row[field_name] = (
            metrics[field_name]
            if metrics is not None
            else None
        )


def build_pairwise_comparisons(
    metadata_df: pd.DataFrame,
    xai_ids: List[str],
    conditions: List[str],
    threshold: float,
    experiment_id: str,
) -> pd.DataFrame:
    indexed_frames = {}

    for condition in conditions:
        condition_df = (
            metadata_df[
                metadata_df["condition"] == condition
            ]
            .copy()
            .set_index("image_id")
        )

        indexed_frames[condition] = condition_df

    rows = []

    for condition_a, condition_b in combinations(
        conditions,
        2,
    ):
        frame_a = indexed_frames[condition_a]
        frame_b = indexed_frames[condition_b]

        for image_id in xai_ids:
            row_a = frame_a.loc[image_id]
            row_b = frame_b.loc[image_id]

            true_label_a = int(row_a["true_label"])
            true_label_b = int(row_b["true_label"])

            if true_label_a != true_label_b:
                raise ValueError(
                    f"True-label mismatch for image {image_id}: "
                    f"{condition_a}={true_label_a}, "
                    f"{condition_b}={true_label_b}."
                )

            predicted_label_a = int(
                row_a["predicted_label"]
            )
            predicted_label_b = int(
                row_b["predicted_label"]
            )

            same_target = (
                predicted_label_a
                == predicted_label_b
            )

            comparison_row = {
                "experiment_id": experiment_id,
                "comparison": (
                    f"{condition_a}_vs_{condition_b}"
                ),
                "condition_a": condition_a,
                "condition_b": condition_b,
                "model_experiment_id_a": row_a[
                    "model_experiment_id"
                ],
                "model_experiment_id_b": row_b[
                    "model_experiment_id"
                ],
                "checkpoint_path_a": row_a[
                    "checkpoint_path"
                ],
                "checkpoint_path_b": row_b[
                    "checkpoint_path"
                ],
                "image_id": str(image_id),
                "true_label": true_label_a,
                "predicted_label_a": predicted_label_a,
                "predicted_label_b": predicted_label_b,
                "target_class_a": int(
                    row_a["target_class"]
                ),
                "target_class_b": int(
                    row_b["target_class"]
                ),
                "confidence_a": float(
                    row_a["confidence"]
                ),
                "confidence_b": float(
                    row_b["confidence"]
                ),
                "confidence_difference_a_minus_b": (
                    float(row_a["confidence"])
                    - float(row_b["confidence"])
                ),
                "absolute_confidence_difference": abs(
                    float(row_a["confidence"])
                    - float(row_b["confidence"])
                ),
                "correct_a": int(row_a["correct"]),
                "correct_b": int(row_b["correct"]),
                "correctness_agreement": int(
                    int(row_a["correct"])
                    == int(row_b["correct"])
                ),
                "both_correct": int(
                    int(row_a["correct"]) == 1
                    and int(row_b["correct"]) == 1
                ),
                "both_incorrect": int(
                    int(row_a["correct"]) == 0
                    and int(row_b["correct"]) == 0
                ),
                "prediction_agreement": int(
                    same_target
                ),
                "same_target": int(
                    same_target
                ),
                "raw_heatmap_path_a": row_a[
                    "raw_heatmap_path"
                ],
                "raw_heatmap_path_b": row_b[
                    "raw_heatmap_path"
                ],
                "overlay_path_a": row_a[
                    "overlay_path"
                ],
                "overlay_path_b": row_b[
                    "overlay_path"
                ],
            }

            if same_target:
                metrics = compare_heatmap_files(
                    row_a["raw_heatmap_path"],
                    row_b["raw_heatmap_path"],
                    threshold=threshold,
                )

                add_comparison_fields(
                    row=comparison_row,
                    metrics=metrics,
                )

                comparison_row["comparison_status"] = (
                    "computed"
                )

            else:
                add_comparison_fields(
                    row=comparison_row,
                    metrics=None,
                )

                comparison_row["comparison_status"] = (
                    "skipped_target_changed"
                )

            rows.append(comparison_row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final centralized, IID federated, and "
            "non-IID federated Grad-CAM explanations."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the Experiment 5 YAML configuration.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config)

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

    print("\nLoad canonical validation and XAI data:")

    _, val_df, xai_ids = prepare_vindr_full_data(
        config=config,
        interim_dir=dirs["interim_dir"],
    )

    normalized_xai_ids = [
        str(image_id)
        for image_id in xai_ids
    ]

    expected_xai_count = int(
        config["xai_subset_size"]
    )

    if len(normalized_xai_ids) != expected_xai_count:
        raise ValueError(
            "Loaded canonical XAI subset size does not match "
            f"the configuration. Expected {expected_xai_count}, "
            f"found {len(normalized_xai_ids)}."
        )

    if len(normalized_xai_ids) != len(
        set(normalized_xai_ids)
    ):
        raise ValueError(
            "The canonical XAI subset contains duplicate IDs."
        )

    validation_ids = set(
        val_df["image_id"].astype(str)
    )

    missing_xai_ids = [
        image_id
        for image_id in normalized_xai_ids
        if image_id not in validation_ids
    ]

    if missing_xai_ids:
        raise ValueError(
            f"{len(missing_xai_ids)} canonical XAI IDs are "
            f"missing from the validation split. Examples: "
            f"{missing_xai_ids[:5]}"
        )

    print("Validation samples:", len(val_df))
    print("Canonical XAI images:", len(normalized_xai_ids))
    print("Canonical XAI validation passed.")

    image_source = str(
        config["image_source"]
    ).strip().lower()

    metadata_frames = []

    print("\nGenerate centralized Grad-CAM:")

    centralized_source = config[
        "centralized_source"
    ]

    centralized_df = generate_centralized_metadata(
        config=config,
        source=centralized_source,
        val_df=val_df,
        xai_ids=normalized_xai_ids,
        heatmap_dir=dirs["heatmap_dir"],
        metrics_dir=dirs["metrics_dir"],
        device=device,
    )

    centralized_df = validate_source_metadata(
        metadata_df=centralized_df,
        source=centralized_source,
        expected_image_ids=normalized_xai_ids,
        expected_round=None,
        expected_image_source=image_source,
    )

    metadata_frames.append(centralized_df)

    print("\nLoad validated IID and non-IID Grad-CAM metadata:")

    for source_key in [
        "iid_source",
        "non_iid_source",
    ]:
        source = config[source_key]

        source_df = pd.read_csv(
            source["metadata_path"]
        )

        source_df = validate_source_metadata(
            metadata_df=source_df,
            source=source,
            expected_image_ids=normalized_xai_ids,
            expected_round=int(
                source["checkpoint_round"]
            ),
            expected_image_source=image_source,
        )

        metadata_frames.append(source_df)

    combined_metadata_df = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    combined_metadata_df[
        "comparison_experiment_id"
    ] = config["experiment_id"]

    expected_combined_rows = (
        expected_xai_count
        * len(SOURCE_KEYS)
    )

    if len(combined_metadata_df) != expected_combined_rows:
        raise ValueError(
            "Unexpected combined metadata row count. "
            f"Expected {expected_combined_rows}, "
            f"found {len(combined_metadata_df)}."
        )

    duplicate_mask = combined_metadata_df.duplicated(
        subset=[
            "condition",
            "image_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            combined_metadata_df.loc[
                duplicate_mask,
                [
                    "condition",
                    "image_id",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "Duplicate condition-image records found: "
            f"{examples}"
        )

    true_label_counts = (
        combined_metadata_df
        .groupby("image_id")["true_label"]
        .nunique()
    )

    inconsistent_labels = true_label_counts[
        true_label_counts != 1
    ]

    if not inconsistent_labels.empty:
        raise ValueError(
            "True labels differ between model conditions. "
            f"Examples: {inconsistent_labels.head().to_dict()}"
        )

    combined_metadata_path = (
        dirs["metrics_dir"]
        / f"{config['experiment_id']}_"
          "cross_model_gradcam_metadata.csv"
    )

    combined_metadata_df.to_csv(
        combined_metadata_path,
        index=False,
    )

    print(
        "Saved combined cross-model Grad-CAM metadata:",
        combined_metadata_path,
    )
    print(
        "Combined metadata validation passed:",
        f"{expected_xai_count} images × "
        f"{len(SOURCE_KEYS)} models = "
        f"{expected_combined_rows} records.",
    )

    print("\nCompute pairwise cross-model similarities:")

    conditions = [
        EXPECTED_CONDITIONS[source_key]
        for source_key in SOURCE_KEYS
    ]

    comparison_df = build_pairwise_comparisons(
        metadata_df=combined_metadata_df,
        xai_ids=normalized_xai_ids,
        conditions=conditions,
        threshold=float(
            config["heatmap_threshold"]
        ),
        experiment_id=config["experiment_id"],
    )

    expected_comparison_rows = (
        expected_xai_count
        * 3
    )

    if len(comparison_df) != expected_comparison_rows:
        raise ValueError(
            "Unexpected comparison row count. "
            f"Expected {expected_comparison_rows}, "
            f"found {len(comparison_df)}."
        )

    comparison_path = (
        dirs["metrics_dir"]
        / f"{config['experiment_id']}_"
          "cross_model_xai_similarity.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "Saved pairwise cross-model similarity metrics:",
        comparison_path,
    )

    print("\nComparison statuses:")
    print(
        comparison_df[
            [
                "comparison",
                "comparison_status",
            ]
        ]
        .value_counts()
        .sort_index()
    )

    computed_df = comparison_df[
        comparison_df["comparison_status"] == "computed"
    ]

    if not computed_df.empty:
        degenerate_count = int(
            computed_df["is_degenerate_pair"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        print(
            "Degenerate computed heatmap pairs:",
            degenerate_count,
        )

    print("\nFinal output check:")
    print_output_check(dirs)


if __name__ == "__main__":
    main()