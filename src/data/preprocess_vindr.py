from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_ANNOTATION_COLUMNS = {
    "image_id",
    "class_id",
}


def load_vindr_annotations(train_csv: str | Path) -> pd.DataFrame:
    """Load and minimally validate the VinDr annotation table."""
    train_csv = Path(train_csv)

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    if not train_csv.is_file():
        raise ValueError(f"Configured train CSV is not a file: {train_csv}")

    df = pd.read_csv(train_csv)

    missing_columns = REQUIRED_ANNOTATION_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "VinDr annotation CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError(f"VinDr annotation CSV is empty: {train_csv}")

    if df["image_id"].isna().any():
        raise ValueError("VinDr annotation CSV contains missing image_id values.")

    if df["class_id"].isna().any():
        raise ValueError("VinDr annotation CSV contains missing class_id values.")

    print("Raw annotation table shape:", df.shape)
    print("Class distribution by annotation rows:")
    print(df["class_id"].value_counts().sort_index())
    print("Unique image IDs:", df["image_id"].nunique())
    print("Total annotation rows:", len(df))

    return df


def create_binary_image_labels(
    annotation_df: pd.DataFrame,
    train_image_dir: str | Path,
) -> pd.DataFrame:
    """Create one binary label per image.

    Label convention:
    - 0 = normal: all annotation class_id values are 14
    - 1 = abnormal: at least one class_id is not 14
    """
    train_image_dir = Path(train_image_dir)

    if not train_image_dir.exists():
        raise FileNotFoundError(
            f"Train image directory not found: {train_image_dir}"
        )

    if not train_image_dir.is_dir():
        raise NotADirectoryError(
            f"Train image path is not a directory: {train_image_dir}"
        )

    image_level_df = (
        annotation_df.groupby("image_id")["class_id"]
        .apply(
            lambda class_ids: int(
                any(class_id != 14 for class_id in class_ids)
            )
        )
        .reset_index()
        .rename(columns={"class_id": "binary_label"})
    )

    if image_level_df["image_id"].duplicated().any():
        raise ValueError(
            "Duplicate image IDs remain after image-level aggregation."
        )

    invalid_labels = set(
        image_level_df["binary_label"].dropna().unique()
    ).difference({0, 1})

    if invalid_labels:
        raise ValueError(
            f"Unexpected binary labels found: {sorted(invalid_labels)}"
        )

    image_level_df["dicom_path"] = image_level_df["image_id"].apply(
        lambda image_id: str(train_image_dir / f"{image_id}.dicom")
    )

    image_level_df["file_exists"] = image_level_df["dicom_path"].apply(
        lambda path: Path(path).is_file()
    )

    missing_count = int((~image_level_df["file_exists"]).sum())
    print("Missing DICOM files:", missing_count)

    image_level_df = (
        image_level_df[image_level_df["file_exists"]]
        .drop(columns=["file_exists"])
        .reset_index(drop=True)
    )

    if image_level_df.empty:
        raise ValueError(
            "No usable VinDr images remain after checking DICOM paths."
        )

    print("Image-level table shape:", image_level_df.shape)
    print("Binary label counts:")
    print(image_level_df["binary_label"].value_counts().sort_index())
    print("Binary label proportions:")
    print(
        image_level_df["binary_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    return image_level_df


def validate_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> None:
    """Check that train and validation splits are usable and disjoint."""
    if train_df.empty:
        raise ValueError("Training split is empty.")

    if val_df.empty:
        raise ValueError("Validation split is empty.")

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])

    overlap = train_ids.intersection(val_ids)

    if overlap:
        example_ids = sorted(overlap)[:5]
        raise ValueError(
            "Train/validation leakage detected. "
            f"Overlapping image IDs include: {example_ids}"
        )

    if train_df["image_id"].duplicated().any():
        raise ValueError("Training split contains duplicate image IDs.")

    if val_df["image_id"].duplicated().any():
        raise ValueError("Validation split contains duplicate image IDs.")

    print("Split integrity check passed.")
    print("Train images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Image overlap:", 0)


def make_debug_split(
    image_df: pd.DataFrame,
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the same fixed-size stratified split used in Phase 5."""
    if train_size <= 0:
        raise ValueError("train_size must be greater than zero.")

    if val_size <= 0:
        raise ValueError("val_size must be greater than zero.")

    total_size = train_size + val_size

    if total_size > len(image_df):
        raise ValueError(
            f"Requested debug size {total_size} exceeds the "
            f"available {len(image_df)} images."
        )

    debug_df, _ = train_test_split(
        image_df,
        train_size=total_size,
        stratify=image_df["binary_label"],
        random_state=seed,
    )

    train_df, val_df = train_test_split(
        debug_df,
        train_size=train_size,
        test_size=val_size,
        stratify=debug_df["binary_label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    validate_split_integrity(train_df, val_df)

    print("Debug train shape:", train_df.shape)
    print("Debug val shape:", val_df.shape)

    print("\nTrain label distribution:")
    print(train_df["binary_label"].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(val_df["binary_label"].value_counts().sort_index())

    return train_df, val_df


def make_full_split(
    image_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a full-dataset stratified train/validation split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "val_fraction must be between 0 and 1, exclusive."
        )

    train_df, val_df = train_test_split(
        image_df,
        test_size=val_fraction,
        stratify=image_df["binary_label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    validate_split_integrity(train_df, val_df)

    print("Full train shape:", train_df.shape)
    print("Full validation shape:", val_df.shape)

    print("\nFull train label distribution:")
    print(train_df["binary_label"].value_counts().sort_index())

    print("\nFull validation label distribution:")
    print(val_df["binary_label"].value_counts().sort_index())

    print("\nFull train label proportions:")
    print(
        train_df["binary_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print("\nFull validation label proportions:")
    print(
        val_df["binary_label"]
        .value_counts(normalize=True)
        .sort_index()
    )

    return train_df, val_df


def select_fixed_xai_ids(
    val_df: pd.DataFrame,
    xai_subset_size: int,
    seed: int,
    stratified: bool = False,
) -> List[str]:
    """Select reproducible validation images for XAI analysis."""
    if xai_subset_size <= 0:
        raise ValueError("xai_subset_size must be greater than zero.")

    sample_size = min(xai_subset_size, len(val_df))

    if sample_size == len(val_df):
        sampled_df = val_df.copy()
    elif stratified:
        sampled_df, _ = train_test_split(
            val_df,
            train_size=sample_size,
            stratify=val_df["binary_label"],
            random_state=seed,
        )
    else:
        sampled_df = val_df.sample(
            n=sample_size,
            random_state=seed,
        )

    sampled_df = sampled_df.reset_index(drop=True)

    print("Fixed XAI subset size:", len(sampled_df))
    print("Fixed XAI label distribution:")
    print(sampled_df["binary_label"].value_counts().sort_index())

    return sampled_df["image_id"].tolist()


def save_split_outputs(
    image_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    xai_ids: List[str],
    interim_dir: Path,
    split_prefix: str,
) -> None:
    """Save image labels, data splits, and the fixed XAI subset."""
    interim_dir.mkdir(parents=True, exist_ok=True)

    image_level_path = interim_dir / "vindr_binary_image_level.csv"
    train_split_path = interim_dir / f"{split_prefix}_train_split.csv"
    val_split_path = interim_dir / f"{split_prefix}_val_split.csv"
    xai_ids_path = interim_dir / f"{split_prefix}_xai_image_ids.csv"

    image_df.to_csv(image_level_path, index=False)
    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    xai_df = val_df[val_df["image_id"].isin(xai_ids)][
        ["image_id", "binary_label", "dicom_path"]
    ].copy()

    xai_df.to_csv(xai_ids_path, index=False)

    print("Saved image-level labels to:", image_level_path)
    print("Saved train split:", train_split_path)
    print("Saved validation split:", val_split_path)
    print("Saved fixed XAI image IDs to:", xai_ids_path)

def load_existing_full_split(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load a previously saved canonical train/validation/XAI split."""
    required_path_keys = [
        "canonical_train_split_path",
        "canonical_val_split_path",
        "canonical_xai_ids_path",
    ]

    missing_keys = [
        key
        for key in required_path_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Split reuse is enabled, but these config keys are missing: "
            + ", ".join(missing_keys)
        )

    train_path = Path(
        config["canonical_train_split_path"]
    )
    val_path = Path(
        config["canonical_val_split_path"]
    )
    xai_path = Path(
        config["canonical_xai_ids_path"]
    )

    for path in [train_path, val_path, xai_path]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Canonical split file not found: {path}"
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    xai_df = pd.read_csv(xai_path)

    required_split_columns = {
        "image_id",
        "binary_label",
        "dicom_path",
    }

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
    ]:
        missing_columns = required_split_columns.difference(
            split_df.columns
        )

        if missing_columns:
            raise ValueError(
                f"Canonical {split_name} split is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

    if "image_id" not in xai_df.columns:
        raise ValueError(
            "Canonical XAI file must contain an image_id column."
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    validate_split_integrity(
        train_df=train_df,
        val_df=val_df,
    )

    xai_ids = (
        xai_df["image_id"]
        .astype(str)
        .tolist()
    )

    if not xai_ids:
        raise ValueError(
            "Canonical XAI image ID file is empty."
        )

    if len(xai_ids) != len(set(xai_ids)):
        raise ValueError(
            "Canonical XAI image ID file contains duplicates."
        )

    val_ids = set(
        val_df["image_id"].astype(str)
    )

    missing_xai_ids = [
        image_id
        for image_id in xai_ids
        if image_id not in val_ids
    ]

    if missing_xai_ids:
        raise ValueError(
            f"{len(missing_xai_ids)} canonical XAI image IDs are "
            "not present in the validation split. Examples: "
            f"{missing_xai_ids[:5]}"
        )

    print("Loaded canonical training split:", train_path)
    print("Loaded canonical validation split:", val_path)
    print("Loaded canonical XAI subset:", xai_path)
    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("XAI samples:", len(xai_ids))

    return train_df, val_df, xai_ids

def prepare_vindr_debug_data(
    config: Dict[str, Any],
    interim_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare the confirmed Phase 5 fixed-size debug data."""
    raw_df = load_vindr_annotations(config["train_csv"])

    image_df = create_binary_image_labels(
        annotation_df=raw_df,
        train_image_dir=config["train_image_dir"],
    )

    train_size = int(
        config.get(
            "subset_train_size",
            config.get("debug_train_size", 1000),
        )
    )

    val_size = int(
        config.get(
            "subset_val_size",
            config.get("debug_val_size", 250),
        )
    )

    train_df, val_df = make_debug_split(
        image_df=image_df,
        train_size=train_size,
        val_size=val_size,
        seed=int(config["seed"]),
    )

    xai_subset_size = int(
        config.get(
            "xai_subset_size",
            config.get("num_gradcam_images", 12),
        )
    )

    xai_ids = select_fixed_xai_ids(
        val_df=val_df,
        xai_subset_size=xai_subset_size,
        seed=int(config["seed"]),
        stratified=False,
    )

    save_split_outputs(
        image_df=image_df,
        train_df=train_df,
        val_df=val_df,
        xai_ids=xai_ids,
        interim_dir=interim_dir,
        split_prefix="debug",
    )

    return train_df, val_df, xai_ids


def prepare_vindr_full_data(
    config: Dict[str, Any],
    interim_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare or load the full VinDr binary dataset for Phase 6."""
    reuse_split_files = bool(
        config.get("reuse_split_files", False)
    )

    if reuse_split_files:
        return load_existing_full_split(config)

    raw_df = load_vindr_annotations(config["train_csv"])

    image_df = create_binary_image_labels(
        annotation_df=raw_df,
        train_image_dir=config["train_image_dir"],
    )

    val_fraction = float(config.get("val_fraction", 0.2))

    train_df, val_df = make_full_split(
        image_df=image_df,
        val_fraction=val_fraction,
        seed=int(config["seed"]),
    )

    xai_subset_size = int(config.get("xai_subset_size", 200))

    xai_ids = select_fixed_xai_ids(
        val_df=val_df,
        xai_subset_size=xai_subset_size,
        seed=int(config["seed"]),
        stratified=True,
    )

    split_prefix = config.get(
        "split_file_prefix",
        config["experiment_id"],
    )

    save_split_outputs(
        image_df=image_df,
        train_df=train_df,
        val_df=val_df,
        xai_ids=xai_ids,
        interim_dir=interim_dir,
        split_prefix=split_prefix,
    )

    return train_df, val_df, xai_ids