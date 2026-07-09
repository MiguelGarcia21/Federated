from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def load_vindr_annotations(train_csv: str | Path) -> pd.DataFrame:
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)

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
    """Create image-level binary labels.

    Label convention from the Kaggle notebook:
    - 0 = normal image, all annotation class_id values are 14
    - 1 = abnormal image, at least one class_id is not 14
    """
    train_image_dir = Path(train_image_dir)

    image_level_df = (
        annotation_df
        .groupby("image_id")["class_id"]
        .apply(lambda class_ids: int(any(class_id != 14 for class_id in class_ids)))
        .reset_index()
        .rename(columns={"class_id": "binary_label"})
    )

    image_level_df["dicom_path"] = image_level_df["image_id"].apply(
        lambda image_id: str(train_image_dir / f"{image_id}.dicom")
    )

    image_level_df["file_exists"] = image_level_df["dicom_path"].apply(
        lambda path: Path(path).exists()
    )

    missing_count = (~image_level_df["file_exists"]).sum()
    print("Missing DICOM files:", missing_count)

    image_level_df = (
        image_level_df[image_level_df["file_exists"]]
        .drop(columns=["file_exists"])
        .reset_index(drop=True)
    )

    print("Image-level table shape:", image_level_df.shape)
    print("Binary label counts:")
    print(image_level_df["binary_label"].value_counts().sort_index())
    print("Binary label proportions:")
    print(image_level_df["binary_label"].value_counts(normalize=True).sort_index())

    return image_level_df


def make_debug_split(
    image_df: pd.DataFrame,
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_size = train_size + val_size

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

    print("Debug train shape:", train_df.shape)
    print("Debug val shape:", val_df.shape)

    print("\nTrain label distribution:")
    print(train_df["binary_label"].value_counts().sort_index())

    print("\nValidation label distribution:")
    print(val_df["binary_label"].value_counts().sort_index())

    return train_df, val_df


def select_fixed_xai_ids(
    val_df: pd.DataFrame,
    xai_subset_size: int,
    seed: int,
) -> List[str]:
    xai_ids = (
        val_df
        .sample(
            n=min(xai_subset_size, len(val_df)),
            random_state=seed,
        )["image_id"]
        .tolist()
    )
    return xai_ids


def prepare_vindr_debug_data(
    config: Dict[str, Any],
    interim_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load annotations, create binary labels, create debug splits, and save CSVs."""
    raw_df = load_vindr_annotations(config["train_csv"])

    image_df = create_binary_image_labels(
        annotation_df=raw_df,
        train_image_dir=config["train_image_dir"],
    )

    image_level_path = interim_dir / "vindr_binary_image_level.csv"
    image_df.to_csv(image_level_path, index=False)
    print("Saved image-level labels to:", image_level_path)

    train_size = int(config.get("subset_train_size", config.get("debug_train_size", 1000)))
    val_size = int(config.get("subset_val_size", config.get("debug_val_size", 250)))

    train_df, val_df = make_debug_split(
        image_df=image_df,
        train_size=train_size,
        val_size=val_size,
        seed=int(config["seed"]),
    )

    train_split_path = interim_dir / "debug_train_split.csv"
    val_split_path = interim_dir / "debug_val_split.csv"

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    print("Saved train split:", train_split_path)
    print("Saved val split:", val_split_path)

    xai_subset_size = int(config.get("xai_subset_size", config.get("num_gradcam_images", 12)))

    xai_ids = select_fixed_xai_ids(
        val_df=val_df,
        xai_subset_size=xai_subset_size,
        seed=int(config["seed"]),
    )

    xai_ids_path = interim_dir / "fixed_xai_image_ids.csv"
    pd.DataFrame({"image_id": xai_ids}).to_csv(xai_ids_path, index=False)

    print("Saved fixed XAI image IDs to:", xai_ids_path)
    print(xai_ids)

    return train_df, val_df, xai_ids