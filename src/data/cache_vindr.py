import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml
from PIL import Image

from src.data.datasets import (
    get_cached_image_path,
    normalize_cache_extension,
    read_dicom_as_uint8,
    validate_image_cache,
)


REQUIRED_CACHE_COLUMNS = {
    "image_id",
    "binary_label",
    "dicom_path",
}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load the YAML configuration used to build the image cache."""
    config_path = Path(config_path)

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


def load_canonical_splits(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the canonical Experiment 1 train and validation splits."""
    required_config_keys = [
        "canonical_train_split_path",
        "canonical_val_split_path",
    ]

    missing_keys = [
        key
        for key in required_config_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Cache generation requires these configuration keys: "
            + ", ".join(missing_keys)
        )

    train_path = Path(
        config["canonical_train_split_path"]
    )

    val_path = Path(
        config["canonical_val_split_path"]
    )

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Canonical training split not found: {train_path}"
        )

    if not val_path.is_file():
        raise FileNotFoundError(
            f"Canonical validation split not found: {val_path}"
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for split_name, split_df in [
        ("training", train_df),
        ("validation", val_df),
    ]:
        missing_columns = REQUIRED_CACHE_COLUMNS.difference(
            split_df.columns
        )

        if missing_columns:
            raise ValueError(
                f"Canonical {split_name} split is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        if split_df.empty:
            raise ValueError(
                f"Canonical {split_name} split is empty."
            )

        if split_df["image_id"].duplicated().any():
            raise ValueError(
                f"Canonical {split_name} split contains duplicate "
                "image IDs."
            )

    train_ids = set(
        train_df["image_id"].astype(str)
    )

    val_ids = set(
        val_df["image_id"].astype(str)
    )

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(
            "Canonical train/validation overlap detected. "
            f"Examples: {sorted(overlap)[:5]}"
        )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("Loaded canonical training split:", train_path)
    print("Loaded canonical validation split:", val_path)
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Total images:", len(train_df) + len(val_df))

    return train_df, val_df


def resolve_cache_dir(
    config: Dict[str, Any],
) -> Path:
    """Resolve the configured or default image-cache directory."""
    configured_cache_dir = config.get(
        "image_cache_dir"
    )

    if configured_cache_dir:
        return Path(configured_cache_dir)

    if "project_root" not in config:
        raise KeyError(
            "project_root is required when image_cache_dir "
            "is not explicitly configured."
        )

    image_size = int(config["image_size"])

    return (
        Path(config["project_root"])
        / "data"
        / "processed"
        / f"vindr_binary_{image_size}_png"
    )


def validate_source_files(
    image_df: pd.DataFrame,
) -> None:
    """Verify that all source DICOM files exist before conversion."""
    missing_paths: List[Path] = []

    for dicom_path in image_df["dicom_path"]:
        path = Path(dicom_path)

        if not path.is_file():
            missing_paths.append(path)

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} source DICOM files are missing. "
            f"Examples: {missing_paths[:5]}"
        )

    print("Source DICOM validation passed.")
    print("DICOM files checked:", len(image_df))


def cached_png_is_valid(
    cached_path: Path,
    image_size: int,
) -> bool:
    """Check whether an existing cached PNG can be reused."""
    if not cached_path.is_file():
        return False

    if cached_path.stat().st_size <= 0:
        return False

    try:
        with Image.open(cached_path) as image:
            image.load()

            return (
                image.format == "PNG"
                and image.size == (
                    image_size,
                    image_size,
                )
            )
    except Exception:
        return False


def cache_one_image(
    task: Tuple[
        str,
        str,
        str,
        str,
        int,
        bool,
    ],
) -> Dict[str, str]:
    """Convert one DICOM image to a lossless resized PNG."""
    (
        image_id,
        dicom_path,
        cache_dir,
        cache_extension,
        image_size,
        overwrite,
    ) = task

    cached_path = get_cached_image_path(
        cache_dir=cache_dir,
        image_id=image_id,
        cache_extension=cache_extension,
    )

    if (
        not overwrite
        and cached_png_is_valid(
            cached_path=cached_path,
            image_size=image_size,
        )
    ):
        return {
            "image_id": image_id,
            "status": "skipped",
            "cache_path": str(cached_path),
        }

    image_array = read_dicom_as_uint8(
        dicom_path
    )

    image = Image.fromarray(
        image_array
    ).convert("L")

    image = image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )

    temporary_path = cached_path.with_name(
        f"{cached_path.name}."
        f"{multiprocessing.current_process().pid}.tmp"
    )

    try:
        image.save(
            temporary_path,
            format="PNG",
            optimize=False,
            compress_level=1,
        )

        temporary_path.replace(cached_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if not cached_png_is_valid(
        cached_path=cached_path,
        image_size=image_size,
    ):
        raise RuntimeError(
            f"Generated cache file failed validation: {cached_path}"
        )

    return {
        "image_id": image_id,
        "status": "created",
        "cache_path": str(cached_path),
    }


def build_cache_manifest(
    image_df: pd.DataFrame,
    cache_dir: Path,
    cache_extension: str,
    image_size: int,
) -> Path:
    """Write the final manifest describing every cached image."""
    manifest_df = image_df[
        [
            "image_id",
            "binary_label",
            "dicom_path",
            "split",
        ]
    ].copy()

    manifest_df["cache_path"] = (
        manifest_df["image_id"]
        .astype(str)
        .apply(
            lambda image_id: str(
                get_cached_image_path(
                    cache_dir=cache_dir,
                    image_id=image_id,
                    cache_extension=cache_extension,
                )
            )
        )
    )

    manifest_df["cache_format"] = "png"
    manifest_df["cache_width"] = image_size
    manifest_df["cache_height"] = image_size

    manifest_path = cache_dir / "manifest.csv"

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    print("Saved cache manifest:", manifest_path)

    return manifest_path


def build_vindr_image_cache(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_workers: int,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """Build and validate the complete canonical VinDr PNG cache."""
    if num_workers <= 0:
        raise ValueError(
            "num_workers must be greater than zero."
        )

    image_size = int(config["image_size"])

    if image_size <= 0:
        raise ValueError(
            "image_size must be greater than zero."
        )

    cache_extension = normalize_cache_extension(
        config.get(
            "image_cache_extension",
            ".png",
        )
    )

    cache_dir = resolve_cache_dir(config)

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_cache_df = train_df.copy()
    train_cache_df["split"] = "train"

    val_cache_df = val_df.copy()
    val_cache_df["split"] = "validation"

    image_df = pd.concat(
        [
            train_cache_df,
            val_cache_df,
        ],
        ignore_index=True,
    )

    if image_df["image_id"].duplicated().any():
        duplicate_ids = (
            image_df.loc[
                image_df["image_id"].duplicated(
                    keep=False
                ),
                "image_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate image IDs found while constructing the "
            f"cache. Examples: {duplicate_ids[:5]}"
        )

    validate_source_files(image_df)

    print("Cache directory:", cache_dir)
    print("Cache image size:", image_size)
    print("Cache extension:", cache_extension)
    print("Cache workers:", num_workers)
    print("Overwrite existing files:", overwrite)

    tasks = [
        (
            str(row.image_id),
            str(row.dicom_path),
            str(cache_dir),
            cache_extension,
            image_size,
            overwrite,
        )
        for row in image_df.itertuples(
            index=False
        )
    ]

    created_count = 0
    skipped_count = 0
    completed_count = 0
    failures = []

    if num_workers == 1:
        for task in tasks:
            try:
                result = cache_one_image(task)

                if result["status"] == "created":
                    created_count += 1
                else:
                    skipped_count += 1

            except Exception as exc:
                failures.append(
                    (
                        task[0],
                        str(exc),
                    )
                )

            completed_count += 1

            if (
                completed_count % 250 == 0
                or completed_count == len(tasks)
            ):
                print(
                    "Cache progress: "
                    f"{completed_count}/{len(tasks)}"
                )

    else:
        multiprocessing_context = (
            multiprocessing.get_context("spawn")
        )

        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=multiprocessing_context,
        ) as executor:
            future_to_image_id = {
                executor.submit(
                    cache_one_image,
                    task,
                ): task[0]
                for task in tasks
            }

            for future in as_completed(
                future_to_image_id
            ):
                image_id = future_to_image_id[
                    future
                ]

                try:
                    result = future.result()

                    if result["status"] == "created":
                        created_count += 1
                    else:
                        skipped_count += 1

                except Exception as exc:
                    failures.append(
                        (
                            image_id,
                            str(exc),
                        )
                    )

                completed_count += 1

                if (
                    completed_count % 250 == 0
                    or completed_count == len(tasks)
                ):
                    print(
                        "Cache progress: "
                        f"{completed_count}/{len(tasks)}"
                    )

    print("Created cache files:", created_count)
    print("Reused cache files:", skipped_count)

    if failures:
        print("Cache failures:", len(failures))

        for image_id, error_message in failures[:10]:
            print(
                f"- {image_id}: {error_message}"
            )

        raise RuntimeError(
            f"Cache generation failed for "
            f"{len(failures)} images. "
            "Valid completed files were preserved and can be "
            "reused when the command is restarted."
        )

    validate_image_cache(
        df=image_df,
        cache_dir=cache_dir,
        cache_extension=cache_extension,
    )

    manifest_path = build_cache_manifest(
        image_df=image_df,
        cache_dir=cache_dir,
        cache_extension=cache_extension,
        image_size=image_size,
    )

    print("VinDr image cache completed.")
    print("Cached images:", len(image_df))

    return cache_dir, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable lossless PNG cache from the canonical "
            "VinDr train and validation DICOM images."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a YAML configuration containing the canonical "
            "split paths and image size."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel cache workers. When omitted, use "
            "cache_num_workers from the configuration or default to 8."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate valid existing cache files instead of "
            "reusing them."
        ),
    )

    args = parser.parse_args()

    config = load_config(args.config)

    train_df, val_df = load_canonical_splits(
        config
    )

    num_workers = (
        args.workers
        if args.workers is not None
        else int(
            config.get(
                "cache_num_workers",
                8,
            )
        )
    )

    cache_dir, manifest_path = (
        build_vindr_image_cache(
            config=config,
            train_df=train_df,
            val_df=val_df,
            num_workers=num_workers,
            overwrite=args.overwrite,
        )
    )

    print("\nCache generation output:")
    print("Cache directory:", cache_dir)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()