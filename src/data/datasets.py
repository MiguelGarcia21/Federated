from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as T
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

REQUIRED_DATASET_COLUMNS = {
    "image_id",
    "binary_label",
    "dicom_path",
}


def read_dicom_as_uint8(path: str | Path) -> np.ndarray:
    """Read a VinDr DICOM image and convert it to an 8-bit array.

    The preprocessing behavior remains consistent with the confirmed
    Phase 5 implementation.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    if not path.is_file():
        raise ValueError(f"DICOM path is not a file: {path}")

    try:
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array
    except Exception as exc:
        raise RuntimeError(
            f"Could not read DICOM pixel data from {path}: {exc}"
        ) from exc

    if pixel_array.size == 0:
        raise ValueError(f"DICOM image contains no pixel data: {path}")

    image = apply_voi_lut(pixel_array, ds).astype(np.float32)

    if not np.isfinite(image).all():
        raise ValueError(
            f"DICOM image contains non-finite pixel values: {path}"
        )

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    low, high = np.percentile(image, (1, 99))

    if high > low:
        image = np.clip(image, low, high)

    image_min = float(image.min())
    image_max = float(image.max())

    image = image - image_min

    if image_max > image_min:
        image = image / (image_max - image_min)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    image = (image * 255.0).clip(0, 255).astype(np.uint8)

    return image

def normalize_cache_extension(extension: str) -> str:
    """Normalize and validate the cached-image extension."""
    normalized_extension = str(extension).strip().lower()

    if not normalized_extension:
        raise ValueError(
            "cache_extension cannot be empty."
        )

    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"

    if normalized_extension != ".png":
        raise ValueError(
            "The VinDr image cache currently supports only "
            "lossless PNG files."
        )

    return normalized_extension


def get_cached_image_path(
    cache_dir: str | Path,
    image_id: str,
    cache_extension: str = ".png",
) -> Path:
    """Return the expected cached-image path for one image ID."""
    normalized_image_id = str(image_id).strip()

    if not normalized_image_id:
        raise ValueError("image_id cannot be empty.")

    normalized_extension = normalize_cache_extension(
        cache_extension
    )

    return (
        Path(cache_dir)
        / f"{normalized_image_id}{normalized_extension}"
    )


def validate_image_cache(
    df: pd.DataFrame,
    cache_dir: str | Path,
    cache_extension: str = ".png",
) -> None:
    """Validate that every unique image in a DataFrame has a cache file."""
    if "image_id" not in df.columns:
        raise ValueError(
            "Cache validation requires an image_id column."
        )

    if df.empty:
        raise ValueError(
            "Cannot validate an image cache for an empty DataFrame."
        )

    cache_dir = Path(cache_dir)

    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Image cache directory not found: {cache_dir}"
        )

    normalized_extension = normalize_cache_extension(
        cache_extension
    )

    image_ids = (
        df["image_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    missing_paths = []
    empty_paths = []

    for image_id in image_ids:
        cached_path = get_cached_image_path(
            cache_dir=cache_dir,
            image_id=image_id,
            cache_extension=normalized_extension,
        )

        if not cached_path.is_file():
            missing_paths.append(cached_path)
            continue

        if cached_path.stat().st_size <= 0:
            empty_paths.append(cached_path)

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} cached images are missing. "
            f"Examples: {missing_paths[:5]}"
        )

    if empty_paths:
        raise ValueError(
            f"{len(empty_paths)} cached images are empty. "
            f"Examples: {empty_paths[:5]}"
        )

    print("Image cache validation passed.")
    print("Cache directory:", cache_dir)
    print("Cached images checked:", len(image_ids))
    print("Cache extension:", normalized_extension)

class VinDrBinaryDataset(Dataset):
    """PyTorch Dataset for VinDr binary classification.

    Images can be read from the original DICOM files or from a
    preprocessed lossless PNG cache.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[object] = None,
        image_source: str = "dicom",
        cache_dir: Optional[str | Path] = None,
        cache_extension: str = ".png",
    ):
        missing_columns = REQUIRED_DATASET_COLUMNS.difference(
            df.columns
        )

        if missing_columns:
            raise ValueError(
                "Dataset DataFrame is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        if df.empty:
            raise ValueError("Dataset DataFrame is empty.")

        invalid_labels = set(
            df["binary_label"].dropna().unique()
        ).difference({0, 1})

        if invalid_labels:
            raise ValueError(
                f"Unexpected binary labels found: "
                f"{sorted(invalid_labels)}"
            )

        normalized_source = (
            str(image_source)
            .strip()
            .lower()
        )

        if normalized_source not in {"dicom", "cache"}:
            raise ValueError(
                "image_source must be either 'dicom' or 'cache'."
            )

        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.image_source = normalized_source
        self.cache_dir: Optional[Path] = None
        self.cache_extension = normalize_cache_extension(
            cache_extension
        )

        if self.image_source == "cache":
            if cache_dir is None:
                raise ValueError(
                    "cache_dir is required when "
                    "image_source='cache'."
                )

            resolved_cache_dir = Path(cache_dir)

            if not resolved_cache_dir.is_dir():
                raise FileNotFoundError(
                    "Configured image cache directory does not "
                    f"exist: {resolved_cache_dir}"
                )

            self.cache_dir = resolved_cache_dir

    def __len__(self) -> int:
        return len(self.df)

    def _load_dicom_image(
        self,
        dicom_path: str | Path,
    ) -> Image.Image:
        """Load one original DICOM image as an RGB PIL image."""
        image_array = read_dicom_as_uint8(dicom_path)

        return Image.fromarray(
            image_array
        ).convert("RGB")

    def _load_cached_image(
        self,
        image_id: str,
    ) -> Image.Image:
        """Load one cached lossless PNG image."""
        if self.cache_dir is None:
            raise RuntimeError(
                "Cache loading was requested without a cache directory."
            )

        cached_path = get_cached_image_path(
            cache_dir=self.cache_dir,
            image_id=image_id,
            cache_extension=self.cache_extension,
        )

        if not cached_path.is_file():
            raise FileNotFoundError(
                f"Cached image not found: {cached_path}"
            )

        try:
            with Image.open(cached_path) as cached_image:
                return cached_image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Could not read cached image {cached_path}: {exc}"
            ) from exc

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])

        if self.image_source == "cache":
            image = self._load_cached_image(image_id)
        else:
            image = self._load_dicom_image(
                row["dicom_path"]
            )

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            int(row["binary_label"]),
            dtype=torch.long,
        )

        return image, label, image_id


def get_train_transform(
    image_size: int,
    horizontal_flip: bool = True,
    rotation_degrees: float = 0.0,
):
    """Create the training transform.

    Defaults preserve Phase 5 behavior:
    resize + random horizontal flip + ImageNet normalization.

    Rotation is available for Phase 6 but remains disabled unless explicitly
    enabled in an experiment runner.
    """
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero.")

    transforms = [
        T.Resize((image_size, image_size)),
    ]

    if horizontal_flip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    if rotation_degrees > 0:
        transforms.append(
            T.RandomRotation(degrees=rotation_degrees)
        )

    transforms.extend([
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return T.Compose(transforms)


def get_val_transform(image_size: int):
    """Create the deterministic validation transform."""
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero.")

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def make_dataloader(
    df: pd.DataFrame,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = False,
    image_source: str = "dicom",
    cache_dir: Optional[str | Path] = None,
    cache_extension: str = ".png",
) -> DataLoader:
    """Create a DataLoader with safe defaults for debug and full runs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = VinDrBinaryDataset(
        df=df,
        transform=transform,
        image_source=image_source,
        cache_dir=cache_dir,
        cache_extension=cache_extension,
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )


def check_first_batch(loader: DataLoader) -> None:
    """Load and print one batch as an early data-pipeline check."""
    if len(loader.dataset) == 0:
        raise ValueError("Cannot inspect an empty DataLoader.")

    images, labels, image_ids = next(iter(loader))

    print("Batch image tensor:", images.shape)
    print("Batch image dtype:", images.dtype)
    print("Batch labels:", labels[:10])
    print("Example image IDs:", image_ids[:3])

    if not torch.isfinite(images).all():
        raise ValueError(
            "First batch contains NaN or infinite image values."
        )


def denormalize_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            "Expected image tensor with shape [3, height, width]."
        )

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=image_tensor.dtype,
        device=image_tensor.device,
    ).view(3, 1, 1)

    image = image_tensor.detach() * std + mean
    image = torch.clamp(image, 0, 1)

    return (
        image
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )