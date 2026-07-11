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


class VinDrBinaryDataset(Dataset):
    """PyTorch Dataset for VinDr binary normal/abnormal classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[object] = None,
    ):
        missing_columns = REQUIRED_DATASET_COLUMNS.difference(df.columns)

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
                f"Unexpected binary labels found: {sorted(invalid_labels)}"
            )

        self.df = df.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = read_dicom_as_uint8(row["dicom_path"])
        image = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(
            int(row["binary_label"]),
            dtype=torch.long,
        )

        image_id = str(row["image_id"])

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
) -> DataLoader:
    """Create a DataLoader with safe defaults for debug and full runs."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    dataset = VinDrBinaryDataset(
        df=df,
        transform=transform,
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