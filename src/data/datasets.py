from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_dicom_as_uint8(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))

    image = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    low, high = np.percentile(image, (1, 99))
    image = np.clip(image, low, high)

    image = image - image.min()
    image = image / (image.max() + 1e-8)

    image = (image * 255).astype(np.uint8)

    return image


class VinDrBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[object] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = read_dicom_as_uint8(row["dicom_path"])
        image = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["binary_label"], dtype=torch.long)
        image_id = row["image_id"]

        return image, label, image_id


def get_train_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_dataloader(
    df: pd.DataFrame,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = VinDrBinaryDataset(df=df, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def check_first_batch(loader: DataLoader) -> None:
    images, labels, image_ids = next(iter(loader))

    print("Batch image tensor:", images.shape)
    print("Batch labels:", labels[:10])
    print("Example image IDs:", image_ids[:3])


def denormalize_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean
    image = torch.clamp(image, 0, 1)

    return image.permute(1, 2, 0).numpy()