from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.split_clients import validate_client_splits

# Pre-declared protocol values, not tuned against validation performance.
QUANTITY_SIZES = (1800, 2100, 2400, 2700, 3000)
QUANTITY_ABNORMAL = (527, 615, 703, 791, 879)
QUANTITY_NORMAL = (1273, 1485, 1697, 1909, 2121)
CLIENT_GAMMAS = (0.93, 1.14, 1.00, 0.86, 1.07)
CONDITIONS = ("feature_only", "quantity_only", "feature_quantity")


class GammaTransform:
    """Picklable torchvision-compatible PIL transform for a single client."""

    def __init__(self, gamma: float):
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("SUP003 gamma must be positive and finite")

    def __call__(self, image):
        from torchvision.transforms.functional import adjust_gamma
        return adjust_gamma(image, gamma=self.gamma, gain=1.0)

    def __repr__(self):
        return f"GammaTransform(gamma={self.gamma:.6g})"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_quantity_splits(train_df: pd.DataFrame, seed: int) -> list[pd.DataFrame]:
    """Deterministic Q1 splits: fixed class quotas with no replacement."""
    required = {"image_id", "binary_label", "dicom_path"}
    if not required.issubset(train_df.columns):
        raise ValueError(f"Missing canonical fields: {sorted(required - set(train_df.columns))}")
    if train_df["image_id"].duplicated().any():
        raise ValueError("Source train membership has duplicate IDs")
    if len(train_df) != 12000:
        raise ValueError(f"Expected exactly 12000 canonical training rows, got {len(train_df)}")

    counts = train_df["binary_label"].value_counts().to_dict()
    if counts != {0: 8485, 1: 3515}:
        raise ValueError(f"Unexpected source label counts: {counts}")
    rng = np.random.default_rng(int(seed))
    class_parts: dict[int, list[pd.DataFrame]] = {}
    for label, sizes in ((0, QUANTITY_NORMAL), (1, QUANTITY_ABNORMAL)):
        ordered = (train_df.loc[train_df["binary_label"] == label]
                   .sort_values("image_id", kind="stable").reset_index(drop=True))
        ordered = ordered.iloc[rng.permutation(len(ordered))].reset_index(drop=True)
        boundaries = np.cumsum((0, *sizes))
        class_parts[label] = [ordered.iloc[boundaries[i]:boundaries[i+1]]
                              for i in range(len(sizes))]

    clients = []
    for cid, total in enumerate(QUANTITY_SIZES):
        client = pd.concat([class_parts[0][cid], class_parts[1][cid]], ignore_index=True)
        client = client.iloc[rng.permutation(len(client))].reset_index(drop=True)
        if len(client) != total:
            raise AssertionError(f"Client {cid} quota mismatch")
        clients.append(client)
    verify_client_memberships(train_df, clients, None)
    return clients


def verify_client_memberships(
    train_df: pd.DataFrame,
    clients: Sequence[pd.DataFrame],
    val_df: pd.DataFrame | None,
) -> None:
    validate_client_splits(train_df, list(clients))
    if len(clients) != 5:
        raise ValueError("SUP003 requires exactly 5 clients")
    source = train_df.set_index("image_id")["binary_label"]
    for cid, client in enumerate(clients):
        if not client["image_id"].map(source).eq(client["binary_label"]).all():
            raise ValueError(f"Client {cid} labels differ from canonical source")
    if val_df is not None:
        if set(train_df["image_id"]) & set(val_df["image_id"]):
            raise ValueError("Canonical train/validation overlap")
        if val_df["image_id"].duplicated().any():
            raise ValueError("Canonical validation duplicates")


def verify_condition(
    condition: str,
    clients: Sequence[pd.DataFrame],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    iid_clients: Sequence[pd.DataFrame] | None = None,
) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported SUP003 condition: {condition}")
    verify_client_memberships(train_df, clients, val_df)
    if condition == "feature_only":
        for cid, client in enumerate(clients):
            counts = client["binary_label"].value_counts().to_dict()
            if (len(client), counts.get(0, 0), counts.get(1, 0)) != (2400, 1697, 703):
                raise ValueError(f"Feature-only client {cid} is not the IID quota")
            if iid_clients is not None and set(client["image_id"]) != set(iid_clients[cid]["image_id"]):
                raise ValueError(f"Feature-only client {cid} does not match IID membership")
    else:
        for cid, client in enumerate(clients):
            counts = client["binary_label"].value_counts().to_dict()
            actual = (len(client), counts.get(0, 0), counts.get(1, 0))
            expected = (QUANTITY_SIZES[cid], QUANTITY_NORMAL[cid], QUANTITY_ABNORMAL[cid])
            if actual != expected:
                raise ValueError(f"Q1 client {cid} mismatch: {actual} != {expected}")


def client_manifest(clients: Sequence[pd.DataFrame], condition: str) -> pd.DataFrame:
    from src.data.split_clients import build_client_distribution_summary
    summary = build_client_distribution_summary(list(clients), f"sup003_{condition}")
    summary["gamma"] = CLIENT_GAMMAS if condition in ("feature_only", "feature_quantity") else [1.0] * 5
    summary["num_batches_per_epoch"] = (summary["num_samples"] + 15) // 16
    return summary
