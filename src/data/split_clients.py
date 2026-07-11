from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def validate_client_splits(
    source_df: pd.DataFrame,
    client_dfs: List[pd.DataFrame],
) -> None:
    """Validate that client splits are complete, unique, and disjoint."""
    if not client_dfs:
        raise ValueError("No client splits were created.")

    source_ids = set(source_df["image_id"])
    collected_ids = []

    for client_id, client_df in enumerate(client_dfs):
        if client_df.empty:
            raise ValueError(f"Client {client_id} has no samples.")

        if client_df["image_id"].duplicated().any():
            raise ValueError(
                f"Client {client_id} contains duplicate image IDs."
            )

        collected_ids.extend(client_df["image_id"].tolist())

    if len(collected_ids) != len(set(collected_ids)):
        raise ValueError(
            "At least one image was assigned to multiple clients."
        )

    collected_id_set = set(collected_ids)

    missing_ids = source_ids.difference(collected_id_set)
    unexpected_ids = collected_id_set.difference(source_ids)

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} training images were not assigned "
            "to any client."
        )

    if unexpected_ids:
        raise ValueError(
            f"{len(unexpected_ids)} unexpected image IDs appeared "
            "in client splits."
        )

    print("Client split integrity check passed.")
    print("Total assigned samples:", len(collected_ids))
    print("Unique assigned samples:", len(collected_id_set))


def build_client_distribution_summary(
    client_dfs: List[pd.DataFrame],
    split_type: str,
) -> pd.DataFrame:
    """Create one summary row per client."""
    rows = []

    for client_id, client_df in enumerate(client_dfs):
        class_counts = (
            client_df["binary_label"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        normal_count = int(class_counts.loc[0])
        abnormal_count = int(class_counts.loc[1])
        total_count = len(client_df)

        rows.append({
            "client_id": client_id,
            "split_type": split_type,
            "num_samples": total_count,
            "normal_count": normal_count,
            "abnormal_count": abnormal_count,
            "normal_rate": normal_count / total_count,
            "abnormal_rate": abnormal_count / total_count,
        })

    return pd.DataFrame(rows)


def save_client_splits(
    client_dfs: List[pd.DataFrame],
    interim_dir: Path,
    experiment_id: str,
    split_type: str,
) -> Tuple[List[Path], Path]:
    """Save client membership files and a distribution summary."""
    client_split_dir = (
        interim_dir
        / "client_splits"
        / experiment_id
    )
    client_split_dir.mkdir(parents=True, exist_ok=True)

    split_paths = []

    for client_id, client_df in enumerate(client_dfs):
        split_path = (
            client_split_dir
            / f"client_{client_id}_{split_type}.csv"
        )

        client_df.to_csv(split_path, index=False)
        split_paths.append(split_path)

        print("Saved client split:", split_path)

    summary_df = build_client_distribution_summary(
        client_dfs=client_dfs,
        split_type=split_type,
    )

    summary_path = (
        client_split_dir
        / f"{experiment_id}_{split_type}_distribution.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    print("Saved client distribution summary:", summary_path)
    print(summary_df)

    return split_paths, summary_path


def create_iid_client_splits(
    train_df: pd.DataFrame,
    num_clients: int,
    seed: int,
    interim_dir: Optional[Path] = None,
    experiment_id: str = "debug",
) -> List[pd.DataFrame]:
    """Create stratified IID client splits.

    The splitting method is the same as Phase 5: StratifiedKFold with
    shuffling and a fixed seed.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")

    if len(train_df) < num_clients:
        raise ValueError(
            "Training set contains fewer samples than clients."
        )

    class_counts = train_df["binary_label"].value_counts()

    if (class_counts < num_clients).any():
        raise ValueError(
            "Each class must contain at least num_clients samples "
            "for stratified client splitting."
        )

    skf = StratifiedKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=seed,
    )

    client_dfs = []
    labels = train_df["binary_label"].to_numpy()

    for client_id, (_, client_indices) in enumerate(
        skf.split(train_df, labels)
    ):
        client_df = (
            train_df
            .iloc[client_indices]
            .reset_index(drop=True)
        )

        client_dfs.append(client_df)

        print(f"Client {client_id}: {len(client_df)} samples")
        print(
            client_df["binary_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    validate_client_splits(
        source_df=train_df,
        client_dfs=client_dfs,
    )

    if interim_dir is not None:
        save_client_splits(
            client_dfs=client_dfs,
            interim_dir=interim_dir,
            experiment_id=experiment_id,
            split_type="iid",
        )

    return client_dfs


def _allocate_class_to_clients(
    class_df: pd.DataFrame,
    client_weights: np.ndarray,
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    """Allocate one class across clients according to normalized weights."""
    shuffled_df = class_df.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)

    normalized_weights = client_weights / client_weights.sum()

    raw_counts = normalized_weights * len(shuffled_df)
    counts = np.floor(raw_counts).astype(int)

    remainder = len(shuffled_df) - counts.sum()

    if remainder > 0:
        fractional_order = np.argsort(
            raw_counts - counts
        )[::-1]

        for index in fractional_order[:remainder]:
            counts[index] += 1

    allocations = []
    start_index = 0

    for count in counts:
        end_index = start_index + int(count)

        allocations.append(
            shuffled_df.iloc[start_index:end_index].copy()
        )

        start_index = end_index

    return allocations


def create_label_skew_client_splits(
    train_df: pd.DataFrame,
    num_clients: int,
    seed: int,
    interim_dir: Optional[Path] = None,
    experiment_id: str = "experiment",
    skew_strength: float = 4.0,
) -> List[pd.DataFrame]:
    """Create reproducible non-IID splits using binary label skew.

    Clients at opposite ends receive different class preferences, while
    middle clients are closer to balanced.

    `skew_strength=1.0` is close to IID.
    Larger values create stronger class-ratio differences.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")

    if skew_strength < 1.0:
        raise ValueError(
            "skew_strength must be greater than or equal to 1.0."
        )

    normal_df = (
        train_df[train_df["binary_label"] == 0]
        .reset_index(drop=True)
    )

    abnormal_df = (
        train_df[train_df["binary_label"] == 1]
        .reset_index(drop=True)
    )

    if normal_df.empty or abnormal_df.empty:
        raise ValueError(
            "Both binary classes are required for label-skew splitting."
        )

    rng = np.random.default_rng(seed)

    normal_weights = np.linspace(
        skew_strength,
        1.0,
        num_clients,
    )

    abnormal_weights = normal_weights[::-1].copy()

    normal_allocations = _allocate_class_to_clients(
        class_df=normal_df,
        client_weights=normal_weights,
        rng=rng,
    )

    abnormal_allocations = _allocate_class_to_clients(
        class_df=abnormal_df,
        client_weights=abnormal_weights,
        rng=rng,
    )

    client_dfs = []

    for client_id in range(num_clients):
        client_df = pd.concat(
            [
                normal_allocations[client_id],
                abnormal_allocations[client_id],
            ],
            ignore_index=True,
        )

        client_df = client_df.sample(
            frac=1.0,
            random_state=seed + client_id,
        ).reset_index(drop=True)

        client_dfs.append(client_df)

        print(f"Client {client_id}: {len(client_df)} samples")
        print(
            client_df["binary_label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    validate_client_splits(
        source_df=train_df,
        client_dfs=client_dfs,
    )

    if interim_dir is not None:
        save_client_splits(
            client_dfs=client_dfs,
            interim_dir=interim_dir,
            experiment_id=experiment_id,
            split_type="non_iid_label_skew",
        )

    return client_dfs