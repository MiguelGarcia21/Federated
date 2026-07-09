from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_iid_client_splits(
    train_df: pd.DataFrame,
    num_clients: int,
    seed: int,
    interim_dir: Optional[Path] = None,
) -> List[pd.DataFrame]:
    """Create the same stratified IID client splits used in the notebook."""
    skf = StratifiedKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=seed,
    )

    client_dfs = []
    y = train_df["binary_label"].values

    for client_id, (_, client_idx) in enumerate(skf.split(train_df, y)):
        client_df = train_df.iloc[client_idx].reset_index(drop=True)
        client_dfs.append(client_df)

        print(f"Client {client_id}: {len(client_df)} samples")
        print(client_df["binary_label"].value_counts().sort_index().to_dict())

        if interim_dir is not None:
            client_split_path = interim_dir / f"debug_client_{client_id}_split.csv"
            client_df.to_csv(client_split_path, index=False)
            print("Saved client split:", client_split_path)

    return client_dfs
