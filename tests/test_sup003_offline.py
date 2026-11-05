"""No Flower or GPU: offline SUP003 partition and feature regression checks.

After manually adding the new modules and trainer patch:
  python tests/test_sup003_offline.py \
    --train data/interim/exp01_centralized_vindr_binary_seed42_train_split.csv \
    --val data/interim/exp01_centralized_vindr_binary_seed42_val_split.csv
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import get_train_transform, get_val_transform
from src.data.split_clients import create_iid_client_splits, create_label_skew_client_splits
from src.data.sup003_heterogeneity import (
    CLIENT_GAMMAS, GammaTransform, allocate_quantity_splits,
    verify_condition,
)


def signature(clients):
    return hashlib.sha256(json.dumps(
        [sorted(c["image_id"].astype(str).tolist()) for c in clients],
        separators=(",", ":")
    ).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    assert len(train) == 12000 and len(val) == 3000
    assert set(train["image_id"]).isdisjoint(set(val["image_id"]))

    results = {}
    for seed in (42, 7, 123):
        iid = create_iid_client_splits(train, 5, seed)
        # Existing label-skew pathway must remain usable; pass original parameter explicitly.
        legacy = create_label_skew_client_splits(train, 5, seed, skew_strength=4.0)
        assert len(legacy) == 5
        q_a = allocate_quantity_splits(train, seed)
        q_b = allocate_quantity_splits(train.sample(frac=1.0, random_state=seed), seed)
        assert signature(q_a) == signature(q_b), "Q1 changed when row order shuffled"
        verify_condition("feature_only", iid, train, val, iid_clients=iid)
        verify_condition("quantity_only", q_a, train, val)
        verify_condition("feature_quantity", q_b, train, val)
        assert signature(q_a) == signature(q_b)
        results[str(seed)] = {"iid_sha256": signature(iid), "quantity_sha256": signature(q_a)}

    # Transform test (synthetic PIL image): original path still available and the
    # proposed gamma is label-independent, deterministic and applied before ToTensor.
    pixels = np.tile(np.arange(0, 256, dtype=np.uint8), (256, 1))
    pil = Image.fromarray(pixels, mode="L").convert("RGB")
    base = get_train_transform(image_size=224, horizontal_flip=False, rotation_degrees=0)
    original = base(pil)
    for gamma in CLIENT_GAMMAS:
        gamma_tf = T.Compose([base.transforms[0], GammaTransform(gamma), *base.transforms[1:]])
        a = gamma_tf(pil)
        b = gamma_tf(pil)
        assert np.array_equal(a.numpy(), b.numpy()), f"gamma={gamma} not deterministic"
        if abs(gamma - 1.0) > 1e-12:
            assert not np.array_equal(a.numpy(), original.numpy()), "Gamma had no feature effect"
    assert get_val_transform(224) is not None
    print("SUP003_OFFLINE_PASS")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
