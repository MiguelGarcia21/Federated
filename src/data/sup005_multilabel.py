"""SUP-005-only 14-finding label, dataloader and metric helpers.

PROPOSAL ONLY. Author manually creates src/data/sup005_multilabel.py.
No existing binary dataset/source is modified.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.datasets import (
    VinDrBinaryDataset,
    get_train_transform,
    get_val_transform,
    validate_image_cache,
)

CLASS_NAMES = (
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
    "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
    "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
    "Pulmonary fibrosis",
)
LABEL_COLUMNS = ["finding_%02d" % i for i in range(14)]


def build_labels(train_csv: Path, canonical_train: Path, canonical_val: Path
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw reader/box annotations; union per image, never per box."""
    raw = pd.read_csv(train_csv, dtype={"image_id": str, "rad_id": str})
    needed = {"image_id", "class_id", "class_name", "rad_id"}
    if not needed.issubset(raw.columns) or raw.empty:
        raise ValueError("Original annotations are missing required reader/class columns")
    if raw[list(needed)].isna().any().any():
        raise ValueError("Original annotations contain null reader/image/class values")
    raw["class_id"] = pd.to_numeric(raw["class_id"], errors="raise")
    if not np.isfinite(raw["class_id"]).all() or not np.equal(raw["class_id"], np.floor(raw["class_id"])).all():
        raise ValueError("Non-integer class ID")
    raw["class_id"] = raw["class_id"].astype(int)
    if not set(raw["class_id"]).issubset(set(range(15))):
        raise ValueError("Expected only class IDs 0..14")
    for c, name in enumerate(CLASS_NAMES):
        names = raw.loc[raw.class_id.eq(c), "class_name"].unique().tolist()
        if names != [name]:
            raise ValueError("class_id=%d name mismatch: %r" % (c, names))
    if set(raw.loc[raw.class_id.eq(14), "class_name"].unique()) != {"No finding"}:
        raise ValueError("class_id=14 must map to No finding")
    readers_per_image = raw.groupby("image_id")["rad_id"].nunique()
    if not readers_per_image.eq(3).all():
        raise ValueError("Expected exactly three annotating radiologists per image")
    normal_ids = set(raw.loc[raw.class_id.eq(14), "image_id"])
    positive = raw[raw.class_id.lt(14)][["image_id", "rad_id", "class_id"]].drop_duplicates()
    abnormal_ids = set(positive["image_id"])
    if normal_ids & abnormal_ids:
        raise ValueError("No-finding and pathology records conflict")
    if normal_ids | abnormal_ids != set(readers_per_image.index):
        raise ValueError("Unaccounted-for image labels")
    votes = positive.groupby(["image_id", "class_id"])["rad_id"].nunique()
    if not votes.le(3).all():
        raise ValueError("Reader vote count exceeded 3")
    mat = pd.crosstab(positive["image_id"], positive["class_id"]).gt(0).astype(np.uint8)
    mat = mat.reindex(index=readers_per_image.index, columns=range(14), fill_value=0)
    mat.columns = LABEL_COLUMNS
    mat.index.name = "image_id"
    labels = mat.reset_index()
    all_manifest = []
    for name, path, n_expected in (("train", canonical_train, 12000), ("val", canonical_val, 3000)):
        split = pd.read_csv(path, dtype={"image_id": str})
        if len(split) != n_expected or split.image_id.isna().any() or split.image_id.duplicated().any():
            raise ValueError("Invalid canonical %s split size/uniqueness" % name)
        if not {"image_id", "binary_label", "dicom_path"}.issubset(split.columns):
            raise ValueError("Canonical split missing required binary columns")
        merged = split.merge(labels, on="image_id", how="left", validate="one_to_one", indicator=True)
        if not merged._merge.eq("both").all() or merged[LABEL_COLUMNS].isna().any().any():
            raise ValueError("Canonical %s split missing original annotation labels" % name)
        merged = merged.drop(columns=["_merge"])
        merged[LABEL_COLUMNS] = merged[LABEL_COLUMNS].astype(np.uint8)
        any_positive = merged[LABEL_COLUMNS].max(axis=1)
        if not np.array_equal(any_positive.to_numpy(), merged.binary_label.to_numpy()):
            raise ValueError("Canonical binary labels disagree with finding union")
        if any(Path(p).stem != i for p, i in zip(merged.dicom_path, merged.image_id)):
            raise ValueError("Canonical image ID / DICOM filename mismatch")
        all_manifest.append(merged)
    train, val = all_manifest
    if set(train.image_id) & set(val.image_id):
        raise ValueError("Train/validation image leakage")
    if set(train.image_id) | set(val.image_id) != set(labels.image_id):
        raise ValueError("Canonical split union does not cover original 15k annotations")
    if not train[LABEL_COLUMNS].sum().gt(0).all() or not val[LABEL_COLUMNS].sum().gt(0).all():
        raise ValueError("A 14-finding category is absent in train or validation")
    return train, val


class VinDrMultilabelDataset(VinDrBinaryDataset):
    """Reuse identical image decoder/transforms, changing only target vector."""
    def __init__(self, df: pd.DataFrame, **kwargs):
        if not set(LABEL_COLUMNS).issubset(df.columns):
            raise ValueError("14 finding columns missing")
        labels = df[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError("Finding labels must be complete binary values")
        super().__init__(df=df, **kwargs)
        self.labels = torch.as_tensor(labels.copy(), dtype=torch.float32)

    def __getitem__(self, idx):
        image, unused_binary_label, image_id = super().__getitem__(idx)
        return image, self.labels[idx], image_id


def make_loader(df: pd.DataFrame, cfg: Dict, training: bool) -> DataLoader:
    size = int(cfg["image_size"])
    transform = (get_train_transform(size, bool(cfg.get("horizontal_flip", True)),
                                     float(cfg.get("rotation_degrees", 0.0)))
                 if training else get_val_transform(size))
    cache = cfg.get("image_cache_dir") if cfg.get("image_source") == "cache" else None
    dataset = VinDrMultilabelDataset(
        df=df, transform=transform, image_source=cfg.get("image_source", "dicom"),
        cache_dir=cache, cache_extension=cfg.get("image_cache_extension", ".png"),
    )
    workers = int(cfg.get("fl_num_workers" if cfg["condition"] != "centralized" else "num_workers", 0))
    return DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=training,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, drop_last=False)


def train_pos_weight(train: pd.DataFrame) -> torch.Tensor:
    """Uncapped global per-class positive weight; fixed from training split alone."""
    pos = train[LABEL_COLUMNS].sum(axis=0).to_numpy(dtype=np.float64)
    n = len(train)
    if (pos <= 0).any() or (pos >= n).any():
        raise ValueError("Training labels need both positives and negatives per finding")
    return torch.tensor((n - pos) / pos, dtype=torch.float32)


def score_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
                      ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Per-class AP/ROC/F1 with explicit undefined-metric handling."""
    if y_true.shape != probabilities.shape or y_true.ndim != 2 or y_true.shape[1] != 14:
        raise ValueError("Expected matching (n,14) truth/probability matrices")
    if not np.isin(y_true, [0, 1]).all() or not np.isfinite(probabilities).all():
        raise ValueError("Invalid input to multilabel metrics")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be within [0,1]")
    prediction = probabilities >= threshold
    cls_rows = []
    for j, name in enumerate(CLASS_NAMES):
        yt = y_true[:, j].astype(bool)
        yp = prediction[:, j]
        pos = int(yt.sum()); neg = int((~yt).sum())
        tp = int((yt & yp).sum()); tn = int((~yt & ~yp).sum())
        fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
        defined = pos > 0 and neg > 0
        ap = float(average_precision_score(yt, probabilities[:, j])) if defined else float("nan")
        auc = float(roc_auc_score(yt, probabilities[:, j])) if defined else float("nan")
        cls_rows.append(dict(class_id=j, class_name=name, support=pos, negatives=neg,
                             tp=tp, tn=tn, fp=fp, fn=fn, precision=precision, recall=recall,
                             f1=f1, average_precision=ap, roc_auc=auc,
                             undefined_reason="" if defined else "missing positive or negative labels"))
    df = pd.DataFrame(cls_rows)
    sum_tp = int(df.tp.sum()); sum_fp = int(df.fp.sum()); sum_fn = int(df.fn.sum())
    micro_f1 = (2 * sum_tp / (2 * sum_tp + sum_fp + sum_fn)
                if 2 * sum_tp + sum_fp + sum_fn else float("nan"))
    flat_y = y_true.ravel()
    flat_p = probabilities.ravel()
    metrics = {
        "macro_ap": float(df.average_precision.mean(skipna=True)),
        "macro_roc_auc": float(df.roc_auc.mean(skipna=True)),
        "macro_f1": float(df.f1.mean(skipna=True)),
        "micro_ap": float(average_precision_score(flat_y, flat_p)) if 0 < flat_y.sum() < len(flat_y) else float("nan"),
        "micro_f1": float(micro_f1),
        "exact_match": float(np.all(prediction == y_true, axis=1).mean()),
        "no_finding_exact_match": (float(np.all(~prediction[np.all(y_true == 0, axis=1)], axis=1).mean())
                                   if np.any(np.all(y_true == 0, axis=1)) else float("nan")),
        "num_no_finding_images": int(np.all(y_true == 0, axis=1).sum()),
        "hamming_loss": float(np.not_equal(prediction, y_true).mean()),
        "num_valid_ap_classes": int(df.average_precision.notna().sum()),
        "num_samples": int(y_true.shape[0]),
        "threshold": float(threshold),
    }
    return metrics, df


@torch.no_grad()
def evaluate_multilabel(model, loader, criterion, device) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    model.eval()
    y_parts: List[np.ndarray] = []
    p_parts: List[np.ndarray] = []
    ids: List[str] = []
    summed_loss = 0.0
    for image, target, batch_ids in loader:
        image, target = image.to(device), target.to(device)
        logits = model(image)
        if logits.shape != target.shape or target.shape[1] != 14:
            raise RuntimeError("14-logit model / float-label shape mismatch")
        summed_loss += float(criterion(logits, target).item()) * len(batch_ids)
        y_parts.append(target.cpu().numpy())
        p_parts.append(torch.sigmoid(logits).cpu().numpy())
        ids.extend([str(x) for x in batch_ids])
    y = np.concatenate(y_parts)
    p = np.concatenate(p_parts)
    if len(ids) != len(loader.dataset) or len(ids) != len(set(ids)):
        raise RuntimeError("Validation did not produce unique per-image predictions")
    metrics, per_class = score_predictions(y, p)
    metrics["loss"] = summed_loss / len(ids)
    preds = pd.DataFrame({"image_id": ids})
    for j in range(14):
        preds["target_%02d" % j] = y[:, j].astype(np.uint8)
        preds["probability_%02d" % j] = p[:, j]
        preds["prediction_%02d" % j] = (p[:, j] >= 0.5).astype(np.uint8)
    return metrics, per_class, preds


def train_epoch_multilabel(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_sum = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != labels.shape or labels.shape[1] != 14 or labels.dtype != torch.float32:
            raise RuntimeError("Expected float32 labels matching (batch,14) logits")
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite multilabel training loss")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * images.shape[0]
    return loss_sum / len(loader.dataset)
