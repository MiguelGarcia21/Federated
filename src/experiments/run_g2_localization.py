import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.datasets import VinDrBinaryDataset, get_val_transform
from src.models.cnn import create_model
from src.xai.gradcam import GradCAM


# Resolve paths relative to the repository
REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPO_ROOT.parent / "scp_bridge"

INPUTS = BRIDGE / "g2_inputs"
CHECKPOINTS = BRIDGE / "checkpoints"
CACHE = BRIDGE / "vindr_xai59_png"

RUNS = {
    "centralized": (
        "exp01_centralized_vindr_binary_seed42",
        "centralized_resnet18_debug_best.pt",
    ),
    "iid": ("exp02_fedavg_iid_5clients_seed42", "flower_fedavg_round_50.pt"),
    "non_iid": ("exp03_fedavg_noniid_5clients_seed42", "flower_fedavg_round_50.pt"),
}
METRICS = ("pointing_game", "energy_in_box", "box_area_fraction", "dice_box_proxy", "iou_box_proxy")
SIZE = 224


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_inputs(limit):
    roster_file = INPUTS / "exp01_centralized_vindr_binary_seed42_xai_image_ids.csv"
    annotations_file = INPUTS / "train.csv"
    dimensions_file = INPUTS / "vindr_xai59_dicom_dimensions.csv"
    for file in (roster_file, annotations_file, dimensions_file):
        if not file.is_file():
            raise FileNotFoundError(f"Required input CSV not found: {file}")
    roster = pd.read_csv(roster_file, dtype={"image_id": str})
    annotations = pd.read_csv(annotations_file, dtype={"image_id": str})
    dimensions = pd.read_csv(dimensions_file, dtype={"image_id": str})
    if len(roster) != 200 or roster.image_id.duplicated().any():
        raise ValueError("Expected the original 200 unique-image XAI roster")
    positives = roster.loc[roster.binary_label == 1].sort_values("image_id").copy()
    if len(positives) != 59:
        raise ValueError(f"Expected 59 abnormal images; got {len(positives)}")
    if dimensions.image_id.duplicated().any():
        raise ValueError("Duplicate DICOM dimension IDs")
    if limit:
        positives = positives.head(limit).copy()
    if not CACHE.is_dir():
        raise FileNotFoundError(f"PNG cache not found: {CACHE}")
    missing = [sid for sid in positives.image_id if not (CACHE / f"{sid}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cached PNGs: {missing[:5]}")
    dimensions = dimensions.set_index("image_id")
    masks = {}
    for image_id in positives.image_id:
        if image_id not in dimensions.index:
            raise ValueError(f"Missing DICOM geometry for {image_id}")
        dim = dimensions.loc[image_id]
        if dim.status != "ok" or dim.width <= 0 or dim.height <= 0:
            raise ValueError(f"Invalid source geometry: {image_id}")
        boxes = annotations.loc[(annotations.image_id == image_id) & (annotations.class_id != 14)]
        if boxes.rad_id.nunique() != 3:
            raise ValueError(f"Expected 3 radiologists for {image_id}")
        masks[image_id] = make_reader_masks(boxes, float(dim.width), float(dim.height))
    return positives, masks, (roster_file, annotations_file, dimensions_file)


def make_reader_masks(boxes, width, height):
    x = (np.arange(SIZE) + 0.5)[None, :]
    y = (np.arange(SIZE) + 0.5)[:, None]
    out = {}
    for reader, rows in boxes.groupby("rad_id"):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        for r in rows.itertuples():
            x0, x1 = float(r.x_min), float(r.x_max)
            y0, y1 = float(r.y_min), float(r.y_max)
            if not np.isfinite([x0, x1, y0, y1]).all() or not (
                0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
            ):
                raise ValueError(f"Invalid annotation for reader {reader}")
            mask |= ((x >= x0 * SIZE / width) & (x < x1 * SIZE / width)
                     & (y >= y0 * SIZE / height) & (y < y1 * SIZE / height))
        if not mask.any():
            raise ValueError(f"Empty reader annotation for {reader}")
        out[str(reader)] = mask
    return out


def load_checkpoint(condition):
    run, filename = RUNS[condition]
    path = CHECKPOINTS / run / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    # Checkpoints must be author-owned/trusted; weights_only=False can unpickle data.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint has unexpected structure: {path}")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint config is invalid: {path}")
    if str(config.get("experiment_id")) != run or int(config.get("seed", -1)) != 42:
        raise ValueError(f"Checkpoint experiment/seed mismatch: {path}")
    if config.get("model_name", "resnet18") != "resnet18" or int(config.get("num_classes", 2)) != 2:
        raise ValueError(f"Checkpoint architecture mismatch: {path}")
    if condition != "centralized" and int(checkpoint.get("round", -1)) != 50:
        raise ValueError(f"Not a Round-50 checkpoint: {path}")
    model = create_model("resnet18", num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, path


def score(heatmap, mask):
    threshold_mask = heatmap >= 0.5
    intersection = np.count_nonzero(threshold_mask & mask)
    threshold_area = np.count_nonzero(threshold_mask)
    annotation_area = np.count_nonzero(mask)
    union = threshold_area + annotation_area - intersection
    return {
        "pointing_game": int(mask.flat[int(np.argmax(heatmap))]),
        "energy_in_box": float(heatmap[mask].sum(dtype=np.float64) / heatmap.sum(dtype=np.float64)),
        "box_area_fraction": annotation_area / mask.size,
        "dice_box_proxy": 2 * intersection / (threshold_area + annotation_area),
        "iou_box_proxy": intersection / union,
    }


def main():
    parser = argparse.ArgumentParser(description="Seed-42 Grad-CAM radiologist-box localization")
    parser.add_argument("--limit-images", type=int, default=0, help="3 for pilot; omit for full 59")
    parser.add_argument("--out", type=Path, required=True, help="NEW output directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit_images < 0 or args.limit_images > 59 or args.threads < 1:
        parser.error("Invalid --limit-images or --threads")
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}")
    torch.set_num_threads(args.threads)
    positives, reader_masks, metadata_files = read_inputs(args.limit_images)
    dataset = VinDrBinaryDataset(positives, transform=get_val_transform(SIZE),
                                image_source="cache", cache_dir=CACHE)
    # Load all three checkpoints successfully BEFORE writing any outputs.
    sources = {condition: load_checkpoint(condition) for condition in RUNS}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "heatmaps").mkdir()
    meta = {str(path.name): file_hash(path) for path in metadata_files}
    (args.out / "run_config.json").write_text(json.dumps({
        "seed": 42, "target_class": 1, "image_source": "cached_png",
        "n_images": len(dataset), "metadata_sha256": meta, "thread_count": args.threads,
        "metric_threshold": 0.5, "source_cache": str(CACHE),
        "annotation_aggregation": "reader-union then reader-mean then image-mean",
    }, indent=2), encoding="utf-8")
    image_rows, reader_rows, provenance = [], [], []
    for condition, (model, ckpt) in sources.items():
        provenance.append({"condition": condition, "seed": 42,
                           "checkpoint": str(ckpt), "sha256": file_hash(ckpt)})
        dest = args.out / "heatmaps" / condition
        dest.mkdir()
        gradcam = GradCAM(model, model.layer4[-1])
        try:
            for i in range(len(dataset)):
                tensor, true_label, image_id = dataset[i]
                if int(true_label) != 1:
                    raise AssertionError(f"Expected abnormal image: {image_id}")
                heatmap, target, p_abnormal = gradcam.generate(tensor, class_idx=1)
                if target != 1 or heatmap.shape != (SIZE, SIZE) or not np.isfinite(heatmap).all():
                    raise ValueError(f"Invalid Grad-CAM output: {condition} {image_id}")
                np.save(dest / f"{image_id}.npy", heatmap.astype(np.float32), allow_pickle=False)
                degenerate = float(np.std(heatmap)) < 1e-12 or float(np.sum(heatmap)) < 1e-12
                vals = []
                for radiologist, mask in reader_masks[image_id].items():
                    metrics = score(heatmap, mask) if not degenerate else {
                        k: (float(mask.mean()) if k == "box_area_fraction" else np.nan) for k in METRICS
                    }
                    vals.append(metrics)
                    reader_rows.append({"condition": condition, "seed": 42, "image_id": image_id,
                                        "radiologist": radiologist, "degenerate": degenerate, **metrics})
                average = {metric: (float(np.mean([row[metric] for row in vals]))
                                    if not degenerate or metric == "box_area_fraction" else np.nan)
                           for metric in METRICS}
                image_rows.append({"condition": condition, "seed": 42, "image_id": image_id,
                                   "target_class": 1, "predicted_class": int(p_abnormal > 0.5),
                                   "abnormal_probability": p_abnormal, "degenerate": degenerate,
                                   **average})
                if (i + 1) % 10 == 0 or i + 1 == len(dataset):
                    print(f"{condition}: {i + 1}/{len(dataset)}", flush=True)
        finally:
            gradcam.remove_hooks()
    maps = pd.DataFrame(image_rows)
    readers = pd.DataFrame(reader_rows)
    if len(maps) != 3 * len(dataset) or len(readers) != 9 * len(dataset):
        raise AssertionError("Incorrect output row counts")
    if maps.duplicated(["condition", "seed", "image_id"]).any():
        raise AssertionError("Duplicate map identifiers")
    maps.to_csv(args.out / "image_metrics.csv", index=False)
    readers.to_csv(args.out / "reader_metrics.csv", index=False)
    pd.DataFrame(provenance).to_csv(args.out / "checkpoint_provenance.csv", index=False)
    summary = maps.groupby("condition").agg(
        images=("image_id", "size"),
        nondegenerate=("degenerate", lambda v: int((~v).sum())),
        abnormal_predictions=("predicted_class", "sum"),
        **{key: (key, "mean") for key in METRICS},
    )
    summary.to_csv(args.out / "summary_seed42.csv")
    print(f"COMPLETE: {len(maps)} maps and {len(readers)} radiologist comparisons")
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()