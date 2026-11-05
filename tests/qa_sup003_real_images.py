"""Gate C on RunPod: quantify SUP003 gamma on real cached training images.

python tests/qa_sup003_real_images.py \
 --train data/interim/exp01_centralized_vindr_binary_seed42_train_split.csv \
 --cache data/processed/vindr_binary_224_png --out outputs/sup003_feature_QA
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms as T

from src.data.sup003_heterogeneity import CLIENT_GAMMAS, GammaTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing real-image QA: {args.out}")
    if args.n < 10 or args.n > 200:
        raise ValueError("Use --n between 10 and 200")
    df = pd.read_csv(args.train).sort_values("image_id")
    # Fixed, class-stratified sampling; these never become training examples on disk.
    sampled = []
    for label in (0, 1):
        subset = df[df["binary_label"] == label]
        sampled.append(subset.sample(n=args.n // 2, random_state=42))
    images = pd.concat(sampled, ignore_index=True)
    resize = T.Resize((224, 224))
    records = []
    thumbnails = []
    for _, row in images.iterrows():
        image_path = args.cache / f'{row["image_id"]}.png'
        with Image.open(image_path) as im:
            image = resize(im.convert("RGB"))
        original = np.asarray(image).astype(np.float32)
        comparisons = [image]
        for cid, gamma in enumerate(CLIENT_GAMMAS):
            modified_image = GammaTransform(gamma)(image)
            modified = np.asarray(modified_image).astype(np.float32)
            diff = np.abs(modified - original)
            records.append({
                "image_id": str(row["image_id"]),
                "label": int(row["binary_label"]),
                "client_id": cid,
                "gamma": gamma,
                "mean_abs_pixel_delta_0to255": float(diff.mean()),
                "original_mean": float(original.mean()),
                "modified_mean": float(modified.mean()),
                "original_black_fraction": float((original <= 1).mean()),
                "modified_black_fraction": float((modified <= 1).mean()),
                "original_white_fraction": float((original >= 254).mean()),
                "modified_white_fraction": float((modified >= 254).mean()),
            })
            comparisons.append(modified_image)
        if len(thumbnails) < 6:
            thumbnails.append(comparisons)
    args.out.mkdir(parents=True)
    pd.DataFrame(records).to_csv(args.out / "gamma_pixel_QA.csv", index=False)
    # Small reference montage; inspect visually before full training.
    if thumbnails:
        w, h = 6 * 224, len(thumbnails) * 248
        canvas = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for ridx, comps in enumerate(thumbnails):
            for cidx, image in enumerate(comps):
                canvas.paste(image, (cidx * 224, ridx * 248 + 24))
                draw.text((cidx * 224 + 4, ridx * 248 + 5),
                          "Original" if cidx == 0 else f"C{cidx - 1} gamma={CLIENT_GAMMAS[cidx - 1]}",
                          fill=(0, 0, 0))
        canvas.save(args.out / "gamma_visual_QA.png")
    summary = {"status": "VISUAL_REVIEW_REQUIRED", "n_images": len(images),
               "gamma_values": list(CLIENT_GAMMAS),
               "note": "Compare intensity/clipping CSV and inspect montage before training."
               " No samples or labels were rewritten."}
    (args.out / "QA_readme.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
