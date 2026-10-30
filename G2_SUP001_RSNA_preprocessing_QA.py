#!/usr/bin/env python3
"""G2 SUP-001: read-only RSNA preprocessing compatibility QA.

Runs the *existing repository's* VinDrBinaryDataset (DICOM path) and
get_val_transform() on a fixed, stratified sample from the frozen RSNA
Stage-2 manifest. Independently constructs the in-memory cached-PNG route
implemented by cache_vindr.py, then compares resulting model-input tensors.

Does NOT change the repository, original images, labels, manifest, or checkpoints.
Only writes new audit outputs under DATA_DIR/sup001_preprocess_qa.

Requirements: the repository's existing virtual environment, including PyTorch,
torchvision, pandas, numpy, Pillow, and pydicom with an available JPEG decoder.
Python >=3.10 (repository datasets.py uses str | Path).

Typical local command from C:\\Users\\migar\\Desktop\\Stuff\\thesis\\test:
    python .\\G2_SUP001_RSNA_preprocessing_QA.py --data-dir . --repo-root ..\\Federated
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SHA256 = "b1268958d49b9d4d407bb1511f6fc1412c3c41931837cf1a1007d4725c3a2bf6"
EXPECTED_DATASETS_SHA256 = "4a6b7c376058bcf2288e489be4b2f028f245f1f4cb10833abd097ea49b56ff6b"
EXPECTED_CLASSES = ("Normal", "Lung Opacity", "No Lung Opacity / Not Normal")
EXPECTED_COUNTS = {"Normal": 8851, "Lung Opacity": 6012, "No Lung Opacity / Not Normal": 11821}
IMAGE_SIZE = 224


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_manifest(data_dir: Path):
    cohort_dir = data_dir / "sup001_cohort_audit"
    manifest = cohort_dir / "rsna_stage2_external_binary_manifest.csv"
    audit = cohort_dir / "rsna_stage2_external_cohort_audit.json"
    if not manifest.is_file() or not audit.is_file():
        raise FileNotFoundError("Frozen manifest or cohort audit JSON is missing")
    actual_hash = sha256(manifest)
    provenance = json.loads(audit.read_text(encoding="utf-8"))
    if actual_hash != EXPECTED_SHA256 or actual_hash != provenance.get("sha256", {}).get(manifest.name):
        raise ValueError("Frozen manifest hash does not match the approved SUP-001 cohort")

    by_class = defaultdict(list)
    seen = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        mandatory = {"rsna_exam_id", "dicom_filename", "rsna_detailed_class", "binary_abnormal_label"}
        if not reader.fieldnames or not mandatory.issubset(reader.fieldnames):
            raise ValueError("Unexpected cohort manifest columns")
        for row in reader:
            exam_id = row["rsna_exam_id"].strip()
            filename = row["dicom_filename"].strip()
            category = row["rsna_detailed_class"].strip()
            label = row["binary_abnormal_label"].strip()
            if (not exam_id or exam_id in seen or filename != exam_id + ".dcm"
                    or "/" in filename or "\\" in filename):
                raise ValueError("Duplicate, invalid, or unsafe manifest image ID/filename")
            if category not in EXPECTED_COUNTS or label != str(int(category != "Normal")):
                raise ValueError("Unexpected RSNA label/category in manifest")
            seen.add(exam_id)
            by_class[category].append((exam_id, filename, int(label)))
    counts = {c: len(by_class[c]) for c in EXPECTED_CLASSES}
    if counts != EXPECTED_COUNTS or len(seen) != 26684:
        raise ValueError("Frozen cohort class counts do not match audited values")
    return manifest, actual_hash, by_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True,
                        help="Repository root containing src/data/datasets.py")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Defaults to DATA_DIR/stage_2_train_images")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to DATA_DIR/sup001_preprocess_qa")
    parser.add_argument("--sample-per-class", type=int, default=25)
    args = parser.parse_args()
    if args.sample_per_class < 1:
        raise ValueError("--sample-per-class must be positive")

    root = args.data_dir.expanduser().resolve()
    repo = args.repo_root.expanduser().resolve()
    images_dir = (args.images_dir or (root / "stage_2_train_images")).expanduser().resolve()
    out_dir = (args.out_dir or (root / "sup001_preprocess_qa")).expanduser().resolve()
    module_path = repo / "src" / "data" / "datasets.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find repository module: {module_path}")
    if sha256(module_path) != EXPECTED_DATASETS_SHA256:
        raise ValueError("Local datasets.py differs from reviewed uploaded version; request review before proceeding")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Cannot find RSNA images directory: {images_dir}")
    manifest_path, manifest_hash, by_class = load_frozen_manifest(root)

    # Use the actual imported repository code; do not duplicate the DICOM pipeline.
    sys.path.insert(0, str(repo))
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    from PIL import Image, ImageDraw
    from src.data.datasets import VinDrBinaryDataset, get_val_transform, read_dicom_as_uint8

    transform = get_val_transform(IMAGE_SIZE)
    rng = random.Random(20261029)
    sample = []
    for category in EXPECTED_CLASSES:
        sample.extend((category, *item) for item in
                      rng.sample(by_class[category], min(args.sample_per_class, len(by_class[category]))))

    rows = []
    previews = []
    failures = []
    diffs = []
    direct_finite = 0
    for index, (category, exam_id, filename, label) in enumerate(sample, 1):
        dicom_path = images_dir / filename
        try:
            if not dicom_path.is_file():
                raise FileNotFoundError(str(dicom_path))
            df = pd.DataFrame([{"image_id": exam_id, "binary_label": label, "dicom_path": str(dicom_path)}])
            dataset = VinDrBinaryDataset(df, transform=transform, image_source="dicom")
            tensor_direct, returned_label, returned_id = dataset[0]
            if returned_id != exam_id or int(returned_label) != label:
                raise ValueError("Repository Dataset returned the wrong image identifier or label")
            if tensor_direct.shape != (3, IMAGE_SIZE, IMAGE_SIZE) or tensor_direct.dtype != torch.float32:
                raise ValueError(f"Unexpected direct-DICOM model-input tensor: {tensor_direct.shape}, {tensor_direct.dtype}")
            if not torch.isfinite(tensor_direct).all().item():
                raise ValueError("Non-finite model input tensor")
            direct_finite += 1

            # Emulate cache_vindr.py's image steps and the existing cached-image loader,
            # using an *in-memory* PNG rather than altering the actual image/cache files.
            uint8_image = read_dicom_as_uint8(dicom_path)
            if uint8_image.ndim != 2 or uint8_image.dtype != np.uint8:
                raise ValueError("Expected a grayscale uint8 output from read_dicom_as_uint8")
            image_gray = Image.fromarray(uint8_image).convert("L")
            image_gray = image_gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            buffer = io.BytesIO()
            image_gray.save(buffer, format="PNG", optimize=False, compress_level=1)
            buffer.seek(0)
            with Image.open(buffer) as png_image:
                image_rgb = png_image.convert("RGB")
            tensor_cached = transform(image_rgb)
            if tensor_cached.shape != tensor_direct.shape or not torch.isfinite(tensor_cached).all().item():
                raise ValueError("Invalid cached-route model-input tensor")

            abs_diff = (tensor_direct - tensor_cached).abs()
            max_diff = float(abs_diff.max().item())
            mean_diff = float(abs_diff.mean().item())
            exact_equal = bool(torch.equal(tensor_direct, tensor_cached))
            diffs.append(max_diff)
            row = {
                "image_id": exam_id,
                "rsna_class": category,
                "label": label,
                "raw_min": int(uint8_image.min()),
                "raw_max": int(uint8_image.max()),
                "raw_unique": int(len(np.unique(uint8_image))),
                "direct_tensor_min": float(tensor_direct.min().item()),
                "direct_tensor_max": float(tensor_direct.max().item()),
                "max_absolute_tensor_difference": max_diff,
                "mean_absolute_tensor_difference": mean_diff,
                "pixelwise_exact_match": exact_equal,
                "status": "ok",
                "error": "",
            }
            rows.append(row)
            if sum(c == category for c, *_ in previews) < 4:
                previews.append((category, exam_id, image_gray.copy()))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"image_id": exam_id, "rsna_class": category, "error": message})
            rows.append({"image_id": exam_id, "rsna_class": category, "label": label,
                         "raw_min": "", "raw_max": "", "raw_unique": "", "direct_tensor_min": "",
                         "direct_tensor_max": "", "max_absolute_tensor_difference": "",
                         "mean_absolute_tensor_difference": "", "pixelwise_exact_match": "",
                         "status": "failed", "error": message})
            print(f"FAIL {index}/{len(sample)} ({category}): {exam_id} {message[:200]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "rsna_preprocessing_qa_sample.csv"
    fields = ("image_id", "rsna_class", "label", "raw_min", "raw_max", "raw_unique",
              "direct_tensor_min", "direct_tensor_max", "max_absolute_tensor_difference",
              "mean_absolute_tensor_difference", "pixelwise_exact_match", "status", "error")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # A reviewable contact sheet is created only from already decoded sample images.
    sheet_path = out_dir / "rsna_preprocessing_contact_sheet.png"
    if previews:
        width, height = 245, 278
        cols = 4
        sheet = Image.new("RGB", (cols * width, 3 * height), "#202020")
        draw = ImageDraw.Draw(sheet)
        for i, (category, exam_id, image) in enumerate(previews[:12]):
            x = (i % cols) * width
            y = (i // cols) * height
            sheet.paste(image.convert("RGB"), (x + 10, y + 6))
            draw.text((x + 10, y + 234), category[:30], fill="white")
            draw.text((x + 10, y + 249), exam_id[:32], fill="white")
        sheet.save(sheet_path)

    all_ok = len(rows) == len(sample) and not failures
    exact = all_ok and all(r["pixelwise_exact_match"] for r in rows)
    # Differences require review, not automatic acceptance, even if tiny.
    status = ("PREPROCESSING_SAMPLE_QA_PASS_EXACT_CACHE_MATCH" if exact else
              "PREPROCESSING_SAMPLE_QA_REVIEW_REQUIRED" if all_ok else
              "PREPROCESSING_SAMPLE_QA_FAIL")
    result = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Deterministic 25-per-class by default; actual repository direct-DICOM Dataset versus in-memory cached-PNG path",
        "frozen_manifest_sha256": manifest_hash,
        "repo_root": str(repo),
        "repository_datasets_sha256": sha256(module_path),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sample_requested": len(sample),
        "sample_processed": direct_finite,
        "failed": len(failures),
        "all_tensor_shapes": "3x224x224 float32" if all_ok else "see failures",
        "direct_vs_cache_exact_match_count": sum(1 for row in rows if row["pixelwise_exact_match"] is True),
        "direct_vs_cache_max_absolute_difference": max(diffs) if diffs else None,
        "direct_vs_cache_mean_of_max_absolute_differences": sum(diffs)/len(diffs) if diffs else None,
        "zero_dynamic_range_count": sum(1 for row in rows if isinstance(row["raw_unique"], int) and row["raw_unique"] == 1),
        "failures_first_15": failures[:15],
        "outputs": {"sample_csv": str(csv_path), "contact_sheet": str(sheet_path) if previews else None},
        "caveats": [
            "This checks only the sampled images, not every pixel of the full cohort.",
            "It compares two currently implemented inference routes on RSNA data; it cannot reconstruct historical centralized/federated training inputs.",
            "Visual inspection of the contact sheet remains necessary.",
            "The approved RSNA binary label mapping is operational rather than identical to VinDr-CXR labels.",
            "No checkpoint loading or external model performance is assessed here.",
        ],
    }
    json_path = out_dir / "rsna_preprocessing_qa.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=== G2 / SUP-001 RSNA REPOSITORY PREPROCESSING QA ===")
    print("Status:", status)
    print("Frozen manifest SHA-256:", manifest_hash)
    print("Repository datasets.py SHA-256:", result["repository_datasets_sha256"])
    print(f"Processed: {direct_finite}/{len(sample)}; failed: {len(failures)}")
    print(f"Direct vs cache exactly matching tensors: {result['direct_vs_cache_exact_match_count']}/{len(sample)}")
    print("Largest max absolute difference:", result["direct_vs_cache_max_absolute_difference"])
    print("Constant-intensity decoded samples:", result["zero_dynamic_range_count"])
    print("Report:", json_path)
    print("Sample CSV:", csv_path)
    if previews:
        print("Visual contact sheet:", sheet_path)
    print("NOTE: Visual review and frozen-checkpoint loading QA are still pending.")
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
