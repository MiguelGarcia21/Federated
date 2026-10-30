#!/usr/bin/env python3
"""G2/SUP-001: Read-only nine-model RSNA external-inference smoke test.

This is a QA gate, NOT a performance experiment. It selects a fixed 5-image
sample per RSNA category by default and evaluates all nine accepted frozen
checkpoints on EXACTLY those same 15 radiographs, with the repository's own
DICOM loader and deterministic validation transform.

Source DICOMs, checkpoint files, configurations and repository code are never
modified. Writes only JSON and CSV reports under --out-dir.

Example from PowerShell in C:\\Users\\migar\\Desktop\\Stuff\\thesis\\Federated:
 python .\\G2_SUP001_RSNA_inference_smoke_QA.py --data-dir ..\\test --checkpoints-root ..\\complete_checkpoints\\checkpoints --repo-root .

The script requires the trusted prior structure-QA report and hash inventory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

APPROVED_MANIFEST_SHA256 = "b1268958d49b9d4d407bb1511f6fc1412c3c41931837cf1a1007d4725c3a2bf6"
REVIEWED_DATASETS_SHA256 = "4a6b7c376058bcf2288e489be4b2f028f245f1f4cb10833abd097ea49b56ff6b"
RSNA_CATEGORIES = ("Normal", "Lung Opacity", "No Lung Opacity / Not Normal")
EXPECTED_COUNTS = {"Normal": 8851, "Lung Opacity": 6012, "No Lung Opacity / Not Normal": 11821}
EXPECTED_CKPTS = {
    ("Centralized", 42): ("exp01_centralized_vindr_binary_seed42", "centralized_resnet18_debug_best.pt", "epoch", 2),
    ("Centralized", 7): ("exp01_centralized_vindr_binary_seed7", "centralized_resnet18_best.pt", "epoch", 10),
    ("Centralized", 123): ("exp01_centralized_vindr_binary_seed123", "centralized_resnet18_best.pt", "epoch", 9),
    ("IID", 42): ("exp02_fedavg_iid_5clients_seed42", "flower_fedavg_round_50.pt", "round", 50),
    ("IID", 7): ("exp02_fedavg_iid_5clients_seed7", "flower_fedavg_round_50.pt", "round", 50),
    ("IID", 123): ("exp02_fedavg_iid_5clients_seed123", "flower_fedavg_round_50.pt", "round", 50),
    ("Non-IID", 42): ("exp03_fedavg_noniid_5clients_seed42", "flower_fedavg_round_50.pt", "round", 50),
    ("Non-IID", 7): ("exp03_fedavg_noniid_5clients_seed7", "flower_fedavg_round_50.pt", "round", 50),
    ("Non-IID", 123): ("exp03_fedavg_noniid_5clients_seed123", "flower_fedavg_round_50.pt", "round", 50),
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().lower()


def verify_manifest(data_dir: Path, per_class: int):
    manifest = data_dir / "sup001_cohort_audit" / "rsna_stage2_external_binary_manifest.csv"
    cohort_qa = manifest.with_name("rsna_stage2_external_cohort_audit.json")
    if not manifest.is_file() or not cohort_qa.is_file():
        raise FileNotFoundError("Frozen RSNA cohort manifest or audit JSON missing under data-dir/sup001_cohort_audit")
    actual_sha = digest_file(manifest)
    previous = json.loads(cohort_qa.read_text(encoding="utf-8"))
    if actual_sha != APPROVED_MANIFEST_SHA256 or previous.get("sha256", {}).get(manifest.name) != actual_sha:
        raise ValueError("Frozen RSNA manifest SHA-256 mismatch")

    groups = defaultdict(list)
    ids_seen = set()
    with manifest.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"rsna_exam_id", "dicom_filename", "rsna_detailed_class", "binary_abnormal_label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Unexpected frozen manifest column names")
        for row in reader:
            image_id = row["rsna_exam_id"].strip()
            filename = row["dicom_filename"].strip()
            category = row["rsna_detailed_class"].strip()
            label = row["binary_abnormal_label"].strip()
            if not image_id or image_id in ids_seen or category not in RSNA_CATEGORIES:
                raise ValueError("Duplicate/blank image ID or unexpected detailed RSNA category")
            if filename != image_id + ".dcm" or "/" in filename or "\\" in filename:
                raise ValueError("Unsafe or inconsistent DICOM filename in manifest")
            if label != str(int(category != "Normal")):
                raise ValueError("RSNA binary label does not match frozen category mapping")
            ids_seen.add(image_id)
            groups[category].append((image_id, filename, int(label)))

    counts = {key: len(groups[key]) for key in RSNA_CATEGORIES}
    if counts != EXPECTED_COUNTS or len(ids_seen) != 26684:
        raise ValueError(f"Frozen cohort counts not as approved: {counts}")
    if per_class < 1 or per_class > min(counts.values()):
        raise ValueError("per-class must be at least 1 and no greater than smallest RSNA category")
    rng = random.Random(20261029)
    selected = []
    for cat in RSNA_CATEGORIES:
        selected.extend((cat, *record) for record in rng.sample(groups[cat], per_class))
    return selected, actual_sha


def verify_models(checkpoints_root: Path):
    qa_path = checkpoints_root / "sup001_checkpoint_structure_qa.json"
    provenance_path = checkpoints_root / "sup001_checkpoint_provenance.csv"
    if not qa_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Missing previous checkpoint structure QA or provenance CSV")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if (qa.get("status") != "CHECKPOINT_STRUCTURE_QA_PASS"
            or qa.get("passed") != 9 or qa.get("failed") != 0):
        raise ValueError("Previous nine-checkpoint structure QA did not pass")
    previous = {(r["condition"], int(r["seed"])): r for r in qa.get("results", [])}
    with provenance_path.open("r", encoding="utf-8-sig", newline="") as source:
        provenance = {(r["Condition"], int(r["Seed"])): r for r in csv.DictReader(source)}
    if set(previous) != set(EXPECTED_CKPTS) or set(provenance) != set(EXPECTED_CKPTS):
        raise ValueError("Checkpoint QA or provenance CSV does not include the nine expected identities")
    for identity, (folder, filename, key, desired) in EXPECTED_CKPTS.items():
        record = previous[identity]
        row = provenance[identity]
        if (record["status"] != "PASS" or not record["strict_model_load"] or
                not record["synthetic_forward_pass"] or
                not record["file_hash_verified"] or not record["config_hash_verified"]):
            raise ValueError(f"Checkpoint not previously approved: {identity}")
        if (record["folder"] != folder or record["checkpoint"] != filename or
                row["Folder"] != folder or row["Checkpoint"] != filename):
            raise ValueError(f"Checkpoint identity mismatch: {identity}")
        if record.get("metadata", {}).get(key) != desired:
            raise ValueError(f"Checkpoint {identity} metadata {key} not {desired}")
        ckpt = checkpoints_root / folder / filename
        if not ckpt.is_file() or int(row["SizeBytes"]) != ckpt.stat().st_size:
            raise ValueError(f"Missing/altered checkpoint file: {ckpt}")
        # Verify current data actually match the previous provenance, not just an old report.
        if digest_file(ckpt) != row["CheckpointSHA256"].strip().lower():
            raise ValueError(f"Checkpoint SHA-256 changed: {ckpt}")
    return provenance, qa_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoints-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--per-class", type=int, default=5, help="Deterministic images per RSNA category (default: 5)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    checkpoints_root = args.checkpoints_root.resolve()
    repo_root = args.repo_root.resolve()
    out_dir = (args.out_dir or (data_dir / "sup001_inference_smoke_qa")).resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    sample, manifest_hash = verify_manifest(data_dir, args.per_class)
    provenance, checkpoint_qa = verify_models(checkpoints_root)
    datasets_path = repo_root / "src/data/datasets.py"
    model_path = repo_root / "src/models/cnn.py"
    if not model_path.is_file() or not datasets_path.is_file():
        raise FileNotFoundError("Repository src/data/datasets.py or src/models/cnn.py not found")
    if digest_file(datasets_path) != REVIEWED_DATASETS_SHA256:
        raise ValueError("datasets.py changed since approved preprocessing audit; review before inference")

    images_dir = data_dir / "stage_2_train_images"
    for category, image_id, filename, label in sample:
        if not (images_dir / filename).is_file():
            raise FileNotFoundError(f"Missing selected RSNA DICOM: {images_dir / filename}")

    import pandas as pd
    import torch
    sys.path.insert(0, str(repo_root))
    from src.data.datasets import VinDrBinaryDataset, get_val_transform
    from src.models.cnn import create_model
    from torch.utils.data import DataLoader

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    frames = [dict(image_id=image_id, binary_label=label, dicom_path=str(images_dir / filename))
              for category, image_id, filename, label in sample]
    dataset = VinDrBinaryDataset(pd.DataFrame(frames), transform=get_val_transform(224), image_source="dicom")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Decode all selected images once and reuse exactly the same input tensors across nine models.
    batches = []
    decoded_ids = []
    for tensors, labels, image_ids in loader:
        if tensors.dtype != torch.float32 or tuple(tensors.shape[1:]) != (3, 224, 224):
            raise ValueError("Unexpected RSNA tensor dtype or shape")
        if not torch.isfinite(tensors).all().item():
            raise ValueError("Nonfinite RSNA model input")
        decoded_ids.extend(image_ids)
        batches.append((tensors, labels, list(image_ids)))
    if decoded_ids != [record[1] for record in sample]:
        raise ValueError("RSNA data loader returned missing, duplicate, or shuffled IDs")

    predictions = []
    summaries = []
    for identity, (folder, filename, meta_key, meta_val) in EXPECTED_CKPTS.items():
        condition, seed = identity
        ckpt_path = checkpoints_root / folder / filename
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Unrecognized saved checkpoint for {identity}")
        if int(ckpt[meta_key]) != meta_val:
            raise ValueError(f"Changed checkpoint metadata for {identity}")
        model = create_model(model_name="resnet18", num_classes=2, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval().to(device)
        n0, n1 = 0, 0
        per_model = []
        with torch.inference_mode():
            for tensors, labels, ids in batches:
                logits = model(tensors.to(device))
                if tuple(logits.shape) != (len(ids), 2) or not torch.isfinite(logits).all().item():
                    raise ValueError(f"Invalid logits from {identity}")
                probs = torch.softmax(logits.float(), dim=1).cpu()
                if not torch.isfinite(probs).all().item() or not torch.allclose(
                    probs.sum(dim=1), torch.ones(len(ids)), atol=1e-5, rtol=0
                ):
                    raise ValueError(f"Invalid probability values from {identity}")
                if not torch.all((probs >= 0) & (probs <= 1)).item():
                    raise ValueError(f"Out-of-range probability from {identity}")
                preds = probs.argmax(dim=1).tolist()
                for j, image_id in enumerate(ids):
                    assigned = int(preds[j]); n0 += assigned == 0; n1 += assigned == 1
                    per_model.append({
                        "condition": condition, "seed": seed, "image_id": image_id,
                        "rsna_detailed_class": next(row[0] for row in sample if row[1] == image_id),
                        "true_binary_label": int(labels[j].item()),
                        "predicted_binary_label": assigned,
                        "probability_abnormal": float(probs[j, 1]),
                        "confidence_predicted_class": float(probs[j, assigned]),
                        "checkpoint_sha256": provenance[identity]["CheckpointSHA256"].lower(),
                    })
        if len(per_model) != len(sample) or [r["image_id"] for r in per_model] != decoded_ids:
            raise ValueError(f"Incomplete/incorrect IDs for {identity}")
        predictions.extend(per_model)
        summaries.append({"condition": condition, "seed": seed, "images": len(per_model),
                          "predicted_normal": n0, "predicted_abnormal": n1,
                          "min_probability_abnormal": min(r["probability_abnormal"] for r in per_model),
                          "max_probability_abnormal": max(r["probability_abnormal"] for r in per_model),
                          "status": "PASS"})
        print(f"{condition:11s} seed {seed:3d} PASS  images={len(per_model)}  predicted normal={n0} abnormal={n1}", flush=True)
        del model, ckpt

    expected_rows = len(sample) * len(EXPECTED_CKPTS)
    if len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} prediction rows, found {len(predictions)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "sup001_rsna_smoke_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as dest:
        fields = ["condition", "seed", "image_id", "rsna_detailed_class", "true_binary_label",
                  "predicted_binary_label", "probability_abnormal", "confidence_predicted_class", "checkpoint_sha256"]
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader(); writer.writerows(predictions)
    report = {
        "status": "RSNA_NINE_MODEL_SMOKE_QA_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Technical inference smoke test only: do NOT interpret 15-image model metrics or deploy clinically",
        "sample_selection_seed": 20261029, "per_class": args.per_class,
        "sample_count": len(sample), "predictions_expected": expected_rows, "predictions_written": len(predictions),
        "rsna_frozen_manifest_sha256": manifest_hash,
        "datasets_module_sha256": digest_file(datasets_path), "model_module_sha256": digest_file(model_path),
        "checkpoint_qa_reference": str(checkpoint_qa),
        "inference_device": str(device), "torch_version": torch.__version__,
        "class_mapping": {"Normal": 0, "Lung Opacity": 1, "No Lung Opacity / Not Normal": 1},
        "summary": summaries,
        "predictions_csv": str(pred_path), "predictions_csv_sha256": digest_file(pred_path),
        "caveats": ["No external model accuracy claim can be drawn from this small smoke sample.",
                    "Historical validation-F1-best checkpoint selection still needs original run metrics to establish fully.",
                    "RSNA labels are an operational mapping, not identical to VinDr-CXR annotations.",
                    "RSNA image/exam IDs do not guarantee unique patients."]
    }
    json_path = out_dir / "sup001_rsna_smoke_qa.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nResult: {report['status']}  ({len(summaries)}/9 models; {len(predictions)} predictions)")
    print("Report:", json_path)
    print("Predictions:", pred_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE QA FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
