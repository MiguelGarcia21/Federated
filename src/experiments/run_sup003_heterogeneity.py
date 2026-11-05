"""Isolated SUP-003 runner. Do not change or invoke old experiment identifiers.

Usage:
 python -m src.experiments.run_sup003_heterogeneity --config configs/sup003_feature_only_seed42.yaml
 python -m src.experiments.run_sup003_heterogeneity --config configs/sup003_feature_only_seed42.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.preprocess_vindr import load_existing_full_split
from src.data.split_clients import create_iid_client_splits, save_client_splits
from src.data.sup003_heterogeneity import (
    CLIENT_GAMMAS, CONDITIONS, allocate_quantity_splits,
    client_manifest, sha256_file, verify_condition,
)
from src.experiments.run_exp02_fedavg import load_config, validate_config
from src.models.cnn import create_model
from src.training.evaluate import evaluate_model
from src.training.train_flower_fedavg import (
    compute_binary_class_weights, make_loader_from_df,
    resolve_client_device, run_fedavg_debug,
)
from src.utils.paths import create_experiment_dirs, save_config_snapshot
from src.utils.seed import set_seed


def assert_empty_outputs(config: dict) -> None:
    """Fail closed: old checkpoint, metrics, and split outputs may never be overwritten."""
    eid = str(config["experiment_id"])
    output_dir = Path(config["output_dir"])
    project_root = Path(config["project_root"])
    paths = (
        output_dir / "checkpoints" / eid,
        output_dir / "metrics" / f"{eid}_flower_fedavg_metrics.csv",
        project_root / "data" / "interim" / "client_splits" / eid,
    )
    collisions = [str(p) for p in paths if p.exists()]
    if collisions:
        raise FileExistsError(f"SUP003 refuses existing output artifacts: {collisions}")


def verify_predictions(pred: pd.DataFrame, val_df: pd.DataFrame) -> None:
    if len(val_df) != 3000 or len(pred) != 3000:
        raise AssertionError("Validation/prediction count must both equal 3000")
    if pred["image_id"].duplicated().any():
        raise AssertionError("Duplicate prediction IDs")
    if set(pred["image_id"]) != set(val_df["image_id"]):
        raise AssertionError("Prediction IDs do not cover canonical validation")
    reference_labels = val_df.set_index("image_id")["binary_label"]
    if not pred["true_label"].eq(pred["image_id"].map(reference_labels)).all():
        raise AssertionError("Prediction label mismatch")
    probs = pred["positive_probability"].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or not ((probs >= 0) & (probs <= 1)).all():
        raise AssertionError("Invalid validation probabilities")


def export_final_predictions(config: dict, val_df: pd.DataFrame, dirs: dict) -> dict:
    final_round = int(config["fl_num_rounds"])
    checkpoint_path = dirs["checkpoint_dir"] / f"flower_fedavg_round_{final_round}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing final checkpoint: {checkpoint_path}")
    # Own trusted checkpoints created by this repository; avoid loading external checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["round"]) != final_round:
        raise AssertionError("Checkpoint/round mismatch")
    if str(checkpoint["config"]["experiment_id"]) != config["experiment_id"]:
        raise AssertionError("Checkpoint/experiment ID mismatch")
    model = create_model(
        model_name=config.get("model_name", "resnet18"),
        num_classes=int(config["num_classes"]),
        pretrained=bool(config.get("pretrained", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = resolve_client_device(config)
    model.to(device)
    val_loader = make_loader_from_df(config=config, df=val_df, shuffle=False)
    canonical_train = pd.read_csv(config["canonical_train_split_path"])
    class_weights = compute_binary_class_weights(canonical_train, "SUP003 canonical train")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    metrics, pred = evaluate_model(
        model=model, loader=val_loader, criterion=criterion,
        device=device, return_predictions=True,
    )
    verify_predictions(pred, val_df)
    pred.insert(0, "condition", config["sup003_condition"])
    pred.insert(1, "seed", int(config["seed"]))
    pred.insert(2, "round", final_round)
    pred_path = dirs["checkpoint_dir"] / "sup003_round_final_predictions.csv"
    pred.to_csv(pred_path, index=False)
    metrics_path = dirs["checkpoint_dir"] / "sup003_round_final_metrics.json"
    # Some secondary measures (e.g., NPV when no negatives are predicted)
    # can legitimately be undefined at Round 1. Preserve that as JSON null.
    json_safe_metrics = {
        key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value)
              else value)
        for key, value in metrics.items()
    }
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(json_safe_metrics, output, indent=2, allow_nan=False)
    return metrics


def run(config_path: Path, smoke: bool) -> None:
    config = load_config(config_path)
    condition = str(config.get("sup003_condition", ""))
    seed = int(config["seed"])
    if condition not in CONDITIONS or seed not in (42, 7, 123):
        raise ValueError("SUP003 requires a recognized condition and seed in {42,7,123}")
    if config.get("split_type") != "iid":
        raise ValueError("Compatibility split_type must be 'iid' in new SUP003 YAML")
    if config.get("reuse_split_files") is not True:
        raise ValueError("SUP003 requires reuse_split_files: true")
    if config.get("class_weight_scope") != "global":
        raise ValueError("SUP003 requires class_weight_scope: global")
    if int(config["fl_num_clients"]) != 5 or int(config["fl_local_epochs"]) != 1:
        raise ValueError("SUP003 protocol requires 5 clients and 1 local epoch")
    if int(config["fl_num_rounds"]) != 50:
        raise ValueError("SUP003 full YAML must specify 50 rounds")
    if (int(config["batch_size"]) != 16
            or int(config["image_size"]) != 224
            or float(config["learning_rate"]) != 1e-4
            or float(config.get("weight_decay", 0.01)) != 0.01
            or bool(config.get("horizontal_flip", True)) is not True
            or float(config.get("rotation_degrees", 0.0)) != 0.0
            or config.get("image_source") != "cache"
            or bool(config.get("pretrained", False))):
        raise ValueError("SUP003 YAML deviates from the locked original training settings")
    gamma_expected = list(CLIENT_GAMMAS) if condition != "quantity_only" else None
    if config.get("sup003_gamma_by_client") != gamma_expected:
        raise ValueError("Gamma profile deviates from predeclared SUP003 protocol")

    # Reuse original validation and system configuration checks, never alter original code.
    validate_config(config)
    if smoke:
        config["experiment_id"] += "_smoke1"
        config["fl_num_rounds"] = 1
        config["save_checkpoints_every"] = 1

    if not config["experiment_id"].startswith("sup003_"):
        raise ValueError("SUP003 output experiment ID must start with 'sup003_'")
    assert_empty_outputs(config)

    # Reinterpret the new experiment metadata only after passing original IID config validation.
    config["split_type"] = f"sup003_{condition}"
    config["run_type"] = "sup003_controlled_heterogeneity"
    set_seed(seed)
    train_df, val_df, xai_ids = load_existing_full_split(config)
    if len(train_df) != 12000 or len(val_df) != 3000:
        raise AssertionError("Canonical 12k/3k membership changed")

    if condition == "feature_only":
        clients = create_iid_client_splits(train_df, 5, seed)
        verify_condition(condition, clients, train_df, val_df, iid_clients=clients)
    else:
        clients = allocate_quantity_splits(train_df, seed)
        verify_condition(condition, clients, train_df, val_df)

    # Only create output directories once all static checks have passed.
    dirs = create_experiment_dirs(config)
    save_config_snapshot(config, dirs)
    save_client_splits(
        clients, dirs["interim_dir"], config["experiment_id"],
        f"sup003_{condition}",
    )
    manifest = client_manifest(clients, condition)
    manifest.to_csv(dirs["checkpoint_dir"] / "sup003_client_manifest.csv", index=False)
    sha_inputs = [
        Path(config["canonical_train_split_path"]),
        Path(config["canonical_val_split_path"]),
        config_path,
        Path(__file__),
        Path(__file__).resolve().parents[1] / "data" / "sup003_heterogeneity.py",
        Path(__file__).resolve().parents[1] / "training" / "train_flower_fedavg.py",
    ]
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            encoding="utf-8", timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None
    provenance = {
        "condition": condition, "seed": seed, "smoke": smoke,
        "gamma_by_client": gamma_expected,
        "train_count": len(train_df), "val_count": len(val_df),
        "xai_id_count": len(xai_ids),
        "source_git_head": git_sha,
        "source_sha256": {str(p): sha256_file(p) for p in sha_inputs},
        "torch_version": torch.__version__,
    }
    with (dirs["checkpoint_dir"] / "sup003_provenance.json").open("w", encoding="utf-8") as out:
        json.dump(provenance, out, indent=2)

    _, metrics_path = run_fedavg_debug(
        config=config, client_dfs=clients, val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"], metrics_dir=dirs["metrics_dir"],
    )
    metrics_df = pd.read_csv(metrics_path)
    expected_rounds = set(range(1, int(config["fl_num_rounds"]) + 1))
    if set(metrics_df["round"].astype(int)) != expected_rounds or len(metrics_df) != len(expected_rounds):
        raise AssertionError("Missing/duplicate metrics rounds")
    # Validate actual Flower participant counts when the optional trainer hook is applied.
    participation_path = dirs["checkpoint_dir"] / "sup003_fit_participation.jsonl"
    if not participation_path.is_file():
        raise AssertionError("SUP003 participation log missing: apply both trainer patches")
    with participation_path.open("r", encoding="utf-8") as handle:
        fit_rows = [json.loads(line) for line in handle if line.strip()]
    if len(fit_rows) != len(expected_rounds):
        raise AssertionError("Participant log round count mismatch")
    if {int(row["round"]) for row in fit_rows} != expected_rounds:
        raise AssertionError("Participant log missing rounds")
    if any(len(row["samples"]) != 5 or sum(row["samples"]) != 12000 for row in fit_rows):
        raise AssertionError("FedAvg participation/samples mismatch")
    final_metrics = export_final_predictions(config, val_df, dirs)
    with (dirs["checkpoint_dir"] / "sup003_ACCEPTED.json").open("w", encoding="utf-8") as out:
        json.dump({"status": "PASS", "experiment_id": config["experiment_id"],
                   "final_round": int(config["fl_num_rounds"]),
                   "final_f1": final_metrics["f1"]}, out, indent=2)
    print("SUP003 PASS:", config["experiment_id"], "F1=", final_metrics["f1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="One round, separate _smoke1 output namespace")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
