"""Isolated SUP007 runner. Six full runs plus separately namespaced smoke runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.preprocess_vindr import load_existing_full_split
from src.data.split_clients import create_iid_client_splits, save_client_splits
from src.experiments.run_exp02_fedavg import load_config, validate_config
from src.models.cnn import create_model
from src.training.evaluate import evaluate_model
from src.training.train_flower_fedavg import (
    SaveModelFedAvgM, compute_binary_class_weights, make_loader_from_df,
    resolve_client_device, run_fedavg_debug,
)
from src.utils.paths import create_experiment_dirs, save_config_snapshot
from src.utils.seed import set_seed

CONDITIONS = {"fedavgm_full", "fedavg_partial"}
SEEDS = {42, 7, 123}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_empty_outputs(config: dict) -> None:
    eid = config["experiment_id"]
    root = Path(config["project_root"])
    out = Path(config["output_dir"])
    paths = [
        out / "checkpoints" / eid,
        out / "metrics" / f"{eid}_flower_fedavg_metrics.csv",
        root / "data" / "interim" / "client_splits" / eid,
    ]
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"SUP007 refuses existing outputs: {existing}")


def verify_protocol(config: dict, smoke: bool) -> None:
    cond = config.get("sup007_condition")
    seed = int(config["seed"])
    if cond not in CONDITIONS or seed not in SEEDS:
        raise ValueError("SUP007 requires approved condition and seed")
    if config["experiment_id"] != f"sup007_{cond}_seed{seed}":
        raise ValueError("SUP007 experiment_id/condition/seed mismatch")
    if config.get("split_type") != "iid":
        raise ValueError("SUP007 requires IID splits")
    if config.get("reuse_split_files") is not True:
        raise ValueError("SUP007 requires canonical split reuse")
    if config.get("class_weight_scope") != "local":
        raise ValueError("SUP007 matches historical original IID local weighting")
    algorithm = "fedavgm" if cond == "fedavgm_full" else "fedavg"
    if config.get("fl_algorithm") != algorithm:
        raise ValueError("SUP007 algorithm/condition mismatch")
    expected_active = 5 if cond == "fedavgm_full" else 3
    if int(config.get("sup007_active_clients", -1)) != expected_active:
        raise ValueError("SUP007 participation mismatch")
    if (int(config["fl_num_clients"]) != 5
            or int(config["fl_num_rounds"]) != 50
            or int(config["fl_local_epochs"]) != 1
            or int(config["batch_size"]) != 16
            or int(config["image_size"]) != 224
            or float(config["learning_rate"]) != 1e-4
            or float(config.get("weight_decay", 0.01)) != 0.01
            or config.get("image_source") != "cache"
            or config.get("pretrained", False) is not False
            or config.get("horizontal_flip") is not True
            or float(config.get("rotation_degrees", 0)) != 0):
        raise ValueError("SUP007 locked training settings changed")
    if cond == "fedavgm_full":
        if (float(config.get("sup007_server_learning_rate", -1)) != 1.0
                or float(config.get("sup007_server_momentum", -1)) != 0.9):
            raise ValueError("SUP007 FedAvgM settings changed")
        # Verify cooperative Python method resolution before a GPU run.
        assert SaveModelFedAvgM.__mro__[2] is fl.server.strategy.FedAvgM
    elif ("sup007_server_momentum" in config
          or "sup007_server_learning_rate" in config):
        raise ValueError("Partial FedAvg must not declare FedAvgM settings")
    if smoke:
        config["experiment_id"] += "_smoke2" if cond == "fedavgm_full" else "_smoke1"
        config["fl_num_rounds"] = 2 if cond == "fedavgm_full" else 1
        config["save_checkpoints_every"] = 1


def compare_baseline_membership(config: dict, clients: list[pd.DataFrame]) -> str:
    """Check historical client membership if its saved manifests are present."""
    seed = int(config["seed"])
    folder = (Path(config["project_root"]) / "data" / "interim"
              / "client_splits" / f"exp02_fedavg_iid_5clients_seed{seed}")
    paths = [folder / f"client_{i}_iid.csv" for i in range(5)]
    if not all(p.is_file() for p in paths):
        return "NOT_VERIFIED_original_client_manifests_missing"
    for i, p in enumerate(paths):
        old = pd.read_csv(p)
        if set(old["image_id"].astype(str)) != set(clients[i]["image_id"].astype(str)):
            raise AssertionError(f"SUP007 client {i} differs from original seed {seed}")
    return "MATCH_original_IID_client_memberships"


def export_final(config: dict, val_df: pd.DataFrame, dirs: dict) -> dict:
    round_n = int(config["fl_num_rounds"])
    checkpoint_path = dirs["checkpoint_dir"] / f"flower_fedavg_round_{round_n}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (int(saved["round"]) != round_n
            or saved["config"]["experiment_id"] != config["experiment_id"]):
        raise AssertionError("SUP007 final checkpoint provenance mismatch")
    model = create_model(
        model_name=config.get("model_name", "resnet18"),
        num_classes=int(config["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    device = resolve_client_device(config)
    model.to(device)
    loader = make_loader_from_df(config=config, df=val_df, shuffle=False)
    canonical_train = pd.read_csv(config["canonical_train_split_path"])
    weights = compute_binary_class_weights(canonical_train, "SUP007 train")
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    metrics, pred = evaluate_model(
        model=model, loader=loader, criterion=criterion,
        device=device, return_predictions=True,
    )
    if (len(pred) != 3000 or pred["image_id"].duplicated().any()
            or set(pred["image_id"]) != set(val_df["image_id"])):
        raise AssertionError("SUP007 final predictions do not cover 3000 images")
    refs = val_df.set_index("image_id")["binary_label"]
    if not pred["true_label"].eq(pred["image_id"].map(refs)).all():
        raise AssertionError("SUP007 prediction labels mismatch")
    prob = pred["positive_probability"].to_numpy(dtype=float)
    if not np.isfinite(prob).all() or not ((0 <= prob) & (prob <= 1)).all():
        raise AssertionError("SUP007 invalid probabilities")
    pred.insert(0, "condition", config["sup007_condition"])
    pred.insert(1, "seed", int(config["seed"]))
    pred.insert(2, "round", round_n)
    pred.to_csv(dirs["checkpoint_dir"] / "sup007_round_final_predictions.csv", index=False)
    safe_metrics = {
        k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v)
            else float(v) if isinstance(v, (float, np.floating, np.integer))
            else v)
        for k, v in metrics.items()
    }
    with (dirs["checkpoint_dir"] / "sup007_round_final_metrics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(safe_metrics, f, indent=2, allow_nan=False)
    return metrics


def run(config_path: Path, smoke: bool) -> None:
    config = load_config(config_path)
    verify_protocol(config, smoke)
    validate_config(config)
    assert_empty_outputs(config)
    set_seed(int(config["seed"]))
    train_df, val_df, _ = load_existing_full_split(config)
    if len(train_df) != 12000 or len(val_df) != 3000:
        raise AssertionError("SUP007 canonical 12k/3k changed")
    clients = create_iid_client_splits(train_df, 5, int(config["seed"]))
    if any(len(c) != 2400 for c in clients):
        raise AssertionError("SUP007 expected 2400 samples/client")
    baseline_membership = compare_baseline_membership(config, clients)
    dirs = create_experiment_dirs(config)
    save_config_snapshot(config, dirs)
    save_client_splits(
        clients, dirs["interim_dir"], config["experiment_id"], "iid"
    )
    source_files = [
        config_path,
        Path(config["canonical_train_split_path"]),
        Path(config["canonical_val_split_path"]),
        Path(__file__),
        Path(__file__).resolve().parents[1] / "training" / "train_flower_fedavg.py",
        Path(__file__).resolve().parents[1] / "data" / "split_clients.py",
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, encoding="utf-8", timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = None
    provenance = {
        "experiment_id": config["experiment_id"],
        "condition": config["sup007_condition"],
        "smoke": smoke,
        "git_head": git_head,
        "flower_version": fl.__version__,
        "torch_version": torch.__version__,
        "original_client_membership": baseline_membership,
        "source_sha256": {str(p): sha256(p) for p in source_files},
    }
    with (dirs["checkpoint_dir"] / "sup007_provenance.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(provenance, f, indent=2)

    _, metric_path = run_fedavg_debug(
        config=config, client_dfs=clients, val_df=val_df,
        checkpoint_dir=dirs["checkpoint_dir"], metrics_dir=dirs["metrics_dir"],
    )
    rows = pd.read_csv(metric_path)
    rounds = int(config["fl_num_rounds"])
    if len(rows) != rounds or set(rows["round"].astype(int)) != set(range(1, rounds+1)):
        raise AssertionError("SUP007 missing/duplicate rounds")
    if not (rows["class_weight_scope"] == "local").all():
        raise AssertionError("SUP007 class weighting metadata mismatch")
    if not (rows["fl_algorithm"] == config["fl_algorithm"]).all():
        raise AssertionError("SUP007 aggregation metadata mismatch")

    log_path = dirs["checkpoint_dir"] / "sup007_fit_participation.jsonl"
    if not log_path.is_file():
        raise AssertionError("SUP007 participation log missing")
    fit_rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    if len(fit_rows) != rounds or sorted(x["round"] for x in fit_rows) != list(range(1,rounds+1)):
        raise AssertionError("SUP007 participation log round coverage mismatch")
    n = int(config["sup007_active_clients"])
    usage = {str(i): 0 for i in range(5)}
    for x in fit_rows:
        if (len(x["actual_cids"]) != n
                or len(set(x["actual_cids"])) != n
                or x["actual_cids"] != x["planned_cids"]
                or x["sample_total"] != 2400*n
                or x["num_failures"] != 0):
            raise AssertionError(f"SUP007 participation invalid round {x['round']}")
        for cid in x["actual_cids"]:
            usage[cid] += 1
    if config["sup007_condition"] == "fedavg_partial" and rounds == 50:
        if set(usage.values()) != {30}:
            raise AssertionError(f"SUP007 partial schedule imbalance {usage}")
    elif config["sup007_condition"] == "fedavgm_full":
        if set(usage.values()) != {rounds}:
            raise AssertionError(f"SUP007 full participation imbalance {usage}")
    final_metrics = export_final(config, val_df, dirs)
    for key in ["f1", "accuracy", "balanced_accuracy", "auc"]:
        if abs(float(final_metrics[key]) - float(rows.iloc[-1][key])) > 1e-6:
            raise AssertionError(f"SUP007 final {key} differs from metric history")
    with (dirs["checkpoint_dir"] / "sup007_ACCEPTED.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump({"status": "PASS", "condition": config["sup007_condition"],
                   "seed": int(config["seed"]), "rounds": rounds,
                   "final_f1": float(final_metrics["f1"]), "client_rounds": usage}, f, indent=2)
    print("SUP007 PASS:", config["experiment_id"], "F1:", final_metrics["f1"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
