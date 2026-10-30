"""Author-applied, SUP-005-only diagnostic experiment entrypoint.

PROPOSAL ONLY: manually create src/experiments/run_sup005_multilabel.py.
Run via: python -m src.experiments.run_sup005_multilabel --config configs/sup005_multilabel.yaml --seed 42 --smoke

Existing exp01/02/03 source and model checkpoints are not modified.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.split_clients import create_iid_client_splits, create_label_skew_client_splits
from src.data.datasets import validate_image_cache
from src.data.sup005_multilabel import (
    CLASS_NAMES, LABEL_COLUMNS, build_labels, evaluate_multilabel, make_loader,
    train_epoch_multilabel, train_pos_weight,
)
from src.models.cnn import create_model
from src.utils.seed import set_seed


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def checked_config(path, seed, smoke):
    with open(path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError("Config must be a YAML mapping")
    needed = ["project_root", "train_csv", "canonical_train_split_path", "canonical_val_split_path",
              "train_image_dir", "image_source", "image_size", "image_cache_extension",
              "batch_size", "learning_rate", "condition", "output_dir", "num_classes"]
    if any(k not in c for k in needed):
        raise ValueError("Missing diagnostic config key(s): %s" % [k for k in needed if k not in c])
    if str(c.get("task")) != "vindr_multilabel_14" or int(c["num_classes"]) != 14:
        raise ValueError("Only task=vindr_multilabel_14 / num_classes=14 is supported")
    if str(c.get("model_name", "resnet18")) != "resnet18" or bool(c.get("pretrained", False)):
        raise ValueError("Require randomly initialized ResNet-18 to match original initialization")
    if c["condition"] not in {"centralized", "iid", "non_iid_label_skew"}:
        raise ValueError("Invalid condition")
    if c["image_source"] not in {"dicom", "cache"}:
        raise ValueError("image_source must be cache or dicom")
    if c["image_source"] == "cache" and not c.get("image_cache_dir"):
        raise ValueError("image_cache_dir required for cache")
    if int(c["batch_size"]) <= 0 or int(c["image_size"]) != 224 or float(c["learning_rate"]) <= 0:
        raise ValueError("Invalid image_size/batch_size/learning_rate")
    if int(c.get("epochs", 10)) <= 0:
        raise ValueError("Invalid central epoch count")
    if int(c.get("fl_num_clients", 5)) != 5 or int(c.get("fl_num_rounds", 50)) < 1:
        raise ValueError("Only five clients and positive round count supported")
    if int(c.get("fl_local_epochs", 1)) != 1:
        raise ValueError("FL local epoch count must equal one")
    if str(c.get("class_weight_scope", "global")) != "global":
        raise ValueError("SUP-005 uses shared weights from full training data")
    if float(c.get("threshold", 0.5)) != 0.5:
        raise ValueError("SUP-005 fixed threshold must be 0.5")
    if int(seed) not in [42, 7, 123]:
        raise ValueError("Supported planned replicate seeds: 42, 7, 123")
    if c["condition"] == "non_iid_label_skew" and float(c.get("non_iid_skew_strength", 4)) != 4.0:
        raise ValueError("Original binary skew strength must remain 4.0")
    c["seed"] = int(seed)
    c["smoke"] = bool(smoke)
    c["num_classes"] = 14
    c["experiment_id"] = "sup005_%s_seed%d%s" % (c["condition"], seed, "_smoke" if smoke else "")
    # All outputs are nested in a wholly separate diagnostic namespace; refuse the binary root.
    root = Path(c["output_dir"]).resolve()
    expected_namespace = root / "sup005_multilabel_14"
    c["run_dir"] = str(expected_namespace / c["condition"] / ("seed_%d" % seed) / ("smoke" if smoke else "full"))
    if not Path(c["run_dir"]).resolve().is_relative_to(expected_namespace.resolve()):
        raise ValueError("Invalid diagnostic output path")
    for k in ["train_csv", "canonical_train_split_path", "canonical_val_split_path"]:
        if not Path(c[k]).is_file():
            raise FileNotFoundError("Required input missing: %s" % c[k])
    return c


def ensure_empty_run_dir(c):
    # Never overwrite an existing run, even a previously interrupted one.
    p = Path(c["run_dir"])
    if p.exists():
        raise FileExistsError("Refusing to overwrite SUP-005 run; archive or rename manually: %s" % p)
    p.mkdir(parents=True, exist_ok=False)
    for child in ("checkpoints", "metrics", "interim"):
        (p / child).mkdir()
    return p


def choose_smoke_subset(train, val, seed):
    """Only smoke mode truncates splits; full runs retain all 12k/3k images."""
    a = train.groupby("binary_label", group_keys=False).sample(n=64, random_state=seed)
    b = val.groupby("binary_label", group_keys=False).sample(n=32, random_state=seed)
    return a.reset_index(drop=True), b.reset_index(drop=True)


def verify_predictions(preds, expected):
    if len(preds) != len(expected) or set(preds.image_id) != set(expected.image_id):
        raise RuntimeError("Saved predictions do not cover the expected validation IDs")


def save_evaluation(metrics, per_class, predictions, folder, stem, expected, cfg):
    verify_predictions(predictions, expected)
    predictions = predictions.copy()
    predictions.insert(0, "condition", cfg["condition"])
    predictions.insert(1, "seed", cfg["seed"])
    predictions.insert(2, "checkpoint", stem)
    predictions.to_csv(folder / (stem + "_predictions.csv"), index=False)
    per_class.insert(0, "condition", cfg["condition"])
    per_class.insert(1, "seed", cfg["seed"])
    per_class.to_csv(folder / (stem + "_per_class.csv"), index=False)
    with open(folder / (stem + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump({k: (None if isinstance(v, (int, float, np.number)) and not np.isfinite(v) else v)
                   for k, v in metrics.items()}, f, indent=2, allow_nan=False)


def evaluate_after_training(model, val, cfg, global_pos_weights, device, folder, stem):
    vloader = make_loader(val, cfg, training=False)
    criterion = nn.BCEWithLogitsLoss(pos_weight=global_pos_weights.to(device))
    metrics, per_class, predictions = evaluate_multilabel(model, vloader, criterion, device)
    save_evaluation(metrics, per_class, predictions, folder, stem, val, cfg)
    return metrics


def central_training(c, train, val, global_pos_weights, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model("resnet18", num_classes=14, pretrained=False).to(device)
    loader = make_loader(train, c, training=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=global_pos_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=float(c["learning_rate"]),
                            weight_decay=float(c.get("weight_decay", 0.01)))
    epochs = 1 if c["smoke"] else int(c.get("epochs", 10))
    rows = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch_multilabel(model, loader, criterion, optimizer, device)
        val_metrics = evaluate_after_training(model, val, c, global_pos_weights, device,
                                              run_dir / "metrics", "epoch_%02d" % epoch)
        rows.append({"epoch": epoch, "train_loss": train_loss,
                     "val_loss": val_metrics["loss"], "macro_ap": val_metrics["macro_ap"],
                     "micro_ap": val_metrics["micro_ap"], "macro_f1": val_metrics["macro_f1"],
                     "micro_f1": val_metrics["micro_f1"], "runtime_seconds": time.perf_counter() - start})
        print("SUP005 centralized epoch %d/%d; train_loss=%.5f macro_AP=%.5f" %
              (epoch, epochs, train_loss, val_metrics["macro_ap"]))
    # Fixed terminal checkpoint: do not select from held-out validation performance.
    torch.save({"model_state_dict": model.state_dict(), "config": c, "epoch": epochs},
               run_dir / "checkpoints" / ("centralized_epoch_%02d.pt" % epochs))
    pd.DataFrame(rows).to_csv(run_dir / "metrics" / "epoch_history.csv", index=False)


def save_clients(clients, run_dir, split_type):
    rows = []
    for cid, frame in enumerate(clients):
        frame[["image_id", "binary_label"] + LABEL_COLUMNS].to_csv(
            run_dir / "interim" / ("client_%d_labels.csv" % cid), index=False)
        r = dict(client_id=cid, split_type=split_type, num_images=len(frame),
                 normal=int((frame.binary_label == 0).sum()),
                 abnormal=int((frame.binary_label == 1).sum()))
        r.update({"positive_%02d" % j: int(frame[k].sum()) for j, k in enumerate(LABEL_COLUMNS)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(run_dir / "interim" / "client_class_support.csv", index=False)


def fl_training(c, train, val, global_pos_weights, run_dir):
    # Import Flower lazily so source validation and centralized experiments work
    # without Flower installed; RunPod must install the compatible project version.
    import flwr as fl
    from src.training.train_flower_fedavg import (
        SaveModelFedAvg, get_model_parameters, history_to_dataframe,
        set_model_parameters, weighted_average,
    )
    split = c["condition"]
    if split == "iid":
        clients = create_iid_client_splits(train, num_clients=5, seed=c["seed"])
    elif split == "non_iid_label_skew":
        clients = create_label_skew_client_splits(
            train, num_clients=5, seed=c["seed"],
            skew_strength=float(c.get("non_iid_skew_strength", 4.0)))
    else:
        raise ValueError("FL condition required")
    save_clients(clients, run_dir, split)
    initial_model = create_model("resnet18", 14, pretrained=False)
    rounds = 1 if c["smoke"] else int(c.get("fl_num_rounds", 50))
    c["fl_num_rounds"] = rounds
    # Reuse the original strategy's exact checkpoint serialization/aggregation.
    class MultiLabelClient(fl.client.NumPyClient):
        def __init__(self, cid):
            self.cid = cid
            self.device = torch.device(c.get("fl_client_device", "cpu"))
            self.model = create_model("resnet18", 14, pretrained=False)
            self.loader = make_loader(clients[cid], c, training=True)
        def get_parameters(self, config):
            return get_model_parameters(self.model)
        def fit(self, parameters, config):
            set_model_parameters(self.model, parameters)
            self.model.to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=global_pos_weights.to(self.device))
            optimizer = optim.AdamW(self.model.parameters(), lr=float(c["learning_rate"]),
                                    weight_decay=float(c.get("weight_decay", 0.01)))
            t0 = time.perf_counter()
            loss = None
            for _ in range(int(config.get("local_epochs", 1))):
                loss = train_epoch_multilabel(self.model, self.loader, criterion, optimizer, self.device)
            return get_model_parameters(self.model), len(clients[self.cid]), {
                "train_loss": float(loss), "fit_runtime_seconds": time.perf_counter() - t0,
            }

    device = torch.device(c.get("fl_client_device", "cpu"))
    eval_model = create_model("resnet18", 14, pretrained=False).to(device)
    eval_loader = make_loader(val, c, training=False)
    eval_criterion = nn.BCEWithLogitsLoss(pos_weight=global_pos_weights.to(device))

    def eval_fn(server_round, parameters, evaluation_config):
        del evaluation_config
        if server_round == 0:
            return None
        set_model_parameters(eval_model, parameters)
        metrics, _, _ = evaluate_multilabel(eval_model, eval_loader, eval_criterion, device)
        public_metrics = {k: float(v) for k, v in metrics.items() if k not in {"loss"} and np.isfinite(v)}
        public_metrics["num_validation_samples"] = len(val)
        return float(metrics["loss"]), public_metrics

    strategy = SaveModelFedAvg(
        global_model=initial_model, checkpoint_dir=run_dir / "checkpoints", config=c,
        fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=5,
        min_available_clients=5,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(initial_model)),
        evaluate_fn=eval_fn,
        on_fit_config_fn=lambda rnd: {"server_round": rnd, "local_epochs": 1},
        fit_metrics_aggregation_fn=weighted_average,
    )
    if bool(c.get("save_initial_checkpoint", True)):
        torch.save({"model_state_dict": initial_model.state_dict(), "config": c, "round": 0},
                   run_dir / "checkpoints" / "flower_fedavg_round_0.pt")
    def client_fn(cid):
        return MultiLabelClient(int(cid)).to_client()
    hist = fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=5, config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": float(c.get("fl_client_num_cpus", 2)),
                          "num_gpus": float(c.get("fl_client_num_gpus", 1.0))},
    )
    history_to_dataframe(hist).to_csv(run_dir / "metrics" / "round_history.csv", index=False)
    terminal = run_dir / "checkpoints" / ("flower_fedavg_round_%d.pt" % rounds)
    if not terminal.is_file():
        raise FileNotFoundError("Final FedAvg checkpoint not saved: %s" % terminal)
    checkpoint = torch.load(terminal, map_location="cpu", weights_only=False)
    eval_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    evaluate_after_training(eval_model, val, c, global_pos_weights, device,
                            run_dir / "metrics", "round_%02d" % rounds)


def main():
    p = argparse.ArgumentParser(description="SUP-005 separate 14-finding multilabel validation")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True, choices=[42, 7, 123])
    p.add_argument("--smoke", action="store_true", help="Tiny training dry-run in its own output directory")
    args = p.parse_args()
    c = checked_config(Path(args.config), args.seed, args.smoke)
    # Validate source and labels before creating any run artifacts.
    train, val = build_labels(Path(c["train_csv"]), Path(c["canonical_train_split_path"]),
                              Path(c["canonical_val_split_path"]))
    all_train = train
    if args.smoke:
        train, val = choose_smoke_subset(train, val, args.seed)
    global_pos_weights = train_pos_weight(all_train)  # always full 12,000 TRAIN only
    if c["image_source"] == "cache":
        validate_image_cache(pd.concat([train, val], ignore_index=True),
                             Path(c["image_cache_dir"]), c.get("image_cache_extension", ".png"))
    set_seed(args.seed)
    run_dir = ensure_empty_run_dir(c)
    for key in ["train_csv", "canonical_train_split_path", "canonical_val_split_path"]:
        c[key + "_sha256"] = file_sha256(c[key])
    (run_dir / "metadata.json").write_text(json.dumps(
        {"condition": c["condition"], "seed": args.seed, "smoke": args.smoke,
         "target_rule": "per-image OR over three readers after deduplication of boxes",
         "finding_names": CLASS_NAMES, "class_id_order": list(range(14)),
         "no_finding_rule": "all-zero 14-target vector", "threshold": 0.5,
         "loss": "BCEWithLogitsLoss", "weight_rule": "global training (N - positives)/positives, uncapped",
         "positive_weights": global_pos_weights.tolist(),
         "split_type": c["condition"],
         "non_iid_semantics": "binary normal/abnormal label skew, not per-finding skew",
         "config": c}, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    train[["image_id", "binary_label"] + LABEL_COLUMNS].to_csv(run_dir / "interim" / "train_labels.csv", index=False)
    val[["image_id", "binary_label"] + LABEL_COLUMNS].to_csv(run_dir / "interim" / "val_labels.csv", index=False)
    print("SUP-005", c["condition"], "seed", args.seed, "train", len(train), "val", len(val))
    if c["condition"] == "centralized":
        central_training(c, train, val, global_pos_weights, run_dir)
    else:
        fl_training(c, train, val, global_pos_weights, run_dir)
    produced = list((run_dir / "checkpoints").glob("*.pt"))
    if not produced:
        raise RuntimeError("No diagnostic checkpoint was saved")
    (run_dir / "checkpoint_sha256.json").write_text(json.dumps(
        {p.name: file_sha256(p) for p in sorted(produced)}, indent=2), encoding="utf-8")
    print("SUP-005 completed:", run_dir)


if __name__ == "__main__":
    main()
