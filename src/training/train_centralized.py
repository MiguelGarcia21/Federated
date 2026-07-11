import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datasets import (
    check_first_batch,
    get_train_transform,
    get_val_transform,
    make_dataloader,
)
from src.models.cnn import create_model, count_trainable_parameters
from src.training.evaluate import evaluate_model, train_one_epoch


def compute_class_weights(
    train_df: pd.DataFrame,
    device,
) -> torch.Tensor:
    """Compute inverse-frequency weights for the two binary classes."""
    class_counts = (
        train_df["binary_label"]
        .value_counts()
        .sort_index()
        .reindex([0, 1], fill_value=1)
    )

    class_weights = len(train_df) / (2.0 * class_counts)

    class_weights_tensor = torch.tensor(
        class_weights.values,
        dtype=torch.float32,
    ).to(device)

    print("Class counts:")
    print(class_counts)
    print("Class weights:", class_weights_tensor)

    return class_weights_tensor


def train_centralized_debug(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_dir: Path,
    metrics_dir: Path,
    device,
) -> Tuple[Path, Path]:
    """Train the centralized Phase 5/Phase 6-compatible baseline.

    The function name is preserved for compatibility with the confirmed
    Phase 5 runner. The implementation now records richer thesis-oriented
    metrics without changing the underlying training logic.
    """
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))

    train_transform = get_train_transform(
        image_size=image_size,
        horizontal_flip=bool(
            config.get("horizontal_flip", True)
        ),
        rotation_degrees=float(
            config.get("rotation_degrees", 0.0)
        ),
    )

    val_transform = get_val_transform(
        image_size=image_size
    )

    train_loader = make_dataloader(
        df=train_df,
        transform=train_transform,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = make_dataloader(
        df=val_df,
        transform=val_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    check_first_batch(train_loader)

    model = create_model(
        model_name=config.get("model_name", "resnet18"),
        num_classes=int(config["num_classes"]),
        pretrained=bool(config.get("pretrained", False)),
    ).to(device)

    trainable_parameters = count_trainable_parameters(model)

    print(model.__class__.__name__)
    print("Trainable parameters:", trainable_parameters)

    class_weights = compute_class_weights(
        train_df=train_df,
        device=device,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    learning_rate = float(
        config.get(
            "learning_rate",
            config.get("centralized_lr", 1e-4),
        )
    )

    epochs = int(
        config.get(
            "epochs",
            config.get("centralized_epochs", 3),
        )
    )

    weight_decay = float(config.get("weight_decay", 0.01))

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    experiment_id = config["experiment_id"]
    run_type = str(
        config.get("run_type", "centralized_debug")
    )
    model_name = config.get("model_name", "resnet18")
    dataset_name = config.get("dataset", "vindr")
    task_name = config.get("task", "binary_abnormal")
    checkpoint_metric = str(
        config.get("checkpoint_metric", "f1")
    )

    if run_type == "centralized_debug":
        checkpoint_filename = "centralized_resnet18_debug_best.pt"
        output_name = "centralized_debug"
    else:
        checkpoint_filename = f"{run_type}_{model_name}_best.pt"
        output_name = run_type

    best_checkpoint_path = (
        checkpoint_dir / checkpoint_filename
    )

    best_predictions_path = (
        metrics_dir
        / f"{experiment_id}_{output_name}_best_predictions.csv"
    )

    centralized_metrics_path = (
        metrics_dir
        / f"{experiment_id}_{output_name}_metrics.csv"
    )

    centralized_rows = []

    best_f1 = -1.0
    best_epoch = None

    run_start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics, predictions_df = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            return_predictions=True,
        )

        epoch_runtime_seconds = (
            time.perf_counter() - epoch_start_time
        )

        cumulative_runtime_seconds = (
            time.perf_counter() - run_start_time
        )

        current_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        is_best_checkpoint = val_metrics["f1"] > best_f1

        if is_best_checkpoint:
            best_f1 = float(val_metrics["f1"])
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "learning_rate": current_learning_rate,
                    "checkpoint_metric": checkpoint_metric,
                    "trainable_parameters": trainable_parameters,
                    "epoch_runtime_seconds": epoch_runtime_seconds,
                    "cumulative_runtime_seconds": cumulative_runtime_seconds,
                },
                best_checkpoint_path,
            )

            predictions_df = predictions_df.copy()
            predictions_df.insert(
                0,
                "experiment_id",
                experiment_id,
            )
            predictions_df.insert(
                1,
                "run_type",
                run_type,
            )
            predictions_df.insert(
                2,
                "epoch",
                epoch,
            )
            predictions_df["seed"] = config["seed"]
            predictions_df["model_name"] = model_name
            predictions_df["dataset"] = dataset_name
            predictions_df["task"] = task_name
            predictions_df["checkpoint_path"] = str(
                best_checkpoint_path
            )

            predictions_df.to_csv(
                best_predictions_path,
                index=False,
            )

        row = {
            "experiment_id": experiment_id,
            "run_type": run_type,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "val_average_precision": val_metrics[
                "average_precision"
            ],
            "val_balanced_accuracy": val_metrics[
                "balanced_accuracy"
            ],
            "val_negative_predictive_value": val_metrics[
                "negative_predictive_value"
            ],
            "val_false_positive_rate": val_metrics[
                "false_positive_rate"
            ],
            "val_false_negative_rate": val_metrics[
                "false_negative_rate"
            ],
            "val_mcc": val_metrics["mcc"],
            "val_brier_score": val_metrics["brier_score"],
            "val_mean_positive_probability": val_metrics[
                "mean_positive_probability"
            ],
            "val_mean_prediction_confidence": val_metrics[
                "mean_prediction_confidence"
            ],
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_num_samples": val_metrics["num_samples"],
            "val_num_actual_positive": val_metrics[
                "num_actual_positive"
            ],
            "val_num_actual_negative": val_metrics[
                "num_actual_negative"
            ],
            "val_actual_positive_rate": val_metrics[
                "actual_positive_rate"
            ],
            "val_num_predicted_positive": val_metrics[
                "num_predicted_positive"
            ],
            "val_num_predicted_negative": val_metrics[
                "num_predicted_negative"
            ],
            "val_positive_prediction_rate": val_metrics[
                "positive_prediction_rate"
            ],
            "val_negative_prediction_rate": val_metrics[
                "negative_prediction_rate"
            ],
            "learning_rate": current_learning_rate,
            "weight_decay": weight_decay,
            "checkpoint_metric": checkpoint_metric,
            "horizontal_flip": bool(
                config.get("horizontal_flip", True)
            ),
            "rotation_degrees": float(
                config.get("rotation_degrees", 0.0)
            ),
            "epoch_runtime_seconds": epoch_runtime_seconds,
            "cumulative_runtime_seconds": (
                cumulative_runtime_seconds
            ),
            "is_best_checkpoint": is_best_checkpoint,
            "best_f1_so_far": best_f1,
            "best_epoch_so_far": best_epoch,
            "seed": config["seed"],
            "model_name": model_name,
            "dataset": dataset_name,
            "task": task_name,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "trainable_parameters": trainable_parameters,
            "checkpoint_path": str(best_checkpoint_path),
            "best_predictions_path": str(
                best_predictions_path
            ),
        }

        centralized_rows.append(row)

        centralized_metrics_df = pd.DataFrame(
            centralized_rows
        )
        centralized_metrics_df.to_csv(
            centralized_metrics_path,
            index=False,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_bal_acc={row['val_balanced_accuracy']:.4f} | "
            f"val_f1={row['val_f1']:.4f} | "
            f"val_auc={row['val_auc']:.4f} | "
            f"val_ap={row['val_average_precision']:.4f} | "
            f"pred_pos_rate="
            f"{row['val_positive_prediction_rate']:.4f} | "
            f"runtime={epoch_runtime_seconds:.2f}s | "
            f"best={is_best_checkpoint}"
        )

    total_runtime_seconds = time.perf_counter() - run_start_time

    print("Saved centralized metrics to:", centralized_metrics_path)
    print("Saved best centralized checkpoint to:", best_checkpoint_path)
    print("Saved best-epoch predictions to:", best_predictions_path)
    print("Best epoch:", best_epoch)
    print("Best validation F1:", best_f1)
    print(
        "Total centralized runtime:",
        f"{total_runtime_seconds:.2f} seconds",
    )

    print(pd.DataFrame(centralized_rows))

    return best_checkpoint_path, centralized_metrics_path