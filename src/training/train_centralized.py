from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.datasets import (
    get_train_transform,
    get_val_transform,
    make_dataloader,
    check_first_batch,
)
from src.models.cnn import create_model, count_trainable_parameters
from src.training.evaluate import train_one_epoch, evaluate_model


def compute_class_weights(train_df: pd.DataFrame, device) -> torch.Tensor:
    class_counts = train_df["binary_label"].value_counts().sort_index()
    class_counts = class_counts.reindex([0, 1], fill_value=1)

    class_weights = len(train_df) / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)

    print("Class counts:")
    print(class_counts)
    print("Class weights:", class_weights)

    return class_weights


def train_centralized_debug(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    checkpoint_dir: Path,
    metrics_dir: Path,
    device,
) -> Tuple[Path, Path]:
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))

    train_transform = get_train_transform(image_size)
    val_transform = get_val_transform(image_size)

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
    ).to(device)

    print(model.__class__.__name__)
    print("Trainable parameters:", count_trainable_parameters(model))

    class_weights = compute_class_weights(train_df=train_df, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    learning_rate = float(config.get("learning_rate", config.get("centralized_lr", 1e-4)))
    epochs = int(config.get("epochs", config.get("centralized_epochs", 3)))

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    centralized_rows = []

    best_f1 = -1.0
    best_checkpoint_path = checkpoint_dir / "centralized_resnet18_debug_best.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "experiment_id": config["experiment_id"],
            "run_type": "centralized_debug",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "seed": config["seed"],
            "model_name": config.get("model_name", "resnet18"),
            "dataset": config.get("dataset", "vindr"),
            "task": config.get("task", "binary_abnormal"),
            "checkpoint_path": str(best_checkpoint_path),
        }

        centralized_rows.append(row)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_f1={row['val_f1']:.4f} | "
            f"val_auc={row['val_auc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )

    centralized_metrics_df = pd.DataFrame(centralized_rows)

    centralized_metrics_path = (
        metrics_dir / f"{config['experiment_id']}_centralized_debug_metrics.csv"
    )
    
    centralized_metrics_df.to_csv(centralized_metrics_path, index=False)

    print("Saved centralized metrics to:", centralized_metrics_path)
    print("Saved best centralized checkpoint to:", best_checkpoint_path)
    print(centralized_metrics_df)

    return best_checkpoint_path, centralized_metrics_path