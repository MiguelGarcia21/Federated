from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.datasets import (
    VinDrBinaryDataset,
    denormalize_tensor,
    get_val_transform,
)
from src.models.cnn import create_model


class GradCAM:
    """Grad-CAM implementation for ResNet-style convolutional models."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Remove registered PyTorch hooks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ):
        """Generate a normalized Grad-CAM heatmap for one image."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.unsqueeze(0).to(
            next(self.model.parameters()).device
        )

        logits = self.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                ).item()
            )

        if class_idx < 0 or class_idx >= logits.shape[1]:
            raise ValueError(
                f"class_idx={class_idx} is outside the model output range."
            )

        confidence = float(
            probabilities[:, class_idx].item()
        )

        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam.astype(np.float32)

        if not np.isfinite(cam).all():
            raise ValueError(
                "Generated Grad-CAM heatmap contains non-finite values."
            )

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        cam = cam - cam_min

        if cam_max > cam_min:
            cam = cam / (cam_max - cam_min)
        else:
            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        return cam, class_idx, confidence


def save_gradcam_overlay(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """Save a Grad-CAM overlay as a PNG image."""
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = denormalize_tensor(image_tensor)

    figure = plt.figure(figsize=(5, 5))

    plt.imshow(image)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=0.35,
        vmin=0.0,
        vmax=1.0,
    )

    if title is not None:
        plt.title(
            title,
            fontsize=8,
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_raw_heatmap(
    heatmap: np.ndarray,
    save_path: Path,
) -> None:
    """Save a normalized Grad-CAM heatmap as a NumPy array."""
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if heatmap.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional Grad-CAM heatmap."
        )

    if not np.isfinite(heatmap).all():
        raise ValueError(
            "Cannot save a heatmap containing non-finite values."
        )

    np.save(
        save_path,
        heatmap.astype(np.float32),
        allow_pickle=False,
    )


def _load_gradcam_model(
    config: Dict[str, Any],
    checkpoint_path: Path,
    device,
) -> nn.Module:
    """Create the configured model and load a trusted project checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Grad-CAM checkpoint not found: {checkpoint_path}"
        )

    model = create_model(
        model_name=config.get(
            "model_name",
            "resnet18",
        ),
        num_classes=int(config["num_classes"]),
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint must contain a dictionary: {checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain model_state_dict: "
            f"{checkpoint_path}"
        )
    checkpoint_config = checkpoint.get("config", {})

    checkpoint_model_name = checkpoint_config.get("model_name")
    current_model_name = config.get("model_name", "resnet18")

    if (
        checkpoint_model_name is not None
        and checkpoint_model_name != current_model_name
    ):
        raise ValueError(
            "Model mismatch between the checkpoint and XAI config: "
            f"checkpoint={checkpoint_model_name!r}, "
            f"config={current_model_name!r}"
        )

    checkpoint_num_classes = checkpoint_config.get("num_classes")

    if (
        checkpoint_num_classes is not None
        and int(checkpoint_num_classes) != int(config["num_classes"])
    ):
        raise ValueError(
            "num_classes mismatch between the checkpoint and XAI config: "
            f"checkpoint={checkpoint_num_classes}, "
            f"config={config['num_classes']}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model


def generate_gradcam_outputs(
    config: Dict[str, Any],
    checkpoint_path: Path,
    val_df: pd.DataFrame,
    xai_ids: Iterable[str],
    heatmap_dir: Path,
    metrics_dir: Path,
    run_type: str,
    device,
    round_num: Optional[int] = None,
) -> Path:
    """Generate Grad-CAM overlays, raw heatmaps, and metadata."""
    model = _load_gradcam_model(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    target_layer = model.layer4[-1]
    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    requested_ids = [
        str(image_id)
        for image_id in xai_ids
    ]

    if not requested_ids:
        gradcam.remove_hooks()
        raise ValueError(
            "No XAI image IDs were provided."
        )

    xai_df = (
        val_df[
            val_df["image_id"]
            .astype(str)
            .isin(requested_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    found_ids = set(
        xai_df["image_id"].astype(str)
    )

    missing_ids = [
        image_id
        for image_id in requested_ids
        if image_id not in found_ids
    ]

    if missing_ids:
        gradcam.remove_hooks()
        raise ValueError(
            f"{len(missing_ids)} requested XAI image IDs were not "
            f"found in the validation split. Examples: "
            f"{missing_ids[:5]}"
        )

    xai_dataset = VinDrBinaryDataset(
        xai_df,
        transform=get_val_transform(
            int(config["image_size"])
        ),
    )

    if round_num is None:
        output_subdir = (
            Path(heatmap_dir)
            / run_type
        )
    else:
        output_subdir = (
            Path(heatmap_dir)
            / run_type
            / f"round_{round_num}"
        )

    overlay_dir = output_subdir / "overlays"
    raw_dir = output_subdir / "raw"

    overlay_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gradcam_rows = []

    try:
        for idx in range(len(xai_dataset)):
            image_tensor, label, image_id = xai_dataset[idx]

            heatmap, predicted_class, confidence = (
                gradcam.generate(image_tensor)
            )

            file_stem = (
                f"{image_id}"
                f"_label{label.item()}"
                f"_pred{predicted_class}"
            )

            overlay_path = (
                overlay_dir
                / f"{file_stem}.png"
            )

            raw_heatmap_path = (
                raw_dir
                / f"{file_stem}.npy"
            )

            if round_num is None:
                title = (
                    f"{image_id} | "
                    f"label={label.item()} | "
                    f"pred={predicted_class} | "
                    f"conf={confidence:.3f}"
                )
            else:
                title = (
                    f"FL round {round_num} | "
                    f"{image_id} | "
                    f"label={label.item()} | "
                    f"pred={predicted_class} | "
                    f"conf={confidence:.3f}"
                )

            save_gradcam_overlay(
                image_tensor=image_tensor,
                heatmap=heatmap,
                save_path=overlay_path,
                title=title,
            )

            save_raw_heatmap(
                heatmap=heatmap,
                save_path=raw_heatmap_path,
            )

            gradcam_rows.append({
                "experiment_id": config["experiment_id"],
                "source_experiment_id": config.get(
                    "source_experiment_id",
                    config["experiment_id"],
                ),
                "dataset": config.get(
                    "dataset",
                    "vindr",
                ),
                "task": config.get(
                    "task",
                    "binary_abnormal",
                ),
                "training_type": run_type,
                "split_type": config.get(
                    "split_type",
                    "centralized",
                ),
                "round": round_num,
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "image_id": str(image_id),
                "true_label": int(
                    label.item()
                ),
                "predicted_label": int(
                    predicted_class
                ),
                "correct": int(
                    predicted_class
                    == int(label.item())
                ),
                "confidence": float(
                    confidence
                ),
                "xai_method": "gradcam",
                "target_class": int(
                    predicted_class
                ),
                "target_layer": "layer4[-1]",
                "heatmap_min": float(
                    heatmap.min()
                ),
                "heatmap_max": float(
                    heatmap.max()
                ),
                "heatmap_mean": float(
                    heatmap.mean()
                ),
                "heatmap_std": float(
                    heatmap.std()
                ),
                "raw_heatmap_path": str(
                    raw_heatmap_path
                ),
                "heatmap_path": str(
                    overlay_path
                ),
                "overlay_path": str(
                    overlay_path
                ),
            })
    finally:
        gradcam.remove_hooks()

    gradcam_df = pd.DataFrame(
        gradcam_rows
    )

    if round_num is None:
        output_filename = (
            f"{config['experiment_id']}_"
            f"{run_type}_gradcam_outputs.csv"
        )
    else:
        output_filename = (
            f"{config['experiment_id']}_"
            f"{run_type}_round_{round_num}_"
            f"gradcam_outputs.csv"
        )

    output_path = (
        Path(metrics_dir)
        / output_filename
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    gradcam_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {run_type} Grad-CAM metadata to:",
        output_path,
    )
    print(
        f"Saved {run_type} Grad-CAM overlays to:",
        overlay_dir,
    )
    print(
        f"Saved {run_type} raw heatmaps to:",
        raw_dir,
    )
    print(gradcam_df.head())

    return output_path