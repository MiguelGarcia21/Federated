from pathlib import Path
from typing import Dict, Any, Iterable, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.datasets import VinDrBinaryDataset, get_val_transform, denormalize_tensor
from src.models.cnn import create_model


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
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

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, image_tensor: torch.Tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        image_tensor = image_tensor.unsqueeze(0).to(
            next(self.model.parameters()).device
        )

        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx


def save_gradcam_overlay(image_tensor, heatmap, save_path: Path, title=None):
    image = denormalize_tensor(image_tensor)

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.35)

    if title is not None:
        plt.title(title, fontsize=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


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
    model = create_model(
        model_name=config.get("model_name", "resnet18"),
        num_classes=int(config["num_classes"]),
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    xai_df = val_df[val_df["image_id"].isin(xai_ids)].reset_index(drop=True)
    xai_dataset = VinDrBinaryDataset(
        xai_df,
        transform=get_val_transform(int(config["image_size"])),
    )

    output_subdir = heatmap_dir / run_type
    output_subdir.mkdir(parents=True, exist_ok=True)

    gradcam_rows = []

    for idx in range(len(xai_dataset)):
        image_tensor, label, image_id = xai_dataset[idx]

        heatmap, pred_class = gradcam.generate(image_tensor)

        save_path = output_subdir / (
            f"{image_id}_label{label.item()}_pred{pred_class}.png"
        )

        if round_num is None:
            title = f"{image_id} | label={label.item()} | pred={pred_class}"
        else:
            title = (
                f"FL round {round_num} | "
                f"{image_id} | label={label.item()} | pred={pred_class}"
            )

        save_gradcam_overlay(
            image_tensor=image_tensor,
            heatmap=heatmap,
            save_path=save_path,
            title=title,
        )

        row = {
            "run_type": run_type,
            "image_id": image_id,
            "true_label": int(label.item()),
            "pred_class_for_gradcam": int(pred_class),
            "heatmap_path": str(save_path),
        }

        if round_num is not None:
            row["round"] = round_num

        gradcam_rows.append(row)

    gradcam.remove_hooks()

    gradcam_df = pd.DataFrame(gradcam_rows)

    if run_type == "centralized_debug":
        output_path = metrics_dir / "centralized_gradcam_outputs.csv"
    elif run_type == "flower_fedavg_debug":
        output_path = metrics_dir / "flower_fedavg_gradcam_outputs.csv"
    else:
        output_path = metrics_dir / f"{run_type}_gradcam_outputs.csv"

    gradcam_df.to_csv(output_path, index=False)

    print(f"Saved {run_type} Grad-CAM metadata to:", output_path)
    print(f"Saved {run_type} Grad-CAM overlays to:", output_subdir)
    print(gradcam_df.head())

    return output_path