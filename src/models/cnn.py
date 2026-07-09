import torch.nn as nn
from torchvision import models


def build_resnet18_binary(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_model(model_name: str = "resnet18", num_classes: int = 2) -> nn.Module:
    if model_name != "resnet18":
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            "Phase 5 debug reproduction supports only 'resnet18'."
        )

    return build_resnet18_binary(num_classes=num_classes)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
