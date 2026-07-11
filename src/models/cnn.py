import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = {
    "resnet18",
}


def build_resnet18_binary(
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """Build a ResNet-18 classifier for binary VinDr classification.

    Parameters
    ----------
    num_classes:
        Number of output classes. Phase 5 and Phase 6 binary experiments
        use two logits.

    pretrained:
        When True, initialize the model with ImageNet weights.
        When False, preserve the confirmed Phase 5 random initialization.
    """
    if num_classes <= 1:
        raise ValueError(
            "num_classes must be greater than 1 for classification."
        )

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def create_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = False,
) -> nn.Module:
    """Create a supported classification model.

    The default behavior remains compatible with Phase 5:
    ResNet-18 with random initialization.
    """
    normalized_name = model_name.strip().lower()

    if normalized_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    if normalized_name == "resnet18":
        return build_resnet18_binary(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise RuntimeError(
        f"Model dispatch failed for supported model: {normalized_name}"
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that are updated during training."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model: nn.Module) -> int:
    """Count all model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )