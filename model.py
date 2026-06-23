from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_binary_resnet18(pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet-18 model with a binary classifier head.
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)

    original_feature_dim = model.fc.in_features
    model.fc = nn.Linear(original_feature_dim, 2)

    return model


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    metadata: dict | None = None,
) -> None:
    """
    Save model weights on CPU for portability.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    cpu_state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    payload = {
        "model_state_dict": cpu_state_dict,
    }

    if metadata is not None:
        payload.update(metadata)

    torch.save(payload, checkpoint_path)


def load_binary_resnet18_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Load an already fine-tuned binary ResNet-18.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = build_binary_resnet18(pretrained=False)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint