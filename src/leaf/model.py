"""ResNet-18 fine-tuned on iNaturalist Apiaceae leaf/whole-plant photos."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_leaf_classifier(n_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """ResNet-18 backbone with replaced final FC layer."""
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_leaf_classifier(checkpoint: Path | str,
                          device: str | torch.device | None = None) -> nn.Module:
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = build_leaf_classifier(n_classes=6, pretrained=False).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
