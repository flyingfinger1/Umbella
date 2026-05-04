"""Single-image inference for the leaf (real-photo) classifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.leaf.model import load_leaf_classifier
from src.leaf.dataset import LeafImageDataset
from src.leaf import SPECIES_ORDER

from .predict import SPECIES_GERMAN


def preprocess_leaf(image: Image.Image | np.ndarray | Path | str,
                    size: int = 224) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        img = image.convert("RGB")

    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - LeafImageDataset.IMAGENET_MEAN) / LeafImageDataset.IMAGENET_STD
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)


@torch.no_grad()
def predict_leaf(model, image_tensor: torch.Tensor, top_k: int = 3) -> dict:
    device = next(model.parameters()).device
    logits = model(image_tensor.to(device))
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    ranked = [(SPECIES_ORDER[i], SPECIES_GERMAN.get(SPECIES_ORDER[i], "?"),
               float(probs[i])) for i in order]
    return {
        "top1_species": ranked[0][0],
        "top1_german": ranked[0][1],
        "top1_confidence": ranked[0][2],
        "top_k": ranked[:top_k],
    }


def predict_leaf_image_path(model, path, top_k: int = 3) -> dict:
    return predict_leaf(model, preprocess_leaf(path), top_k=top_k)
