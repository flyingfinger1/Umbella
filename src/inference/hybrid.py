"""Hybrid inference combining the synth-trained CNN with the iNat-trained leaf classifier.

The two models see complementary information:
  - Synth model: structural diagnostics encoded in the L-system (ray count,
    stem speckles, bracteole reflex, pedicel color)
  - Leaf model: real-photo textures, leaf shape, and lighting/scene cues
    that the synth pipeline cannot model

Three ensemble strategies are exposed:
  - "soft":  weighted mean of class probabilities
  - "max":   per-class take the higher of the two probabilities
  - "vote":  each model votes for its top-1; tie → break by combined confidence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.inference.predict import preprocess_image, SPECIES_GERMAN
from src.inference.predict_leaf import preprocess_leaf
from src.models.classifier import SPECIES_ORDER


@torch.no_grad()
def _probs(model, x: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    logits = model(x.to(device))
    return F.softmax(logits, dim=1).cpu().numpy()[0]


def predict_hybrid(
    synth_model,
    leaf_model,
    image: Path | str | Image.Image | np.ndarray,
    strategy: str = "soft",
    synth_weight: float = 0.5,
    leaf_weight: float = 0.5,
    top_k: int = 3,
) -> dict:
    """Run both models on the same image and combine their predictions.

    Returns:
        {
          "top1_species": str, "top1_german": str, "top1_confidence": float,
          "top_k": [(species, german, prob), ...],
          "strategy": str,
          "synth_probs": dict, "leaf_probs": dict, "combined_probs": dict,
        }
    """
    synth_x = preprocess_image(image)        # 384x384, per-image mean
    leaf_x = preprocess_leaf(image)           # 224x224, ImageNet norm
    synth_probs = _probs(synth_model, synth_x)
    leaf_probs = _probs(leaf_model, leaf_x)

    if strategy == "soft":
        w = synth_weight + leaf_weight
        combined = (synth_weight * synth_probs + leaf_weight * leaf_probs) / max(w, 1e-9)
    elif strategy == "max":
        combined = np.maximum(synth_probs, leaf_probs)
        combined = combined / max(combined.sum(), 1e-9)
    elif strategy == "confidence":
        # whichever model is most confident in its top-1 wins, full weight
        s_top = float(synth_probs.max())
        l_top = float(leaf_probs.max())
        combined = synth_probs.copy() if s_top >= l_top else leaf_probs.copy()
    elif strategy == "leaf_priority":
        # only override leaf if synth is much more confident (>90%) AND
        # leaf is uncertain (<50%) — handles the "synth overconfident on
        # OOD photos" failure mode
        s_top = float(synth_probs.max())
        l_top = float(leaf_probs.max())
        if s_top > 0.90 and l_top < 0.50:
            combined = synth_probs.copy()
        else:
            combined = leaf_probs.copy()
    elif strategy == "vote":
        synth_top = int(np.argmax(synth_probs))
        leaf_top = int(np.argmax(leaf_probs))
        combined = np.zeros_like(synth_probs)
        if synth_top == leaf_top:
            combined[synth_top] = (synth_probs[synth_top] + leaf_probs[leaf_top]) / 2
        else:
            combined[synth_top] = synth_probs[synth_top] * synth_weight
            combined[leaf_top] = leaf_probs[leaf_top] * leaf_weight
        # remainder spread uniformly so probs sum to 1
        used = combined.sum()
        residual = max(1.0 - used, 0.0)
        zero_mask = combined == 0
        if zero_mask.any():
            combined[zero_mask] = residual / zero_mask.sum()
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    order = np.argsort(combined)[::-1]
    ranked = [(SPECIES_ORDER[i], SPECIES_GERMAN.get(SPECIES_ORDER[i], "?"),
               float(combined[i])) for i in order]
    return {
        "top1_species": ranked[0][0],
        "top1_german": ranked[0][1],
        "top1_confidence": ranked[0][2],
        "top_k": ranked[:top_k],
        "strategy": strategy,
        "synth_probs": {SPECIES_ORDER[i]: float(synth_probs[i]) for i in range(6)},
        "leaf_probs": {SPECIES_ORDER[i]: float(leaf_probs[i]) for i in range(6)},
        "combined_probs": {SPECIES_ORDER[i]: float(combined[i]) for i in range(6)},
    }
