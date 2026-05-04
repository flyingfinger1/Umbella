"""Single-image inference helper for the Apiaceae classifier.

Loads a trained checkpoint and runs species prediction on a real photo.
Replicates the training-time preprocessing exactly (resize+center-crop to
square, /255, per-image mean-zero, CHW tensor) so the model sees the same
distribution it was trained on.

Usage from Python:

    from src.inference import load_model, predict_image_path
    model = load_model("data/models/v6_classifier.pt")
    result = predict_image_path(model, "myphoto.jpg")
    print(result["top1_species"], result["top1_confidence"])

Or from the command line via notebooks/21_predict.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.classifier import ApiaceaeCNN, SPECIES_ORDER


SPECIES_GERMAN = {
    "Heracleum_sphondylium": "Wiesen-Bärenklau",
    "Conium_maculatum": "Gefleckter Schierling",
    "Daucus_carota": "Wilde Möhre",
    "Anthriscus_sylvestris": "Wiesen-Kerbel",
    "Aethusa_cynapium": "Hundspetersilie",
    "Pastinaca_sativa": "Pastinak",
}


def load_model(checkpoint_path: Path | str, device: str | torch.device | None = None
               ) -> ApiaceaeCNN:
    """Load a trained ApiaceaeCNN checkpoint."""
    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model = ApiaceaeCNN(n_classes=6).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_image(image: Image.Image | np.ndarray | Path | str,
                     target_size: int = 384) -> torch.Tensor:
    """Real photo (any size/aspect) -> (1, 3, target_size, target_size) tensor.

    Resize so the shorter side matches `target_size`, then center-crop to
    square. Match training preprocessing: /255, per-image mean center, CHW.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        img = image.convert("RGB")

    w, h = img.size
    scale = target_size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0           # (H, W, 3) in [0, 1]
    arr = arr - arr.mean(axis=(0, 1), keepdims=True)          # per-image mean-zero
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return x


@torch.no_grad()
def predict(model: ApiaceaeCNN, image_tensor: torch.Tensor, top_k: int = 3) -> dict:
    """Run model and return ranked predictions with probabilities."""
    device = next(model.parameters()).device
    x = image_tensor.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    ranked = [(SPECIES_ORDER[i], SPECIES_GERMAN.get(SPECIES_ORDER[i], "?"),
               float(probs[i])) for i in order]
    return {
        "top1_species": ranked[0][0],
        "top1_german": ranked[0][1],
        "top1_confidence": ranked[0][2],
        "top_k": ranked[:top_k],
        "all_probs": {SPECIES_ORDER[i]: float(probs[i]) for i in range(6)},
    }


def predict_image_path(model: ApiaceaeCNN, path: Path | str, top_k: int = 3) -> dict:
    """Convenience: preprocess + predict from a file path."""
    return predict(model, preprocess_image(path), top_k=top_k)
