"""Dataset for the iNaturalist-derived leaf classifier."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .fetch_inaturalist import INAT_TAXA


SPECIES_ORDER = list(INAT_TAXA.keys())  # same order as the synthetic side


class LeafImageDataset(Dataset):
    """Loads (rgb_chw_float, species_label_int) from data/leaf_images/<species>/*.jpg.

    Resize so the shorter side matches `size`, then center-crop to square,
    /255, ImageNet-mean normalize (since we'll fine-tune from ImageNet weights).
    Optional augmentation: horizontal flip + small random crop + color jitter.
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, root: Path | str, paths_with_labels: list[tuple[Path, int]],
                 size: int = 224, augment: bool = False):
        self.root = Path(root)
        self.entries = paths_with_labels
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        path, label = self.entries[i]
        img = Image.open(path).convert("RGB")

        # resize so shorter side = size (or size + slack for random crop)
        target = self.size + (32 if self.augment else 0)
        w, h = img.size
        scale = target / min(w, h)
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)

        # center crop (or random crop if augmenting)
        w, h = img.size
        if self.augment:
            left = np.random.randint(0, max(w - self.size, 1))
            top = np.random.randint(0, max(h - self.size, 1))
        else:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        img = img.crop((left, top, left + self.size, top + self.size))

        if self.augment and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment:
            # mild color jitter
            arr = arr * np.random.uniform(0.85, 1.15, size=3).astype(np.float32)
            arr = np.clip(arr, 0, 1)
        arr = (arr - self.IMAGENET_MEAN) / self.IMAGENET_STD

        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return x, int(label)


def collect_image_paths(root: Path | str) -> list[tuple[Path, int]]:
    """Walk data/leaf_images/<species>/*.jpg and return (path, species_idx) list."""
    root = Path(root)
    species_to_idx = {s: i for i, s in enumerate(SPECIES_ORDER)}
    entries = []
    for sp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = sp_dir.name
        if sp not in species_to_idx:
            continue
        idx = species_to_idx[sp]
        for img in sorted(sp_dir.glob("*.jpg")):
            entries.append((img, idx))
    return entries


def stratified_split(entries, train_frac: float = 0.8, val_frac: float = 0.1,
                      seed: int = 0):
    """Split per-species so each split sees every species. Within species,
    splits by observation id (first part of filename) to avoid duplicate-photo
    leak between train and test."""
    rng = np.random.default_rng(seed)
    by_species = defaultdict(list)
    for p, l in entries:
        by_species[l].append((p, l))

    train, val, test = [], [], []
    for label, items in by_species.items():
        # group by obs id (filename prefix before first underscore)
        by_obs = defaultdict(list)
        for p, l in items:
            obs_id = p.stem.split("_")[0]
            by_obs[obs_id].append((p, l))
        obs_ids = list(by_obs.keys())
        rng.shuffle(obs_ids)
        n = len(obs_ids)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        for o in obs_ids[:n_tr]:
            train.extend(by_obs[o])
        for o in obs_ids[n_tr:n_tr + n_va]:
            val.extend(by_obs[o])
        for o in obs_ids[n_tr + n_va:]:
            test.extend(by_obs[o])

    return train, val, test
