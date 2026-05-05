"""Compact CNN for synthetic Apiaceae species classification.

This is the first end-to-end test of the rendering -> ML loop. The point is
NOT to get state-of-the-art accuracy, but to verify that:
  - Render-pipeline information is sufficient for a model to learn species
  - Pipeline plumbing (dataset, batching, training, eval) all works

Model is intentionally small: ~270k params, trainable on CPU in minutes.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------- model -----------------------------------------------------------


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class ApiaceaeCNN(nn.Module):
    """384x384 RGB -> 6 species logits. Stride-2 conv stack + global-avg-pool head."""

    def __init__(self, n_classes: int = 6, in_channels: int = 3):
        super().__init__()
        self.b1 = _block(in_channels, 32)   # 384 -> 192
        self.b2 = _block(32, 64)            # 192 -> 96
        self.b3 = _block(64, 128)           # 96  -> 48
        self.b4 = _block(128, 192)          # 48  -> 24
        self.head = nn.Linear(192, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


# ---------- dataset ---------------------------------------------------------


SPECIES_ORDER = [
    "Heracleum_sphondylium",
    "Conium_maculatum",
    "Daucus_carota",
    "Anthriscus_sylvestris",
    "Aethusa_cynapium",
    "Pastinaca_sativa",
]


class ApiaceaeImageDataset(Dataset):
    """Loads (rgb_chw_float, species_label_int) pairs from training metadata.

    From v9 onward, supports online augmentation: if `online_aug_pool` is
    given, the loader applies a fresh `augment_render(...)` call per __getitem__
    using a different random BG/shading/color-jitter combination each time.
    This decouples augmentation variability from dataset size — 12 epochs
    × 4800 examples = 57 600 unique training views instead of 4800.

    For online aug to work, the saved npz must contain rgb (clean), label,
    and depth — which build_dataset(augment=False) produces.
    """

    def __init__(self, root: Path | str, examples: list[dict],
                 augment: bool = False,
                 online_aug_pool: list[Path] | None = None):
        self.root = Path(root)
        self.examples = examples
        self.augment = augment
        self.online_aug_pool = online_aug_pool
        self.species_to_idx = {s: i for i, s in enumerate(SPECIES_ORDER)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        data = np.load(self.root / ex["image"])
        rgb = data["rgb"]                              # uint8 (H, W, 3)

        if self.online_aug_pool is not None:
            # apply fresh augmentation each load — fresh seed every time
            from src.geometry import augment_render
            label = data["label"]
            depth = data["depth"].astype(np.float32)
            # restore depth NaN convention (build saves 0 for bg → infer from label)
            depth = np.where(label > 0, depth, np.inf)
            rgb = augment_render(rgb, label, depth, seed=None,
                                  bg_pool=self.online_aug_pool)

        rgb = rgb.astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            rgb = rgb[:, ::-1].copy()
        rgb = rgb - rgb.mean(axis=(0, 1), keepdims=True)

        x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        y = int(self.species_to_idx[ex["species"]])
        return x, y


# ---------- splits ----------------------------------------------------------


def instance_stratified_split(
    examples: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split *by instance* (seed) within each species so that all views of one
    plant land in the same split. Avoids near-identical-view leakage."""
    rng = np.random.default_rng(seed)
    by_sp_seed = defaultdict(list)
    for e in examples:
        by_sp_seed[(e["species"], e["seed"])].append(e)

    train, val, test = [], [], []
    species = sorted({e["species"] for e in examples})
    for sp in species:
        seeds = sorted({s for s_sp, s in by_sp_seed if s_sp == sp})
        rng.shuffle(seeds)
        n = len(seeds)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        tr_seeds = set(seeds[:n_tr])
        va_seeds = set(seeds[n_tr:n_tr + n_va])
        te_seeds = set(seeds[n_tr + n_va:])
        for s in tr_seeds:
            train.extend(by_sp_seed[(sp, s)])
        for s in va_seeds:
            val.extend(by_sp_seed[(sp, s)])
        for s in te_seeds:
            test.extend(by_sp_seed[(sp, s)])
    return train, val, test
