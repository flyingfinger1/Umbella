"""Render-time augmentations to make synthetic Apiaceae images look more
plant-photo-like and force the downstream model to ignore lighting/background.

Three components:
  1. natural background (procedural sky-to-ground gradient with noise)
  2. depth-derived Lambert shading (cheaper than ray tracing; gives the plant
     a sense of front/back lit volume, no real self-shadowing)
  3. per-scene color jitter (white-balance / saturation variation)

All operate on a single rendered triplet (rgb, label, depth). The label map
defines the foreground mask — augmentation never touches background pixels of
the label map itself, only their RGB.
"""

from __future__ import annotations

import numpy as np


def make_natural_background(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Procedural sky-to-ground gradient with chromatic noise. RGB uint8."""
    H, W = shape
    horizon = int(H * rng.uniform(0.45, 0.70))

    # randomized palettes for variety
    sky_warm = rng.uniform(-15, 15, 3)
    ground_warm = rng.uniform(-15, 15, 3)
    sky_top = np.array([170, 195, 225]) + sky_warm
    sky_bot = np.array([225, 230, 235]) + sky_warm
    ground_top = np.array([120, 150, 80]) + ground_warm   # near horizon - greener
    ground_bot = np.array([85, 75, 55]) + ground_warm     # bottom - browner

    rows = np.arange(H, dtype=np.float32)                 # (H,)
    sky_t = np.clip(rows / max(horizon, 1), 0, 1)[:, None]      # (H, 1)
    ground_t = np.clip((rows - horizon) / max(H - horizon, 1), 0, 1)[:, None]  # (H, 1)
    sky = sky_top + (sky_bot - sky_top) * sky_t           # (H, 3)
    grnd = ground_top + (ground_bot - ground_top) * ground_t  # (H, 3)

    is_sky = (rows < horizon)[:, None]                    # (H, 1)
    row_color = np.where(is_sky, sky, grnd)               # (H, 3)
    bg = np.broadcast_to(row_color[:, None, :], (H, W, 3)).copy()  # (H, W, 3)

    # vertical streak noise (pseudo-grass below horizon)
    streak_noise = rng.normal(0, 6, size=(H, W, 3))
    streak_noise[:horizon] *= 0.4
    bg = bg + streak_noise

    # mild blur via low-pass average to soften the grain
    k = 3
    pad = np.pad(bg, ((k, k), (k, k), (0, 0)), mode="edge")
    smoothed = np.zeros_like(bg)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            smoothed += pad[k + dy:k + dy + H, k + dx:k + dx + W]
    smoothed /= (2 * k + 1) ** 2
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def lambert_shading(depth: np.ndarray, fg_mask: np.ndarray, light_dir: np.ndarray | None = None
                    ) -> np.ndarray:
    """Compute a per-pixel brightness factor in roughly [0.55, 1.15] from
    depth-derived surface orientation. Background pixels return 1.0.
    """
    if light_dir is None:
        light_dir = np.array([0.3, -0.5, 1.0])
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-12)

    # depth gradient using simple central differences (zero outside foreground)
    d = np.where(fg_mask, depth, 0.0).astype(np.float32)
    gx = np.zeros_like(d)
    gy = np.zeros_like(d)
    gx[:, 1:-1] = 0.5 * (d[:, 2:] - d[:, :-2])
    gy[1:-1, :] = 0.5 * (d[2:, :] - d[:-2, :])

    # surface normal estimate (image space, depth = -forward)
    nx = -gx
    ny = -gy
    nz = np.ones_like(d)
    n_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= n_len; ny /= n_len; nz /= n_len

    # Lambert dot product (clamped) + ambient
    dot = nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2]
    dot = np.clip(dot, 0.0, 1.0)
    brightness = 0.55 + 0.6 * dot
    brightness[~fg_mask] = 1.0
    return brightness.astype(np.float32)


def color_jitter_factors(rng: np.random.Generator) -> np.ndarray:
    """Per-scene RGB scale (white-balance / saturation simulation)."""
    base = rng.uniform(0.88, 1.10, size=3)
    return base.astype(np.float32)


def augment_render(
    rgb: np.ndarray,
    label: np.ndarray,
    depth: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Apply background replacement + shading + color jitter to a render.

    Returns a new uint8 (H, W, 3) array; original `label` and `depth` are
    untouched (the caller still has the clean ground truth for training).
    """
    rng = np.random.default_rng(seed)
    H, W = label.shape
    fg = label > 0

    # 1. natural background under non-foreground pixels
    bg = make_natural_background((H, W), rng)
    out = np.where(fg[..., None], rgb, bg).astype(np.float32)

    # 2. Lambert-style shading on foreground only
    light_dir = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.7, -0.2), 1.0])
    shading = lambert_shading(depth, fg, light_dir)
    out *= shading[..., None]

    # 3. per-scene color jitter (whole image, not just foreground — emulates
    #    camera white balance which affects everything)
    out *= color_jitter_factors(rng)

    return np.clip(out, 0, 255).astype(np.uint8)
