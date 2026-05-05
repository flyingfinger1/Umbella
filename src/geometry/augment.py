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

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


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


def make_texture_background(shape: tuple[int, int], rng: np.random.Generator,
                             image_pool: list[Path],
                             light_blur: tuple[float, float] | None = (0.5, 2.5),
                             ) -> np.ndarray:
    """Sample a random subject-free texture from the curated pool, resize+crop
    to fit, optionally apply light extra blur (BG textures are mostly already
    out-of-focus or texture-only, so we only need light additional softening).
    """
    H, W = shape
    path = image_pool[int(rng.integers(0, len(image_pool)))]
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # blank fallback
        return np.full((H, W, 3), 200, dtype=np.uint8)
    iw, ih = img.size
    scale = max(H / ih, W / iw) * float(rng.uniform(1.0, 1.3))
    img = img.resize((max(int(round(iw * scale)), W),
                      max(int(round(ih * scale)), H)), Image.BILINEAR)
    iw, ih = img.size
    left = int(rng.integers(0, max(iw - W + 1, 1)))
    top = int(rng.integers(0, max(ih - H + 1, 1)))
    img = img.crop((left, top, left + W, top + H))
    if light_blur is not None:
        radius = float(rng.uniform(light_blur[0], light_blur[1]))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.uint8)


# legacy: superseded by make_texture_background, kept for back-compat
PROCEDURAL_PALETTES = {
    "grass":  (np.array([60, 95, 45]),  np.array([135, 175, 90])),
    "soil":   (np.array([55, 45, 35]),  np.array([130, 105, 75])),
    "hedge":  (np.array([35, 60, 30]),  np.array([90, 130, 70])),
    "wall":   (np.array([130, 125, 115]), np.array([200, 195, 185])),
    "sky":    (np.array([130, 165, 200]), np.array([220, 230, 235])),
}


def procedural_outdoor_bg(shape: tuple[int, int], rng: np.random.Generator,
                          palette: str | None = None) -> np.ndarray:
    """Subject-free background. Multi-frequency smooth-noise field, color-mapped
    to a palette (grass / soil / hedge / wall / sky). Optionally with a vertical
    gradient between two palettes (top sky → bottom grass)."""
    H, W = shape
    if palette is None:
        palette = str(rng.choice(list(PROCEDURAL_PALETTES)))
    base, bright = PROCEDURAL_PALETTES[palette]

    # multi-octave smooth noise via low-res random + bilinear upsample
    noise = np.zeros((H, W), dtype=np.float32)
    weights_sum = 0.0
    for octave_size in (8, 16, 32, 64):
        small = rng.uniform(0.0, 1.0, size=(octave_size, octave_size)).astype(np.float32)
        img = Image.fromarray((small * 255).astype(np.uint8))
        up = np.asarray(img.resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        weight = 1.0 / octave_size
        noise += up * weight
        weights_sum += weight
    noise /= weights_sum
    # normalize to full [0, 1]
    nmin, nmax = float(noise.min()), float(noise.max())
    noise = (noise - nmin) / max(nmax - nmin, 1e-6)

    # 30% of the time, add a vertical gradient between two palettes (sky over grass)
    if rng.random() < 0.3:
        other = str(rng.choice(list(PROCEDURAL_PALETTES)))
        base2, bright2 = PROCEDURAL_PALETTES[other]
        rows = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None, None]   # (H, 1, 1)
        col_a = (base + (bright - base) * noise[..., None])                # (H, W, 3)
        col_b = (base2 + (bright2 - base2) * noise[..., None])
        bg = col_a * (1 - rows) + col_b * rows
    else:
        bg = base + (bright - base) * noise[..., None]

    # add per-pixel chroma jitter
    bg = bg + rng.normal(0, 4, size=bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_inat_background(shape: tuple[int, int], rng: np.random.Generator,
                         image_pool: list[Path],
                         blur_radius: float | tuple[float, float] = (3.0, 8.0)
                         ) -> np.ndarray:
    """Sample a random iNat image, resize+crop to `shape`, blur.

    `blur_radius` may be a single value or a (lo, hi) range — in the latter
    case a value is sampled per call to mimic depth-of-field variation
    instead of uniform heavy smear.
    """
    if isinstance(blur_radius, tuple):
        blur_radius = float(rng.uniform(blur_radius[0], blur_radius[1]))
    H, W = shape
    path = image_pool[int(rng.integers(0, len(image_pool)))]
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        # fall back to procedural BG if file unreadable
        return make_natural_background(shape, rng)

    iw, ih = img.size
    # zoom in MORE (1.5–2.5×) so we sample only a fraction of the iNat photo;
    # combined with a corner-biased crop this avoids the centered subject
    scale = max(H / ih, W / iw) * float(rng.uniform(1.5, 2.5))
    img = img.resize((max(int(round(iw * scale)), W),
                      max(int(round(ih * scale)), H)), Image.BILINEAR)
    iw, ih = img.size
    # corner-biased crop: pick one of 4 corners, then jitter slightly
    corner = int(rng.integers(0, 4))
    cx = 0 if corner in (0, 2) else iw - W
    cy = 0 if corner in (0, 1) else ih - H
    jitter_x = int(rng.integers(0, max(iw - W, 1)) // 4)
    jitter_y = int(rng.integers(0, max(ih - H, 1)) // 4)
    left = max(0, min(iw - W, cx + (jitter_x if cx == 0 else -jitter_x)))
    top = max(0, min(ih - H, cy + (jitter_y if cy == 0 else -jitter_y)))
    img = img.crop((left, top, left + W, top + H))

    img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return np.asarray(img, dtype=np.uint8)


def pick_random_background(shape: tuple[int, int], rng: np.random.Generator,
                            inat_pool: list[Path] | None = None,
                            procedural_prob: float = 0.7) -> np.ndarray:
    """Choose a background per call: mostly procedural (subject-free), some iNat
    edge-crops for texture variety. Default 70 % procedural / 30 % iNat."""
    if inat_pool and rng.random() > procedural_prob:
        return make_inat_background(shape, rng, inat_pool)
    return procedural_outdoor_bg(shape, rng)


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
    bg_pool: list[Path] | None = None,
    inat_background_pool: list[Path] | None = None,   # legacy name kept
    inat_bg_prob: float = 0.5,                         # legacy
) -> np.ndarray:
    """Apply background replacement + shading + color jitter to a render.

    `bg_pool` (preferred): list of paths to subject-free background images.
    A random one is sampled per call and used as the foreground's background.

    `inat_background_pool` is a legacy alias kept for back-compat; if provided
    without `bg_pool`, falls back to the older procedural-vs-iNat mix logic.

    Returns a new uint8 (H, W, 3) array; original `label` and `depth` are
    untouched.
    """
    rng = np.random.default_rng(seed)
    H, W = label.shape
    fg = label > 0

    # 1. background
    if bg_pool:
        bg = make_texture_background((H, W), rng, bg_pool)
    elif inat_background_pool:
        # legacy path
        bg = pick_random_background((H, W), rng, inat_pool=inat_background_pool,
                                     procedural_prob=1.0 - inat_bg_prob)
    else:
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
