"""Project a labeled point cloud onto a 2D image (perspective camera).

Point splatting via vectorized scatter with a circular kernel; nearest-z wins
per pixel (z-buffer). Returns RGB image, label map, and depth map together —
useful as training data: the RGB is the input, the label/depth are dense
ground truth for the 2D->3D model.
"""

from __future__ import annotations

import numpy as np


# Default plant-like color palette — labels in our synth correspond to
# different organ types (stem, rays, pedicels, bracts/bracteoles).
DEFAULT_BACKGROUND = (235, 235, 230)


# Per-role colors. Keys cover both Apiaceae synthetic roles and Pheno4D-derived
# skeleton roles. RGB values aim for visually distinct (not photorealistic)
# rendering so a downstream model can pick up organ type at pixel level.
ROLE_TO_RGB: dict[str, tuple[int, int, int]] = {
    # main stem
    "stem": (60, 90, 50),
    "stem-junction": (60, 90, 50),
    "stem-speckle": (105, 50, 90),  # purple-brown — Conium maculatum diagnostic
    "lateral": (90, 120, 60),
    # primary rays of an umbel
    "ray-base": (110, 150, 80),
    "ray-mid": (110, 150, 80),
    "ray-tip": (110, 150, 80),
    "umbellet-center": (80, 110, 60),
    # flower stalks (visually = white flower in real plants)
    "pedicel-tip": (240, 240, 225),
    # diagnostic bract / bracteole structures
    "bract": (130, 90, 50),       # brown
    "bracteole": (210, 150, 60),  # ocher (Aethusa diagnostic when reflexed)
    # Pheno4D leaf roles (kept distinct so real and synthetic clouds render
    # in the same visual space)
    "leaf-base": (90, 140, 75),
    "leaf-mid": (90, 140, 75),
    "leaf-tip": (110, 165, 90),
}


def default_color_for_label(label: int) -> tuple[int, int, int]:
    """Fallback for callers that don't provide role info — coarse 3-tier palette."""
    if label == 1:
        return ROLE_TO_RGB["stem"]
    if 2 <= label <= 30:
        return ROLE_TO_RGB["ray-mid"]
    return ROLE_TO_RGB["pedicel-tip"]


def role_aware_color_callable(
    organ_to_role: dict[int, str],
    default=(180, 180, 180),
    role_overrides: dict[str, tuple[int, int, int]] | None = None,
):
    """Build a `label_to_color` callable from an {organ_id: role_str} mapping.

    `role_overrides` lets callers redefine specific roles per render — used to
    e.g. render Pastinaca's pedicels yellow instead of the default white.
    """
    color_map = {**ROLE_TO_RGB, **(role_overrides or {})}
    def fn(label: int) -> tuple[int, int, int]:
        return color_map.get(organ_to_role.get(int(label), ""), default)
    return fn


def _camera_basis(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray
                  ) -> np.ndarray:
    """Right-handed camera frame: rows = (right, up, forward) world axes.
    Forward points from camera toward target."""
    forward = target - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    cam_up = np.cross(right, forward)
    return np.stack([right, cam_up, forward], axis=0)


def render_pointcloud(
    points: np.ndarray,
    labels: np.ndarray,
    camera_pos: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
    image_size: tuple[int, int] = (512, 512),
    fov_deg: float = 35.0,
    point_radius_px: int = 2,
    background_rgb: tuple[int, int, int] = DEFAULT_BACKGROUND,
    label_to_color = default_color_for_label,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a labeled point cloud to an image.

    Returns:
        rgb       (H, W, 3) uint8
        label_map (H, W)    int32   — 0 means background
        depth     (H, W)    float32 — np.inf means background
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    H, W = image_size

    # camera-frame coordinates
    R = _camera_basis(camera_pos, target, up)
    pts_cam = (points.astype(np.float64) - camera_pos) @ R.T  # (N, 3)
    z = pts_cam[:, 2]                                          # forward = +z
    mask = z > 1e-3
    pts_cam = pts_cam[mask]; z = z[mask]
    lbl = labels[mask].astype(np.int32)

    # perspective projection (square pixels)
    f = (W / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    u = pts_cam[:, 0] / z * f + W / 2.0
    v = -pts_cam[:, 1] / z * f + H / 2.0   # flip y for image-row convention

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    # circular kernel offsets
    r = int(point_radius_px)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    kmask = dx * dx + dy * dy <= r * r
    kx = dx[kmask].astype(np.int32)
    ky = dy[kmask].astype(np.int32)
    nk = kx.size

    # expand: for each input point, produce nk pixel candidates
    all_x = (ui[:, None] + kx[None, :]).ravel()
    all_y = (vi[:, None] + ky[None, :]).ravel()
    all_z = np.repeat(z, nk)
    all_l = np.repeat(lbl, nk)

    # in-bounds filter
    inb = (all_x >= 0) & (all_x < W) & (all_y >= 0) & (all_y < H)
    all_x = all_x[inb]; all_y = all_y[inb]
    all_z = all_z[inb]; all_l = all_l[inb]

    # sort farthest-first so nearest writes win when scattered
    order = np.argsort(-all_z)
    all_x = all_x[order]; all_y = all_y[order]
    all_z = all_z[order]; all_l = all_l[order]

    # init buffers
    depth = np.full((H, W), np.inf, dtype=np.float32)
    label_map = np.zeros((H, W), dtype=np.int32)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = background_rgb[0]
    rgb[..., 1] = background_rgb[1]
    rgb[..., 2] = background_rgb[2]

    # color per scatter element via vectorized table lookup
    unique_lbls = np.unique(all_l)
    color_table = np.zeros((unique_lbls.max() + 1, 3), dtype=np.uint8)
    for ul in unique_lbls:
        color_table[ul] = np.array(label_to_color(int(ul)), dtype=np.uint8)
    colors = color_table[all_l]

    # scatter (later writes overwrite earlier — last in array wins, which is nearest after sort)
    flat = all_y * W + all_x
    depth.ravel()[flat] = all_z
    label_map.ravel()[flat] = all_l
    # for rgb need per-channel scatter
    rgb.reshape(-1, 3)[flat] = colors

    return rgb, label_map, depth
