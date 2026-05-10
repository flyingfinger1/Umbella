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
    # synthetic Apiaceae leaf roles (procedural leaf generator)
    "leaf-petiole":      (85, 125, 60),
    "leaf-rachis":       (95, 140, 70),
    "leaf-pinna-rachis": (105, 150, 75),
    "leaf-midrib":       (115, 165, 85),
    "leaf-edge":         (115, 165, 85),
    "leaf-blade":        (105, 155, 75),
}


def default_color_for_label(label: int) -> tuple[int, int, int]:
    """Fallback for callers that don't provide role info — coarse 3-tier palette."""
    if label == 1:
        return ROLE_TO_RGB["stem"]
    if 2 <= label <= 30:
        return ROLE_TO_RGB["ray-mid"]
    return ROLE_TO_RGB["pedicel-tip"]


DEFAULT_POINT_RADIUS_PX: int = 2

# Per-role splat radius. Most roles render as crisp lines / dots; leaflet
# blade points need a larger splat so neighbouring fill samples merge into
# a continuous filled surface rather than a visible dot pattern.
ROLE_TO_RADIUS_PX: dict[str, int] = {
    "leaf-blade": 3,
    # leaf stems: thin lines. Default 2 splatted them too wide vs. real
    # photos; 1 (3×3 kernel) is a better visual match.
    "leaf-petiole":      1,
    "leaf-rachis":       1,
    "leaf-pinna-rachis": 1,
    "leaf-midrib":       1,
    "leaf-edge":         1,
}


def role_aware_radius_callable(
    organ_to_role: dict[int, str],
    default: int = DEFAULT_POINT_RADIUS_PX,
    role_overrides: dict[str, int] | None = None,
):
    """Build a `label_to_radius` callable from an {organ_id: role_str} mapping.

    Mirrors `role_aware_color_callable`. Used so leaflet blade fill renders
    as a continuous green surface rather than a sparse dot pattern.
    """
    radius_map = {**ROLE_TO_RADIUS_PX, **(role_overrides or {})}
    def fn(label: int) -> int:
        return radius_map.get(organ_to_role.get(int(label), ""), default)
    return fn


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
    label_to_radius = None,
    polygons: list[tuple[np.ndarray, int]] | None = None,
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

    # per-point splat radius — uniform unless `label_to_radius` is given,
    # in which case point types like "leaf-blade" can splat wider so their
    # samples merge into a continuous filled surface instead of dot pattern.
    if label_to_radius is None:
        per_point_r = np.full(lbl.shape, int(point_radius_px), dtype=np.int32)
    else:
        unique = np.unique(lbl)
        rmap = np.full(int(unique.max()) + 1, int(point_radius_px), dtype=np.int32)
        for ul in unique:
            try:
                rmap[ul] = int(label_to_radius(int(ul)))
            except Exception:
                pass
        per_point_r = rmap[lbl]

    # build a kernel per radius value present, then concatenate scatter lists
    all_x_parts: list[np.ndarray] = []
    all_y_parts: list[np.ndarray] = []
    all_z_parts: list[np.ndarray] = []
    all_l_parts: list[np.ndarray] = []
    for r in np.unique(per_point_r):
        sel = per_point_r == r
        if not sel.any():
            continue
        r_int = int(r)
        dy, dx = np.mgrid[-r_int:r_int + 1, -r_int:r_int + 1]
        kmask = dx * dx + dy * dy <= r_int * r_int
        kx = dx[kmask].astype(np.int32)
        ky = dy[kmask].astype(np.int32)
        nk = kx.size
        ui_s = ui[sel]; vi_s = vi[sel]
        z_s = z[sel]; lbl_s = lbl[sel]
        all_x_parts.append((ui_s[:, None] + kx[None, :]).ravel())
        all_y_parts.append((vi_s[:, None] + ky[None, :]).ravel())
        all_z_parts.append(np.repeat(z_s, nk))
        all_l_parts.append(np.repeat(lbl_s, nk))

    all_x = np.concatenate(all_x_parts) if all_x_parts else np.zeros(0, np.int32)
    all_y = np.concatenate(all_y_parts) if all_y_parts else np.zeros(0, np.int32)
    all_z = np.concatenate(all_z_parts) if all_z_parts else np.zeros(0, np.float64)
    all_l = np.concatenate(all_l_parts) if all_l_parts else np.zeros(0, np.int32)

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

    # ---- pass 1: rasterize filled polygons (leaflet blades) -----------------
    # Polygons render before point-splats so that any in-leaf details
    # (midrib, vein points) drawn on top retain visibility, while the
    # leaflet body fills the rest of the leaflet area as a continuous
    # surface — solving the salt-and-pepper appearance the previous
    # blade-fill-points approach suffered from on large leaflets.
    if polygons:
        try:
            from PIL import Image as _Img, ImageDraw as _Draw
        except ImportError:
            _Img = _Draw = None
        if _Img is not None:
            for vtx_3d, organ_id in polygons:
                v_cam = (vtx_3d.astype(np.float64) - camera_pos) @ R.T
                z_v = v_cam[:, 2]
                if (z_v <= 1e-3).any():
                    continue                                    # behind camera
                u_v = v_cam[:, 0] / z_v * f + W / 2.0
                v_v = -v_cam[:, 1] / z_v * f + H / 2.0
                pts = [(int(round(x)), int(round(y))) for x, y in zip(u_v, v_v)]
                # quick reject if entirely off-screen
                if all(x < 0 or x >= W or y < 0 or y >= H for x, y in pts):
                    continue
                mean_depth = float(z_v.mean())
                # rasterize polygon mask via PIL
                mask_img = _Img.new("L", (W, H), 0)
                _Draw.Draw(mask_img).polygon(pts, fill=1)
                mask = np.array(mask_img, dtype=bool)
                if not mask.any():
                    continue
                update = mask & (mean_depth < depth)
                if not update.any():
                    continue
                color = np.array(label_to_color(int(organ_id)), dtype=np.uint8)
                rgb[update] = color
                depth[update] = np.float32(mean_depth)
                label_map[update] = int(organ_id)

    # ---- pass 2: point-splat scatter (skeleton — stems, veins, etc) ---------
    # color per scatter element via vectorized table lookup
    unique_lbls = np.unique(all_l) if all_l.size else np.zeros(0, np.int32)
    if unique_lbls.size:
        color_table = np.zeros((int(unique_lbls.max()) + 1, 3), dtype=np.uint8)
        for ul in unique_lbls:
            color_table[ul] = np.array(label_to_color(int(ul)), dtype=np.uint8)
        colors = color_table[all_l]

        # scatter — later writes overwrite earlier, and we sorted farthest-first
        # so nearest points win. Polygon depths are already in the buffer, so
        # any point closer than its polygon (e.g. midrib at the leaf surface)
        # also wins.
        flat = all_y * W + all_x
        # for the point-splat pass we additionally check against the polygon
        # depths already written, so points behind the leaflet stay hidden.
        existing = depth.ravel()[flat]
        winning = all_z <= existing + 1e-6
        flat = flat[winning]
        depth.ravel()[flat] = all_z[winning]
        label_map.ravel()[flat] = all_l[winning]
        rgb.reshape(-1, 3)[flat] = colors[winning]

    return rgb, label_map, depth
