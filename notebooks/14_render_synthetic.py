"""14: Render synthetic plants to 2D images (the other end of the bridge).

Pipeline per plant:
    SpeciesSpec -> ApiaceaeParams -> Skeleton -> point cloud
                                               -> HPR (camera) -> visible cloud
                                                                -> render -> RGB + label + depth

Produces RGB / label / depth panels per species and saves them as a single HTML.

Usage:
    .venv/Scripts/python.exe notebooks/14_render_synthetic.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SPECIES, generate_apiaceae               # noqa: E402
from src.geometry import (                                          # noqa: E402
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    camera_around,
    render_pointcloud,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
OUT_DIR.mkdir(exist_ok=True)


def label_map_rgb(label_map: np.ndarray) -> np.ndarray:
    """Visualize labels as colors — distinct categorical hue per id."""
    H, W = label_map.shape
    rgb = np.full((H, W, 3), 240, dtype=np.uint8)
    palette = np.array([
        [240, 240, 240],   # 0 = bg
        [40, 60, 100],     # 1 = stem
    ], dtype=np.uint8)
    # extend with hashed colors for higher labels
    rng = np.random.default_rng(0)
    extra = (rng.integers(40, 230, size=(1024, 3))).astype(np.uint8)
    palette = np.vstack([palette, extra])
    rgb = palette[label_map % palette.shape[0]]
    return rgb


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.full((*depth.shape, 3), 240, dtype=np.uint8)
    d = depth.copy()
    lo, hi = float(np.percentile(d[finite], 2)), float(np.percentile(d[finite], 98))
    d_norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    d_norm[~finite] = 1.0
    g = (255 * (1.0 - d_norm)).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~finite] = (240, 240, 240)
    return rgb


def render_species(spec, seed: int = 0):
    skel = generate_apiaceae(spec.sample(seed=seed))
    xyz, lbl = sample_skeleton_pointcloud(skel, points_per_mm2=2.5, noise_mm=0.3, seed=seed)

    target = xyz.mean(0)
    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    cam = camera_around(target, bbox_diag, azimuth_deg=15, elevation_deg=15,
                        distance_factor=1.5)

    visible = hpr_visible_indices(xyz, cam, radius_factor=1000.0)
    xyz_v, lbl_v = xyz[visible], lbl[visible]

    rgb, lblmap, depth = render_pointcloud(
        xyz_v, lbl_v, camera_pos=cam, target=target,
        image_size=(384, 384), fov_deg=40.0, point_radius_px=2,
    )
    return rgb, lblmap, depth, len(visible)


def main() -> None:
    n_species = len(SPECIES)
    fig = make_subplots(
        rows=n_species, cols=3,
        subplot_titles=[f"{spec.german_name} — RGB" if c == 0
                        else f"{spec.german_name} — labels" if c == 1
                        else f"{spec.german_name} — depth"
                        for spec in SPECIES.values() for c in range(3)],
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )

    print(f"{'species':>22s}  {'visible pts':>12s}  {'foreground px':>14s}")
    for r, (key, spec) in enumerate(SPECIES.items(), start=1):
        rgb, lblmap, depth, n_vis = render_species(spec, seed=0)
        fg_px = int(np.isfinite(depth).sum())
        print(f"{key:>22s}  {n_vis:>12,}  {fg_px:>14,}")

        fig.add_trace(go.Image(z=rgb), row=r, col=1)
        fig.add_trace(go.Image(z=label_map_rgb(lblmap)), row=r, col=2)
        fig.add_trace(go.Image(z=depth_to_rgb(depth)), row=r, col=3)

    for i in range(1, n_species + 1):
        for j in range(1, 4):
            fig.update_xaxes(visible=False, row=i, col=j)
            fig.update_yaxes(visible=False, row=i, col=j)
    fig.update_layout(
        title="Synthetic Apiaceae renders — RGB / labels / depth",
        height=300 * n_species, width=1100, margin=dict(l=10, r=10, t=60, b=10),
    )
    out = OUT_DIR / "rendered_synthetic.html"
    fig.write_html(out)
    print(f"\nwrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
