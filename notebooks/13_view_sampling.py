"""13: Apply Hidden Point Removal to make synthetic clouds one-sided.

Three views of one synthetic Heracleum:
  (a) full uniform sample (from notebook 11)
  (b) HPR from a single camera (typical "one photo" style)
  (c) HPR from 4 cameras around the plant (typical "structure-from-motion" coverage)

Re-runs the azimuthal-nonuniformity metric to confirm the gap to real LiDAR
has closed.

Usage:
    .venv/Scripts/python.exe notebooks/13_view_sampling.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SPECIES, generate_apiaceae          # noqa: E402
from src.geometry import (                                     # noqa: E402
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    hpr_multi_view,
    camera_around,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
OUT_DIR.mkdir(exist_ok=True)


def azimuthal_nonuniformity(xyz: np.ndarray, bins: int = 24) -> float:
    if xyz.shape[0] < 10:
        return 0.0
    centered = xyz - xyz.mean(0)
    sub = centered if xyz.shape[0] <= 2000 else centered[
        np.random.default_rng(1).choice(xyz.shape[0], 2000, replace=False)
    ]
    _, _, vh = np.linalg.svd(sub, full_matrices=False)
    axis = vh[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)
    proj_e1 = centered @ e1
    proj_e2 = centered @ e2
    theta = np.arctan2(proj_e2, proj_e1)
    hist, _ = np.histogram(theta, bins=bins, range=(-np.pi, np.pi))
    p = hist / max(hist.sum(), 1)
    uniform_p = 1.0 / bins
    return float(np.sqrt(((p - uniform_p) ** 2).mean()) / uniform_p)


def color_by_label(label):
    palette = {1: "#1f3d7a"}  # stem
    colors = []
    for l in label:
        l = int(l)
        if l == 1:
            colors.append("#1f3d7a")
        elif l < 50:
            colors.append("#d62728")  # rays
        else:
            colors.append("#3aaf3a")  # pedicels/bracteoles
    return colors


def plot_cloud(xyz, label, title: str, out_html: Path, sub: int = 30_000) -> None:
    if xyz.shape[0] > sub:
        idx = np.random.default_rng(0).choice(xyz.shape[0], sub, replace=False)
        xyz = xyz[idx]; label = label[idx]
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(size=1.0, color=color_by_label(label), opacity=0.8),
        name=f"{xyz.shape[0]:,} pts",
    ))
    fig.update_layout(title=title, scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out_html)


def main() -> None:
    spec = SPECIES["Heracleum_sphondylium"]
    skel = generate_apiaceae(spec.sample(seed=0))

    # bump density 5x to compensate for HPR keeping ~30-50% of points
    xyz_full, label_full = sample_skeleton_pointcloud(
        skel, points_per_mm2=2.5, noise_mm=0.3, seed=0,
    )
    print(f"full uniform sample: {xyz_full.shape[0]:,} pts")

    target = xyz_full.mean(0)
    bbox_diag = float(np.linalg.norm(xyz_full.max(0) - xyz_full.min(0)))

    # single camera at azimuth 0, elevation 10
    cam_single = camera_around(target, bbox_diag, azimuth_deg=0, elevation_deg=10)
    idx_single = hpr_visible_indices(xyz_full, cam_single, radius_factor=1000.0)
    xyz_single, label_single = xyz_full[idx_single], label_full[idx_single]
    print(f"HPR single view:     {xyz_single.shape[0]:,} pts  "
          f"({100 * len(idx_single) / len(xyz_full):.1f}% kept)")

    # four cameras around the plant
    cams_multi = [camera_around(target, bbox_diag, azimuth_deg=a, elevation_deg=10)
                  for a in (0, 90, 180, 270)]
    idx_multi = hpr_multi_view(xyz_full, cams_multi, radius_factor=1000.0)
    xyz_multi, label_multi = xyz_full[idx_multi], label_full[idx_multi]
    print(f"HPR 4-view:          {xyz_multi.shape[0]:,} pts  "
          f"({100 * len(idx_multi) / len(xyz_full):.1f}% kept)")

    # azimuthal nonuniformity (whole-plant)
    print()
    print(f"{'variant':>20s}  {'azim_nonuniform':>15s}")
    for name, xyz in [("full uniform", xyz_full),
                      ("HPR single view", xyz_single),
                      ("HPR 4 views", xyz_multi)]:
        print(f"{name:>20s}  {azimuthal_nonuniformity(xyz):>15.2f}")

    # also per-organ check (stem only) — this is the most diagnostic
    print(f"\n{'variant (stem only)':>20s}  {'azim_nonuniform':>15s}")
    for name, xyz, lbl in [("full uniform", xyz_full, label_full),
                           ("HPR single view", xyz_single, label_single),
                           ("HPR 4 views", xyz_multi, label_multi)]:
        stem = xyz[lbl == 1]
        print(f"{name:>20s}  {azimuthal_nonuniformity(stem):>15.2f}")

    plot_cloud(xyz_full, label_full,   "Heracleum — full uniform sample",
               OUT_DIR / "view_full.html")
    plot_cloud(xyz_single, label_single, "Heracleum — HPR single view (one camera)",
               OUT_DIR / "view_single.html")
    plot_cloud(xyz_multi, label_multi,   "Heracleum — HPR 4 views",
               OUT_DIR / "view_multi.html")
    print(f"\nwrote view_full / view_single / view_multi HTMLs in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
