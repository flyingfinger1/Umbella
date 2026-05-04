"""11: Sample point clouds from synthetic Apiaceae skeletons and visualize.

Each species gets one sampled cloud, written to data/clouds/synthetic/<species>/seed*.npz
and rendered as HTML. Size statistics let us tune density vs. realism.

Usage:
    .venv/Scripts/python.exe notebooks/11_synthetic_pointclouds.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SPECIES, generate_apiaceae       # noqa: E402
from src.geometry import sample_skeleton_pointcloud         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_HTML = ROOT / "notebooks" / "output" / "pointclouds"
OUT_NPZ = ROOT / "data" / "clouds" / "synthetic"
OUT_HTML.mkdir(parents=True, exist_ok=True)
OUT_NPZ.mkdir(parents=True, exist_ok=True)

PLOT_SUBSAMPLE = 40_000


def color_by_role(skel, label_per_point):
    """Map each point to a coarse role-color via its organ label."""
    organ_to_role = {1: "stem"}
    for i, (org, role) in enumerate(zip(skel.node_organ, skel.node_role)):
        if org not in organ_to_role:
            organ_to_role[org] = role
    role_color = {
        "stem": "#1f3d7a", "stem-junction": "#1f3d7a",
        "lateral": "#5a8fcf",
        "ray-base": "#d62728", "ray-mid": "#d62728", "ray-tip": "#d62728",
        "umbellet-center": "#222222",
        "pedicel-tip": "#3aaf3a",
        "bract": "#a06000", "bracteole": "#d97a00",
    }
    return [role_color.get(organ_to_role.get(int(l), ""), "#888888") for l in label_per_point]


def plot_cloud(xyz, colors, title, out_html: Path) -> None:
    if xyz.shape[0] > PLOT_SUBSAMPLE:
        idx = np.random.default_rng(0).choice(xyz.shape[0], PLOT_SUBSAMPLE, replace=False)
        xyz_p = xyz[idx]
        colors_p = [colors[i] for i in idx]
    else:
        xyz_p = xyz
        colors_p = colors
    fig = go.Figure(go.Scatter3d(
        x=xyz_p[:, 0], y=xyz_p[:, 1], z=xyz_p[:, 2],
        mode="markers", marker=dict(size=1.0, color=colors_p, opacity=0.7),
        name=f"{xyz.shape[0]:,} pts (showing {xyz_p.shape[0]:,})",
    ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)


def main() -> None:
    print(f"{'species':>22s}  {'edges':>6s}  {'cloud pts':>10s}  {'time (s)':>8s}")
    import time
    for key, spec in SPECIES.items():
        species_dir = OUT_NPZ / key
        species_dir.mkdir(exist_ok=True)
        params = spec.sample(seed=0)
        skel = generate_apiaceae(params)
        t0 = time.time()
        xyz, label = sample_skeleton_pointcloud(skel, points_per_mm2=0.5, noise_mm=0.3, seed=0)
        dt = time.time() - t0
        print(f"{key:>22s}  {len(skel.edges):>6d}  {xyz.shape[0]:>10,}  {dt:>8.2f}")

        # save npz
        np.savez_compressed(species_dir / "seed000.npz", xyz=xyz, label=label)

        # render
        colors = color_by_role(skel, label)
        out_html = OUT_HTML / f"{key}_seed0_cloud.html"
        plot_cloud(xyz, colors, f"{spec.german_name} — synthetic cloud "
                   f"({xyz.shape[0]:,} points)", out_html)

    # disk size of npzs
    sizes = [p.stat().st_size for p in OUT_NPZ.rglob("*.npz")]
    print(f"\nwrote {len(sizes)} npz files, total {sum(sizes) / 1024:.1f} KB")
    print(f"HTMLs in {OUT_HTML.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
