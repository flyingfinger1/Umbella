"""04: Extract a plant skeleton (stem + per-leaf polylines) and visualize it.

Usage:
    .venv/Scripts/python.exe notebooks/04_skeleton.py [plant_id]
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D                       # noqa: E402
from src.geometry import extract_plant_skeleton        # noqa: E402

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def plot_skeleton(cloud, skel, out_html: Path, point_subsample: int = 40_000) -> None:
    rng = np.random.default_rng(0)
    plant_mask = cloud.instance != 0
    pts = cloud.xyz[plant_mask]
    if pts.shape[0] > point_subsample:
        pts = pts[rng.choice(pts.shape[0], point_subsample, replace=False)]

    traces = [go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=1.0, color="#bbbbbb", opacity=0.35),
        name=f"plant points (n={pts.shape[0]:,})",
    )]

    # edges: one Scatter3d per edge group (stem vs leaf vs join) for legend clarity
    role_color = {"stem": "#1f3d7a", "leaf-mid": "#d62728", "leaf-base": "#d62728", "leaf-tip": "#d62728"}
    # collect line segments per organ family
    stem_x, stem_y, stem_z = [], [], []
    leaf_x, leaf_y, leaf_z = [], [], []
    join_x, join_y, join_z = [], [], []
    for i, j in skel.edges:
        a, b = skel.nodes[i], skel.nodes[j]
        oi, oj = skel.node_organ[i], skel.node_organ[j]
        seg_x = [a[0], b[0], None]
        seg_y = [a[1], b[1], None]
        seg_z = [a[2], b[2], None]
        if oi == 1 and oj == 1:
            stem_x += seg_x; stem_y += seg_y; stem_z += seg_z
        elif oi != oj:
            join_x += seg_x; join_y += seg_y; join_z += seg_z
        else:
            leaf_x += seg_x; leaf_y += seg_y; leaf_z += seg_z

    traces.append(go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines",
                               line=dict(color="#1f3d7a", width=8), name="stem skeleton"))
    traces.append(go.Scatter3d(x=leaf_x, y=leaf_y, z=leaf_z, mode="lines",
                               line=dict(color="#d62728", width=4), name="leaf midribs"))
    traces.append(go.Scatter3d(x=join_x, y=join_y, z=join_z, mode="lines",
                               line=dict(color="#666666", width=2, dash="dot"), name="leaf↔stem join"))

    # node markers, colored by role
    role_marker = {"stem": "#1f3d7a", "leaf-base": "#000000", "leaf-tip": "#ffaa00", "leaf-mid": "#d62728"}
    for role, color in role_marker.items():
        idx = [i for i, r in enumerate(skel.node_role) if r == role]
        if not idx:
            continue
        n = skel.nodes[idx]
        traces.append(go.Scatter3d(
            x=n[:, 0], y=n[:, 1], z=n[:, 2],
            mode="markers", marker=dict(size=4, color=color, line=dict(color="white", width=0.5)),
            name=f"{role} ({len(idx)})",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{cloud.plant_id} {cloud.date} — skeleton "
              f"({skel.n_nodes} nodes, {len(skel.edges)} edges)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"  wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    plant_id = sys.argv[1] if len(sys.argv) > 1 else "Tomato03"

    root = Path(__file__).resolve().parents[1] / "data" / "Pheno4D"
    ds = Pheno4D(root)
    files = ds.files(plant_id, annotated_only=True)
    cloud = ds.load(files[-1])
    print(f"{plant_id} {cloud.date}: {cloud.num_points:,} pts")

    skel = extract_plant_skeleton(cloud.xyz, cloud.instance, stem_branches=6, leaf_nodes=6)
    n_leaves = sum(1 for r in skel.node_role if r == "leaf-base")
    print(f"  skeleton: {skel.n_nodes} nodes, {len(skel.edges)} edges, {n_leaves} leaves linked to stem")

    plot_skeleton(cloud, skel, OUT_DIR / f"{plant_id}_{cloud.date}_skeleton.html")


if __name__ == "__main__":
    main()
