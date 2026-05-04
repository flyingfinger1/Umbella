"""05: Diagnostic — show one leaf's full point cloud + its skeleton polyline.

Goal: figure out whether leaf-base / leaf-mid markers sit far from the leaf
because of plot subsampling (notebook 04 plots only 1% of points), because the
Pheno4D annotation stops short of the stem, or because the polyline algorithm
degenerates for small leaves.

Usage:
    .venv/Scripts/python.exe notebooks/05_skeleton_probe.py [plant_id] [leaf_id]
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D                      # noqa: E402
from src.geometry import extract_polyline             # noqa: E402

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    plant_id = sys.argv[1] if len(sys.argv) > 1 else "Tomato03"
    leaf_id_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    root = Path(__file__).resolve().parents[1] / "data" / "Pheno4D"
    ds = Pheno4D(root)
    cloud = ds.load(ds.files(plant_id, annotated_only=True)[-1])

    leaf_ids = sorted(int(x) for x in np.unique(cloud.instance) if x >= 2)
    chosen = leaf_id_arg if leaf_id_arg is not None else leaf_ids[len(leaf_ids) // 2]

    leaf_pts = cloud.xyz[cloud.instance == chosen]
    stem_pts = cloud.xyz[cloud.instance == 1]
    print(f"{plant_id} {cloud.date} — leaf {chosen}: {leaf_pts.shape[0]:,} points")

    # full polyline using the production extractor (anchored on stem)
    poly = extract_polyline(leaf_pts, n_nodes=6, anchor_to=stem_pts)
    print("polyline nodes:")
    for i, p in enumerate(poly):
        d_to_leaf = np.linalg.norm(leaf_pts - p, axis=1).min()
        print(f"  node {i}: xyz={p.tolist()}  min_dist_to_leaf_pt={d_to_leaf:.3f}")

    # also check: how close is leaf-base node to nearest stem point?
    base = poly[0]
    d_base_to_stem = np.linalg.norm(stem_pts - base, axis=1).min()
    print(f"leaf-base -> nearest stem point distance: {d_base_to_stem:.3f}")

    # nearest stem point to the leaf cloud as a whole
    if leaf_pts.shape[0] > 5000:
        idx = np.random.default_rng(0).choice(leaf_pts.shape[0], 5000, replace=False)
        leaf_sample = leaf_pts[idx]
    else:
        leaf_sample = leaf_pts
    if stem_pts.shape[0] > 20_000:
        idx = np.random.default_rng(0).choice(stem_pts.shape[0], 20_000, replace=False)
        stem_sample = stem_pts[idx]
    else:
        stem_sample = stem_pts
    d_pairwise = np.linalg.norm(
        leaf_sample[:, None, :] - stem_sample[None, :, :], axis=2
    )
    leaf_to_stem_min = d_pairwise.min()
    print(f"closest distance leaf<->stem (sampled): {leaf_to_stem_min:.3f}")

    # render: full leaf points (no subsampling), polyline, plus nearby stem
    # nearby stem = stem points within bbox of leaf, padded
    pad = 30.0
    lo = leaf_pts.min(axis=0) - pad
    hi = leaf_pts.max(axis=0) + pad
    near_mask = np.all((stem_pts >= lo) & (stem_pts <= hi), axis=1)
    near_stem = stem_pts[near_mask]

    traces = [
        go.Scatter3d(
            x=leaf_pts[:, 0], y=leaf_pts[:, 1], z=leaf_pts[:, 2],
            mode="markers", marker=dict(size=1.0, color="#7CFC00", opacity=0.5),
            name=f"leaf {chosen} pts ({leaf_pts.shape[0]:,})",
        ),
        go.Scatter3d(
            x=near_stem[:, 0], y=near_stem[:, 1], z=near_stem[:, 2],
            mode="markers", marker=dict(size=1.0, color="#1f3d7a", opacity=0.4),
            name=f"nearby stem pts ({near_stem.shape[0]:,})",
        ),
        go.Scatter3d(
            x=poly[:, 0], y=poly[:, 1], z=poly[:, 2],
            mode="lines+markers",
            line=dict(color="#d62728", width=6),
            marker=dict(size=6, color="#d62728"),
            name="polyline",
        ),
        go.Scatter3d(
            x=[poly[0, 0]], y=[poly[0, 1]], z=[poly[0, 2]],
            mode="markers", marker=dict(size=10, color="#000000", symbol="diamond"),
            name="leaf-base (node 0)",
        ),
        go.Scatter3d(
            x=[poly[-1, 0]], y=[poly[-1, 1]], z=[poly[-1, 2]],
            mode="markers", marker=dict(size=10, color="#ffaa00", symbol="diamond"),
            name="leaf-tip (node 5)",
        ),
    ]

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{plant_id} {cloud.date} — leaf {chosen} probe (FULL points, no subsampling)",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    out_html = OUT_DIR / f"{plant_id}_{cloud.date}_leaf{chosen}_probe.html"
    fig.write_html(out_html)
    print(f"\nwrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
