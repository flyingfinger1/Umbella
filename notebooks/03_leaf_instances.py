"""03: Isolate leaf instances and inspect one leaf.

Pheno4D's `combined_label` gives each leaf a unique id (>=2). We use that to:
  1. visualize the whole plant with each leaf in its own color
  2. extract one specific leaf and report basic stats (bbox, principal axes,
     length-to-width ratio) — these are the building blocks for skeleton extraction.

Usage:
    .venv/Scripts/python.exe notebooks/03_leaf_instances.py [plant_id] [leaf_id]
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D  # noqa: E402

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def per_leaf_colors(n: int) -> list[str]:
    palette = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
    return [palette[i % len(palette)] for i in range(n)]


def plot_per_instance(cloud, out_html: Path, subsample_per_leaf: int = 4_000) -> None:
    rng = np.random.default_rng(0)
    traces = []

    # stem in dark green
    stem_mask = cloud.instance == 1
    if stem_mask.any():
        s = cloud.xyz[stem_mask]
        if s.shape[0] > 8_000:
            s = s[rng.choice(s.shape[0], 8_000, replace=False)]
        traces.append(go.Scatter3d(
            x=s[:, 0], y=s[:, 1], z=s[:, 2],
            mode="markers", marker=dict(size=1.2, color="#0b3d0b"),
            name="stem",
        ))

    leaf_ids = sorted(int(x) for x in np.unique(cloud.instance) if x >= 2)
    colors = per_leaf_colors(len(leaf_ids))
    for lid, color in zip(leaf_ids, colors):
        mask = cloud.instance == lid
        pts = cloud.xyz[mask]
        if pts.shape[0] > subsample_per_leaf:
            pts = pts[rng.choice(pts.shape[0], subsample_per_leaf, replace=False)]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=dict(size=1.5, color=color),
            name=f"leaf {lid} (n={int(mask.sum())})",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{cloud.plant_id} {cloud.date} — {len(leaf_ids)} leaf instances",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"  wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


def leaf_stats(pts: np.ndarray) -> dict:
    """Bbox + PCA-based extent (proxy for length / width / thickness)."""
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending; flip to descending so [0]=length, [1]=width, [2]=thickness
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # extent along each principal axis (range of projection)
    proj = centered @ eigvecs
    extent = proj.max(axis=0) - proj.min(axis=0)
    return {
        "n_points": int(pts.shape[0]),
        "bbox_min": pts.min(axis=0).tolist(),
        "bbox_max": pts.max(axis=0).tolist(),
        "principal_extent_mm": extent.tolist(),
        "length_width_ratio": float(extent[0] / max(extent[1], 1e-6)),
        "thickness_to_width_ratio": float(extent[2] / max(extent[1], 1e-6)),
    }


def plot_one_leaf(cloud, leaf_id: int, out_html: Path) -> None:
    mask = cloud.instance == leaf_id
    if not mask.any():
        raise ValueError(f"leaf id {leaf_id} not present (available: {sorted(int(x) for x in np.unique(cloud.instance) if x >= 2)})")
    pts = cloud.xyz[mask]
    stats = leaf_stats(pts)
    print(f"\nLeaf {leaf_id} stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # subsample for plot
    rng = np.random.default_rng(0)
    if pts.shape[0] > 30_000:
        pts_plot = pts[rng.choice(pts.shape[0], 30_000, replace=False)]
    else:
        pts_plot = pts

    fig = go.Figure(go.Scatter3d(
        x=pts_plot[:, 0], y=pts_plot[:, 1], z=pts_plot[:, 2],
        mode="markers", marker=dict(size=1.8, color="#7CFC00"),
        name=f"leaf {leaf_id}",
    ))
    fig.update_layout(
        title=f"{cloud.plant_id} {cloud.date} — leaf {leaf_id} "
              f"({stats['n_points']:,} pts, L/W={stats['length_width_ratio']:.2f})",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"  wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    plant_id = sys.argv[1] if len(sys.argv) > 1 else "Tomato03"
    leaf_id_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    root = Path(__file__).resolve().parents[1] / "data" / "Pheno4D"
    ds = Pheno4D(root)

    # take the latest annotated scan (most leaves)
    files = ds.files(plant_id, annotated_only=True)
    cloud = ds.load(files[-1])
    leaf_ids = sorted(int(x) for x in np.unique(cloud.instance) if x >= 2)
    print(f"{plant_id} {cloud.date}: {cloud.num_points:,} pts, {len(leaf_ids)} leaves")
    print(f"  leaf ids: {leaf_ids}")

    plot_per_instance(cloud, OUT_DIR / f"{plant_id}_{cloud.date}_instances.html")

    chosen = leaf_id_arg if leaf_id_arg is not None else leaf_ids[len(leaf_ids) // 2]
    plot_one_leaf(cloud, chosen, OUT_DIR / f"{plant_id}_{cloud.date}_leaf{chosen}.html")


if __name__ == "__main__":
    main()
