"""01: Explore Pheno4D — load one annotated maize scan, visualize by class.

Run from project root:
    .venv/Scripts/python.exe notebooks/01_explore_pheno4d.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D  # noqa: E402
from src.datasets.pheno4d import CLASS_NAMES  # noqa: E402

CLASS_COLORS = {0: "#8B4513", 1: "#228B22", 2: "#7CFC00"}  # soil brown, stem dark green, leaf light green
SUBSAMPLE = 80_000  # plotly chokes above ~100k points

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def plot_cloud(cloud, out_html: Path) -> None:
    xyz = cloud.xyz
    sem = cloud.semantic
    if xyz.shape[0] > SUBSAMPLE:
        idx = np.random.default_rng(42).choice(xyz.shape[0], SUBSAMPLE, replace=False)
        xyz = xyz[idx]
        sem = sem[idx]

    traces = []
    for cls_id, name in CLASS_NAMES.items():
        mask = sem == cls_id
        if not mask.any():
            continue
        traces.append(
            go.Scatter3d(
                x=xyz[mask, 0], y=xyz[mask, 1], z=xyz[mask, 2],
                mode="markers",
                marker=dict(size=1.2, color=CLASS_COLORS[cls_id]),
                name=f"{name} (n={int(mask.sum())})",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{cloud.plant_id} — date {cloud.date} — {cloud.num_points:,} points",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"  wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "Pheno4D"
    ds = Pheno4D(root)

    print("Plants in dataset:")
    for p in ds.plants():
        n_files = len(ds.files(p))
        n_annot = len(ds.files(p, annotated_only=True))
        print(f"  {p:10s}  {n_files:3d} files  ({n_annot} annotated)")

    target_plant = "Maize01"
    annotated_files = ds.files(target_plant, annotated_only=True)
    print(f"\nLoading first annotated scan of {target_plant}: {annotated_files[0].name}")
    cloud = ds.load(annotated_files[0])
    print(f"  points: {cloud.num_points:,}")
    print(f"  bbox:   min={cloud.xyz.min(axis=0)} max={cloud.xyz.max(axis=0)}")
    if cloud.semantic is not None:
        unique, counts = np.unique(cloud.semantic, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  class {u} ({CLASS_NAMES.get(int(u), '?')}): {c:,} pts")

    out_html = OUT_DIR / f"{target_plant}_{cloud.date}.html"
    plot_cloud(cloud, out_html)


if __name__ == "__main__":
    main()
