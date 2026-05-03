"""02: Pheno4D growth animation — slider through annotated scans of one plant.

Run from project root:
    .venv/Scripts/python.exe notebooks/02_growth_timeseries.py [plant_id]
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D  # noqa: E402
from src.datasets.pheno4d import CLASS_NAMES  # noqa: E402

CLASS_COLORS = {0: "#8B4513", 1: "#228B22", 2: "#7CFC00"}
SUBSAMPLE_PER_CLASS = 15_000  # per class per frame; keeps file size + render snappy
DROP_SOIL = True               # the plant is what we care about

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def cloud_traces(cloud) -> list[go.Scatter3d]:
    rng = np.random.default_rng(42)
    traces = []
    for cls_id, name in CLASS_NAMES.items():
        if DROP_SOIL and cls_id == 0:
            continue
        mask = cloud.semantic == cls_id
        if not mask.any():
            continue
        xyz = cloud.xyz[mask]
        if xyz.shape[0] > SUBSAMPLE_PER_CLASS:
            idx = rng.choice(xyz.shape[0], SUBSAMPLE_PER_CLASS, replace=False)
            xyz = xyz[idx]
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=CLASS_COLORS[cls_id]),
                name=name,
                showlegend=True,
            )
        )
    return traces


def main() -> None:
    plant_id = sys.argv[1] if len(sys.argv) > 1 else "Maize01"

    root = Path(__file__).resolve().parents[1] / "data" / "Pheno4D"
    ds = Pheno4D(root)

    files = ds.files(plant_id, annotated_only=True)
    print(f"{plant_id}: {len(files)} annotated scans")

    clouds = []
    for f in files:
        c = ds.load(f)
        clouds.append(c)
        plant_pts = int((c.semantic != 0).sum()) if c.semantic is not None else c.num_points
        print(f"  {c.date}: {c.num_points:>9,} pts  ({plant_pts:>7,} plant)")

    # global bbox over all frames so the camera doesn't jump while scrubbing
    if DROP_SOIL:
        all_xyz = np.concatenate([c.xyz[c.semantic != 0] for c in clouds], axis=0)
    else:
        all_xyz = np.concatenate([c.xyz for c in clouds], axis=0)
    bbox_min = all_xyz.min(axis=0)
    bbox_max = all_xyz.max(axis=0)

    locked_scene = dict(
        aspectmode="manual",
        aspectratio=dict(
            x=float(bbox_max[0] - bbox_min[0]),
            y=float(bbox_max[1] - bbox_min[1]),
            z=float(bbox_max[2] - bbox_min[2]),
        ),
        xaxis=dict(range=[bbox_min[0], bbox_max[0]], autorange=False),
        yaxis=dict(range=[bbox_min[1], bbox_max[1]], autorange=False),
        zaxis=dict(range=[bbox_min[2], bbox_max[2]], autorange=False),
    )

    frames = []
    for c in clouds:
        frames.append(go.Frame(data=cloud_traces(c), name=c.date, layout=dict(scene=locked_scene)))

    fig = go.Figure(
        data=cloud_traces(clouds[0]),
        frames=frames,
    )

    steps = [
        dict(
            method="animate",
            label=c.date,
            args=[[c.date], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
        )
        for c in clouds
    ]

    fig.update_layout(
        title=f"{plant_id} — growth across {len(clouds)} annotated scans"
              + (" (soil hidden)" if DROP_SOIL else ""),
        scene=locked_scene,
        margin=dict(l=0, r=0, t=40, b=0),
        sliders=[dict(active=0, currentvalue=dict(prefix="date: "), steps=steps)],
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=0.05, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=600, redraw=True), fromcurrent=True)]),
                dict(label="⏸ pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
    )

    out_html = OUT_DIR / f"{plant_id}_growth.html"
    fig.write_html(out_html)
    print(f"\nwrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
