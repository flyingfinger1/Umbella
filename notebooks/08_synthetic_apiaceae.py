"""08: Generate synthetic Apiaceae plants and visualize them.

Builds a few distinct species-like presets and renders each as a 3D HTML.
The skeletons go through the same Skeleton schema as Pheno4D, so downstream
code (feature extractor, classifier) treats them identically.

Usage:
    .venv/Scripts/python.exe notebooks/08_synthetic_apiaceae.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import ApiaceaeParams, generate_apiaceae   # noqa: E402

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


# A few coarse species-flavored presets — values are illustrative starting
# points, not field-validated. We'll calibrate later against real photos /
# botanical references.
PRESETS = {
    # giant hogweed-ish: tall, very many primary rays, large umbel
    "Heracleum_like": ApiaceaeParams(
        stem_height_mm=1500, n_internodes=6,
        n_primary_rays=45, primary_ray_length_mm=140, primary_ray_half_angle_deg=55,
        n_pedicels=20, pedicel_length_mm=20, pedicel_half_angle_deg=55,
        n_laterals=2, randomness=0.08, seed=1,
    ),
    # poison hemlock-ish: medium height, fewer rays, more lax umbel
    "Conium_like": ApiaceaeParams(
        stem_height_mm=900, n_internodes=5,
        n_primary_rays=14, primary_ray_length_mm=55, primary_ray_half_angle_deg=60,
        n_pedicels=12, pedicel_length_mm=10, pedicel_half_angle_deg=70,
        n_laterals=3, randomness=0.10, seed=2,
    ),
    # wild carrot-ish: medium, moderate rays, very flat-topped (high cone angle)
    "Daucus_like": ApiaceaeParams(
        stem_height_mm=600, n_internodes=4,
        n_primary_rays=22, primary_ray_length_mm=45, primary_ray_half_angle_deg=70,
        n_pedicels=15, pedicel_length_mm=8, pedicel_half_angle_deg=80,
        n_laterals=1, randomness=0.06, seed=3,
    ),
    # cow parsley-ish: tall, slim, fewer pedicels per umbellet
    "Anthriscus_like": ApiaceaeParams(
        stem_height_mm=1100, n_internodes=6,
        n_primary_rays=10, primary_ray_length_mm=70, primary_ray_half_angle_deg=50,
        n_pedicels=8, pedicel_length_mm=12, pedicel_half_angle_deg=65,
        n_laterals=2, randomness=0.09, seed=4,
    ),
}


ROLE_COLORS = {
    "stem": "#1f3d7a",
    "stem-junction": "#1f3d7a",
    "lateral": "#5a8fcf",
    "ray-base": "#d62728",
    "ray-mid": "#d62728",
    "ray-tip": "#d62728",
    "umbellet-center": "#222222",
    "pedicel-tip": "#7CFC00",
}


def plot_skeleton(skel, title: str, out_html: Path) -> None:
    # group edges by "kind" (stem-stem, ray-ray, pedicel) for legend clarity
    kinds = {"stem": ([], [], []), "ray": ([], [], []), "pedicel": ([], [], []), "lateral": ([], [], [])}

    def kind_of(i: int, j: int) -> str:
        oa, ob = skel.node_organ[i], skel.node_organ[j]
        ra, rb = skel.node_role[i], skel.node_role[j]
        if oa == 1 and ob == 1:
            return "stem"
        if "lateral" in (ra, rb):
            return "lateral"
        if "pedicel-tip" in (ra, rb):
            return "pedicel"
        if oa != ob and (ra == "umbellet-center" or rb == "umbellet-center"):
            return "pedicel"
        return "ray"

    for i, j in skel.edges:
        a, b = skel.nodes[i], skel.nodes[j]
        k = kind_of(i, j)
        xs, ys, zs = kinds[k]
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
        zs += [a[2], b[2], None]

    line_styles = {
        "stem":     dict(color="#1f3d7a", width=8),
        "lateral":  dict(color="#5a8fcf", width=6),
        "ray":      dict(color="#d62728", width=3),
        "pedicel":  dict(color="#7CFC00", width=2),
    }
    traces = []
    for k, (xs, ys, zs) in kinds.items():
        if not xs:
            continue
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=line_styles[k], name=k))

    # node markers per role
    role_marker = {
        "stem":             ("#1f3d7a", 3),
        "stem-junction":    ("#000000", 6),
        "lateral":          ("#5a8fcf", 4),
        "umbellet-center":  ("#222222", 6),
        "pedicel-tip":      ("#7CFC00", 4),
    }
    for r, (color, sz) in role_marker.items():
        idx = [i for i, role in enumerate(skel.node_role) if role == r]
        if not idx:
            continue
        n = skel.nodes[idx]
        traces.append(go.Scatter3d(x=n[:, 0], y=n[:, 1], z=n[:, 2],
                                   mode="markers",
                                   marker=dict(size=sz, color=color),
                                   name=f"{r} ({len(idx)})"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{title} — {skel.n_nodes} nodes, {len(skel.edges)} edges",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)
    print(f"  wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    for name, params in PRESETS.items():
        skel = generate_apiaceae(params)
        n_rays = sum(1 for r in skel.node_role if r == "umbellet-center")
        n_pedicels = sum(1 for r in skel.node_role if r == "pedicel-tip")
        n_laterals = sum(1 for r in skel.node_role if r == "lateral") // 3  # 3 nodes per lateral
        bbox = skel.nodes.max(axis=0) - skel.nodes.min(axis=0)
        print(f"{name}: {skel.n_nodes} nodes, {len(skel.edges)} edges, "
              f"{n_rays} primary rays, {n_pedicels} pedicels total, "
              f"{n_laterals} laterals, height={bbox[2]:.0f}mm")
        plot_skeleton(skel, name, OUT_DIR / f"synthetic_{name}.html")


if __name__ == "__main__":
    main()
