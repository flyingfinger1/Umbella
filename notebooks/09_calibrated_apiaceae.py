"""09: Generate plants from calibrated species specs and visualize.

Each species spec defines empirical parameter ranges (see src/synthetic/species.py).
We sample 3 instances per species (with different seeds) so you can see the
species' shape variability.

Usage:
    .venv/Scripts/python.exe notebooks/09_calibrated_apiaceae.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SPECIES, generate_apiaceae   # noqa: E402

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


N_INSTANCES = 3


def plot_one(skel, title: str, out_html: Path) -> None:
    """Render one skeleton to its own HTML — robust against plotly subplot quirks."""
    stem_x, stem_y, stem_z = [], [], []
    ray_x, ray_y, ray_z = [], [], []
    ped_x, ped_y, ped_z = [], [], []
    lat_x, lat_y, lat_z = [], [], []
    bract_x, bract_y, bract_z = [], [], []
    bracteole_x, bracteole_y, bracteole_z = [], [], []
    for i, j in skel.edges:
        a, b = skel.nodes[i], skel.nodes[j]
        ra, rb = skel.node_role[i], skel.node_role[j]
        oa, ob = skel.node_organ[i], skel.node_organ[j]
        if "bract" in (ra, rb):
            bract_x += [a[0], b[0], None]; bract_y += [a[1], b[1], None]; bract_z += [a[2], b[2], None]
        elif "bracteole" in (ra, rb):
            bracteole_x += [a[0], b[0], None]; bracteole_y += [a[1], b[1], None]; bracteole_z += [a[2], b[2], None]
        elif oa == 1 and ob == 1:
            stem_x += [a[0], b[0], None]; stem_y += [a[1], b[1], None]; stem_z += [a[2], b[2], None]
        elif "lateral" in (ra, rb):
            lat_x += [a[0], b[0], None]; lat_y += [a[1], b[1], None]; lat_z += [a[2], b[2], None]
        elif "pedicel-tip" in (ra, rb):
            ped_x += [a[0], b[0], None]; ped_y += [a[1], b[1], None]; ped_z += [a[2], b[2], None]
        else:
            ray_x += [a[0], b[0], None]; ray_y += [a[1], b[1], None]; ray_z += [a[2], b[2], None]

    traces = []
    for xs, ys, zs, color, width, name in [
        (stem_x, stem_y, stem_z, "#1f3d7a", 6, "stem"),
        (lat_x, lat_y, lat_z, "#5a8fcf", 4, "lateral"),
        (ray_x, ray_y, ray_z, "#d62728", 2, "ray"),
        (ped_x, ped_y, ped_z, "#3aaf3a", 1, "pedicel"),
        (bract_x, bract_y, bract_z, "#a06000", 4, "bract (involucre)"),
        (bracteole_x, bracteole_y, bracteole_z, "#d97a00", 3, "bracteole (involucel)"),
    ]:
        if xs:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=color, width=width), name=name,
            ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{title} — {skel.n_nodes} nodes",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html)


def main() -> None:
    species_dir = OUT_DIR / "calibrated"
    species_dir.mkdir(exist_ok=True)
    for key, spec in SPECIES.items():
        n_rays_list, heights, n_nodes_list = [], [], []
        for seed in range(N_INSTANCES):
            params = spec.sample(seed=seed)
            skel = generate_apiaceae(params)
            skel.metadata["species"] = key
            skel.metadata["german_name"] = spec.german_name
            out = species_dir / f"{key}_seed{seed}.html"
            plot_one(skel, f"{spec.german_name} (seed {seed})", out)
            n_rays_list.append(sum(1 for r in skel.node_role if r == "umbellet-center"))
            heights.append(float(skel.nodes[:, 2].max() - skel.nodes[:, 2].min()))
            n_nodes_list.append(skel.n_nodes)
        print(f"{spec.german_name:>22s} ({key}):")
        print(f"  rays/instance: {n_rays_list}")
        print(f"  height_mm:     {[f'{h:.0f}' for h in heights]}")
        print(f"  n_nodes:       {n_nodes_list}")
    print(f"\nwrote {N_INSTANCES * len(SPECIES)} HTMLs to {species_dir.relative_to(Path(__file__).resolve().parents[1])}/")


if __name__ == "__main__":
    main()
