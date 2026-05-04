"""06: Build a skeleton corpus from all annotated Pheno4D scans.

Iterates every annotated scan, extracts the plant skeleton, and writes one
JSON per scan to data/skeletons/<plant_id>/<date>.json. Reports per-scan
timing + summary stats at the end.

Usage:
    .venv/Scripts/python.exe notebooks/06_build_skeleton_corpus.py
"""

from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D                         # noqa: E402
from src.geometry import extract_plant_skeleton, Skeleton  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "Pheno4D"
OUT_ROOT = ROOT / "data" / "skeletons"

# tomato is much bigger and has many leaves -> use slightly larger budget
PARAMS_PER_SPECIES = {
    "Maize":  dict(stem_branches=2, leaf_nodes=8),
    "Tomato": dict(stem_branches=8, leaf_nodes=6),
}


def species_of(plant_id: str) -> str:
    return "Maize" if plant_id.startswith("Maize") else "Tomato"


def main() -> None:
    ds = Pheno4D(DATA_ROOT)
    plants = ds.plants()
    total_files = sum(len(ds.files(p, annotated_only=True)) for p in plants)
    print(f"plants: {len(plants)}, annotated scans total: {total_files}\n")

    summary = []  # (plant_id, date, n_pts, n_nodes, n_edges, n_leaves, n_junctions, dt)
    t_global = time.time()

    for plant_id in plants:
        files = ds.files(plant_id, annotated_only=True)
        params = PARAMS_PER_SPECIES[species_of(plant_id)]
        for f in files:
            t0 = time.time()
            cloud = ds.load(f)
            skel = extract_plant_skeleton(cloud.xyz, cloud.instance, **params)
            skel.metadata = {
                "plant_id": cloud.plant_id,
                "date": cloud.date,
                "source_file": f"{cloud.plant_id}/{f.name}",
                "num_points_orig": int(cloud.num_points),
                "extractor_params": params,
            }
            out = OUT_ROOT / cloud.plant_id / f"{cloud.date}.json"
            skel.save_json(out)
            dt = time.time() - t0

            n_leaves = sum(1 for r in skel.node_role if r == "leaf-base")
            n_junc = sum(1 for r in skel.node_role if r == "stem-junction")
            print(f"  {cloud.plant_id} {cloud.date}: {cloud.num_points:>9,} pts -> "
                  f"{skel.n_nodes:>4d} nodes, {len(skel.edges):>4d} edges, "
                  f"{n_leaves:>2d} leaves, {n_junc:>2d} junctions  ({dt:.1f}s)")
            summary.append((cloud.plant_id, cloud.date, cloud.num_points,
                            skel.n_nodes, len(skel.edges), n_leaves, n_junc, dt))

    # summary
    print(f"\n=== Done in {time.time() - t_global:.1f}s ===")
    n_files = len(summary)
    total_nodes = sum(s[3] for s in summary)
    total_leaves = sum(s[5] for s in summary)
    total_junc = sum(s[6] for s in summary)
    total_pts = sum(s[2] for s in summary)
    print(f"  {n_files} skeletons written to {OUT_ROOT.relative_to(ROOT)}/")
    print(f"  total points processed:  {total_pts:>15,}")
    print(f"  total skeleton nodes:    {total_nodes:>15,}  ({total_pts / max(total_nodes, 1):,.0f}x reduction)")
    print(f"  total leaves identified: {total_leaves:>15,}")
    print(f"  total junctions added:   {total_junc:>15,}")

    # disk size
    sizes = [p.stat().st_size for p in OUT_ROOT.rglob("*.json")]
    print(f"  total disk size:         {sum(sizes) / 1024:>11,.1f} KB across {len(sizes)} files")


if __name__ == "__main__":
    main()
