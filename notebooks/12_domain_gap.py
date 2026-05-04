"""12: Quantify the domain gap between real (Pheno4D) and synthetic clouds.

Compares per-organ properties between one Pheno4D scan and one synthetic
cloud of comparable topological complexity. Focus is on properties that
should differ in known ways:

  - Point density (mm-spacing): driven by scanner resolution / our parameter
  - Azimuthal coverage: real LiDAR sees roughly one side of a cylinder
    (a single arc); our synthetic samples full 360° on the cylinder
  - Per-organ point-count proportions: real has dominant soil/stem ratios,
    synthetic has none of either
  - Bounding-box scale: just to confirm both are mm-units

Usage:
    .venv/Scripts/python.exe notebooks/12_domain_gap.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import Pheno4D                          # noqa: E402
from src.synthetic import SPECIES, generate_apiaceae      # noqa: E402
from src.geometry import sample_skeleton_pointcloud       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "output"
OUT_DIR.mkdir(exist_ok=True)


def organ_stats(xyz: np.ndarray, max_sample: int = 5000) -> dict:
    """Resolution + radial/azimuthal signature of an organ point set."""
    n = xyz.shape[0]
    if n < 10:
        return dict(n=n)
    bbox = xyz.max(0) - xyz.min(0)

    # mean nearest-neighbor distance (k=2 because k=1 returns self)
    sample = xyz if n <= max_sample else xyz[
        np.random.default_rng(0).choice(n, max_sample, replace=False)
    ]
    tree = cKDTree(xyz)
    d, _ = tree.query(sample, k=2)
    nn = d[:, 1]

    # PCA axis from a downsample, then radial+azimuthal in orthonormal frame
    centered = xyz - xyz.mean(0)
    sub = centered if n <= 2000 else centered[
        np.random.default_rng(1).choice(n, 2000, replace=False)
    ]
    _, _, vh = np.linalg.svd(sub, full_matrices=False)
    axis = vh[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper); e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(axis, e1)
    proj_axis = centered @ axis
    proj_e1 = centered @ e1
    proj_e2 = centered @ e2
    radial = np.sqrt(proj_e1**2 + proj_e2**2)
    theta = np.arctan2(proj_e2, proj_e1)  # in [-pi, pi]

    # azimuthal uniformity: std of normalized histogram (high = peaky/one-sided)
    bins = 24
    hist, _ = np.histogram(theta, bins=bins, range=(-np.pi, np.pi))
    p = hist / max(hist.sum(), 1)
    uniform_p = 1.0 / bins
    azimuthal_nonuniformity = float(np.sqrt(((p - uniform_p) ** 2).mean()) / uniform_p)

    return dict(
        n=n,
        bbox_mm=[float(b) for b in bbox],
        bbox_volume_cm3=float(bbox[0] * bbox[1] * bbox[2] / 1000.0),
        density_per_cm3=float(n / max(bbox[0] * bbox[1] * bbox[2] / 1000.0, 1e-9)),
        mean_nn_mm=float(nn.mean()),
        median_nn_mm=float(np.median(nn)),
        median_radial_mm=float(np.median(radial)),
        azimuthal_nonuniformity=azimuthal_nonuniformity,  # 0 = perfectly uniform; ~3+ = single-arc
        theta=theta, radial=radial,
    )


def main() -> None:
    # --- real plant: a Pheno4D Tomato (large, branched, has many leaves) ---
    ds = Pheno4D(ROOT / "data" / "Pheno4D")
    cloud = ds.load(ds.files("Tomato03", annotated_only=True)[-1])
    real_xyz = cloud.xyz
    real_inst = cloud.instance
    print(f"REAL: Tomato03_{cloud.date}: {real_xyz.shape[0]:,} pts")

    # isolate one stem cloud and one leaf cloud
    real_stem = real_xyz[real_inst == 1]
    leaf_ids = sorted(int(x) for x in np.unique(real_inst) if x >= 2)
    chosen_leaf_id = leaf_ids[len(leaf_ids) // 2]
    real_leaf = real_xyz[real_inst == chosen_leaf_id]
    print(f"  stem: {real_stem.shape[0]:,} pts, leaf {chosen_leaf_id}: {real_leaf.shape[0]:,} pts")

    # --- synthetic plant: Heracleum (large, branched, similar bbox order) ---
    spec = SPECIES["Heracleum_sphondylium"]
    skel = generate_apiaceae(spec.sample(seed=0))
    synth_xyz, synth_label = sample_skeleton_pointcloud(skel, points_per_mm2=0.5, noise_mm=0.3, seed=0)
    print(f"\nSYNTH: Heracleum seed 0: {synth_xyz.shape[0]:,} pts")

    synth_stem = synth_xyz[synth_label == 1]
    other_labels = sorted(int(x) for x in np.unique(synth_label) if x >= 2)
    # pick the largest non-stem organ as the synthetic counterpart to a leaf
    counts = {l: int((synth_label == l).sum()) for l in other_labels}
    chosen_organ = max(counts, key=counts.get)
    synth_organ = synth_xyz[synth_label == chosen_organ]
    print(f"  stem: {synth_stem.shape[0]:,} pts, "
          f"largest other organ ({chosen_organ}): {synth_organ.shape[0]:,} pts")

    # --- compute stats ---
    pairs = [
        ("REAL stem (Tomato03)", real_stem),
        ("SYNTH stem (Heracleum)", synth_stem),
        (f"REAL leaf {chosen_leaf_id}", real_leaf),
        (f"SYNTH organ {chosen_organ}", synth_organ),
    ]
    stats = []
    for name, pts in pairs:
        s = organ_stats(pts)
        stats.append((name, s))
    print()
    print(f"{'name':>30s}  {'pts':>7s}  {'bbox z mm':>9s}  {'NN mm':>6s}  {'radial mm':>9s}  {'azim nonuniform':>15s}")
    for name, s in stats:
        print(f"{name:>30s}  {s['n']:>7,}  {s['bbox_mm'][2]:>9.0f}  "
              f"{s['mean_nn_mm']:>6.2f}  {s['median_radial_mm']:>9.2f}  "
              f"{s['azimuthal_nonuniformity']:>15.2f}")

    # whole-plant: per-organ point fractions
    print("\nReal whole-plant organ point fractions:")
    real_class_counts = {0: int((real_inst == 0).sum()), 1: int((real_inst == 1).sum()),
                        2: int((real_inst >= 2).sum())}
    rt = real_xyz.shape[0]
    for k, name in zip([0, 1, 2], ["soil", "stem", "leaf"]):
        print(f"  {name:>10s}: {real_class_counts[k]:>10,}  ({100 * real_class_counts[k] / rt:.1f}%)")
    print("\nSynthetic whole-plant organ point fractions:")
    synth_n_stem = int((synth_label == 1).sum())
    synth_n_other = int((synth_label >= 2).sum())
    sn = synth_xyz.shape[0]
    print(f"  {'stem':>10s}: {synth_n_stem:>10,}  ({100 * synth_n_stem / sn:.1f}%)")
    print(f"  {'organs':>10s}: {synth_n_other:>10,}  ({100 * synth_n_other / sn:.1f}%)")

    # --- visualization: azimuthal histograms side by side ---
    fig = go.Figure()
    bins = 24
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for name, s in stats:
        if "n" in s and s["n"] > 10:
            hist, _ = np.histogram(s["theta"], bins=edges)
            fig.add_trace(go.Bar(
                x=centers, y=hist / hist.sum(),
                name=name, opacity=0.6,
            ))
    fig.update_layout(
        title="Azimuthal point distribution around organ axis (PCA frame)",
        xaxis_title="theta (rad)", yaxis_title="point fraction",
        barmode="overlay", template="plotly_white",
    )
    out = OUT_DIR / "domain_gap_azimuthal.html"
    fig.write_html(out)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
