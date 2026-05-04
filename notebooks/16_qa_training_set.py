"""16: Sanity-check the training dataset.

  - aggregate stats per species (foreground pixels, label diversity, skeleton sizes)
  - random-sample visualization: 2 examples per species × (RGB, label, depth)
  - random-sample skeleton sanity (load JSON + report node/edge counts)

Catches bugs that would otherwise corrupt later training: degenerate renders,
species with biased camera angles, labels that don't match, etc.

Usage:
    .venv/Scripts/python.exe notebooks/16_qa_training_set.py
"""

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import list_examples, load_example      # noqa: E402
from src.geometry import Skeleton                          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training" / "v1"
OUT_DIR = ROOT / "notebooks" / "output"
OUT_DIR.mkdir(exist_ok=True)

N_SAMPLES_PER_SPECIES = 2


def label_map_rgb(label_map: np.ndarray) -> np.ndarray:
    H, W = label_map.shape
    rng = np.random.default_rng(0)
    palette = np.vstack([
        np.array([[240, 240, 240], [40, 60, 100]], dtype=np.uint8),
        rng.integers(40, 230, size=(2048, 3)).astype(np.uint8),
    ])
    return palette[label_map % palette.shape[0]]


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    finite = depth > 0
    if not finite.any():
        return np.full((*depth.shape, 3), 240, dtype=np.uint8)
    d = depth.copy()
    lo, hi = float(np.percentile(d[finite], 2)), float(np.percentile(d[finite], 98))
    d_norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (255 * (1.0 - d_norm)).astype(np.uint8)
    rgb = np.stack([g, g, g], axis=-1)
    rgb[~finite] = (240, 240, 240)
    return rgb


def main() -> None:
    examples = list_examples(DATA)
    print(f"loaded metadata: {len(examples)} examples\n")

    # group by species
    by_species: dict[str, list] = defaultdict(list)
    for e in examples:
        by_species[e["species"]].append(e)

    # aggregate stats — sample to keep it fast
    print(f"{'species':>22s}  {'mean fg_px':>10s}  {'std fg_px':>10s}  "
          f"{'mean labels':>11s}  {'n_nodes mean':>12s}  {'n_nodes std':>11s}")
    skel_cache: dict[str, Skeleton] = {}
    for species, exs in by_species.items():
        sample = exs if len(exs) <= 40 else [exs[i] for i in
                 np.random.default_rng(0).choice(len(exs), 40, replace=False)]
        fg_px = []
        n_uniq = []
        n_nodes = []
        for e in sample:
            d = load_example(DATA, e)
            fg_px.append(int((d["depth"] > 0).sum()))
            n_uniq.append(len(np.unique(d["label"])))
            sp = str(d["skeleton_path"])
            if sp not in skel_cache:
                skel_cache[sp] = Skeleton.load_json(d["skeleton_path"])
            n_nodes.append(skel_cache[sp].n_nodes)
        print(f"{species:>22s}  {np.mean(fg_px):>10.0f}  {np.std(fg_px):>10.0f}  "
              f"{np.mean(n_uniq):>11.1f}  {np.mean(n_nodes):>12.0f}  {np.std(n_nodes):>11.0f}")

    # check: per-species camera azimuth coverage (should be {0, 90, 180, 270} uniformly)
    print("\nazimuth coverage:")
    for species, exs in by_species.items():
        az_counts = defaultdict(int)
        for e in exs:
            az_counts[float(e["azimuth_deg"])] += 1
        spread = ", ".join(f"{int(a)}°:{c}" for a, c in sorted(az_counts.items()))
        print(f"  {species:>22s}  {spread}")

    # visualization: pick N_SAMPLES_PER_SPECIES random examples per species
    rng = np.random.default_rng(42)
    rows_data = []
    for species, exs in by_species.items():
        idx = rng.choice(len(exs), N_SAMPLES_PER_SPECIES, replace=False)
        for i in idx:
            rows_data.append((species, exs[int(i)]))

    n_rows = len(rows_data)
    titles = []
    for species, e in rows_data:
        tag = f"{species[:10]} seed{e['seed']:02d} v{e['view']}"
        titles += [f"{tag} — RGB", f"{tag} — labels", f"{tag} — depth"]
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.02, vertical_spacing=0.015,
    )

    for r, (species, e) in enumerate(rows_data, start=1):
        d = load_example(DATA, e)
        fig.add_trace(go.Image(z=d["rgb"]), row=r, col=1)
        fig.add_trace(go.Image(z=label_map_rgb(d["label"])), row=r, col=2)
        fig.add_trace(go.Image(z=depth_to_rgb(d["depth"])), row=r, col=3)
    for r in range(1, n_rows + 1):
        for c in range(1, 4):
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
    fig.update_layout(
        title="Training set v1 — sanity check",
        height=260 * n_rows, width=1100,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    out = OUT_DIR / "training_set_qa.html"
    fig.write_html(out)
    print(f"\nwrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
