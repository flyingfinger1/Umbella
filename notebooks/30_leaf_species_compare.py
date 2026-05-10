"""30: Render all six Apiaceae species' leaves side-by-side.

Validates that the leaf parameter set differentiates the species visually:
fine-dissected (Daucus, Conium, Anthriscus) vs. broad-leaflet (Heracleum,
Pastinaca) vs. parsley-like (Aethusa).

Each leaf is rendered top-down at the same camera distance — but framed by
its own bbox, so absolute size differences (Heracleum >> Aethusa) get
preserved between tiles via a shared overall scale.

Usage:
    .venv/Scripts/python.exe notebooks/30_leaf_species_compare.py
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic.leaf import SPECIES_LEAVES, generate_apiaceae_leaf  # noqa: E402
from src.geometry import (                                              # noqa: E402
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    camera_around,
    render_pointcloud,
    role_aware_color_callable,
    role_aware_radius_callable,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def render_one(species_key: str, params, image_size: int = 1024,
               common_scale_mm: float | None = None) -> np.ndarray:
    """Render a single leaf top-down. If `common_scale_mm` is given, the
    framing is forced to that physical width — preserving relative size
    between species in the side-by-side comparison."""
    params.seed = 0
    skel, polygons = generate_apiaceae_leaf(params)

    xyz, lbl = sample_skeleton_pointcloud(skel, points_per_mm2=2.5,
                                          noise_mm=0.15, seed=0)
    target = xyz.mean(0).astype(np.float32)
    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))

    organ_to_role = {}
    for o, r in zip(skel.node_organ, skel.node_role):
        organ_to_role.setdefault(o, r)
    color_fn = role_aware_color_callable(organ_to_role)
    radius_fn = role_aware_radius_callable(organ_to_role)

    # if a common scale is requested, fake it via distance_factor scaling
    if common_scale_mm is not None and bbox_diag > 0:
        # framing = common_scale_mm at distance_factor=1 → adjust
        distance_factor = 1.0 * (common_scale_mm / max(bbox_diag, 1.0))
    else:
        distance_factor = 1.4

    cam = camera_around(target.astype(np.float64), bbox_diag,
                        azimuth_deg=0, elevation_deg=89,
                        distance_factor=distance_factor).astype(np.float32)
    visible = hpr_visible_indices(xyz, cam, radius_factor=1000.0)
    rgb, _, _ = render_pointcloud(
        xyz[visible], lbl[visible],
        camera_pos=cam, target=target,
        image_size=(image_size, image_size), fov_deg=40.0,
        point_radius_px=2,
        label_to_color=color_fn,
        label_to_radius=radius_fn,
        polygons=polygons,
    )
    return rgb, bbox_diag


def main() -> None:
    print("=== 30 — six-species leaf comparison ===\n")

    # render each species individually first to discover the largest bbox
    rendered = {}
    bboxes = {}
    for sp, params in SPECIES_LEAVES.items():
        print(f"  rendering {sp} ...")
        rgb, bbox = render_one(sp, params)
        rendered[sp] = rgb
        bboxes[sp] = bbox
        print(f"    bbox diag {bbox:.0f} mm")

    print("\n  re-rendering with common scale to preserve relative sizes ...")
    common = max(bboxes.values()) * 1.1     # 10% padding around the largest leaf
    for sp, params in SPECIES_LEAVES.items():
        rgb, _ = render_one(sp, params, common_scale_mm=common)
        rendered[sp] = rgb

    # 2×3 grid: top row Anthriscus / Conium / Daucus
    #           bottom row Aethusa / Pastinaca / Heracleum
    grid_order = [
        ["Anthriscus_sylvestris", "Conium_maculatum", "Daucus_carota"],
        ["Aethusa_cynapium",      "Pastinaca_sativa", "Heracleum_sphondylium"],
    ]

    # add a label strip at the top of each tile
    try:
        from PIL import Image, ImageDraw, ImageFont
        H = rendered[grid_order[0][0]].shape[0]
        W = rendered[grid_order[0][0]].shape[1]
        label_h = 40
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            font = ImageFont.load_default()
        tiles = []
        for row in grid_order:
            row_imgs = []
            for sp in row:
                tile = Image.new("RGB", (W, H + label_h), (245, 245, 240))
                tile.paste(Image.fromarray(rendered[sp]), (0, label_h))
                draw = ImageDraw.Draw(tile)
                draw.text((10, 10), sp.replace("_", " "), fill=(50, 50, 50),
                          font=font)
                row_imgs.append(np.array(tile))
            tiles.append(np.concatenate(row_imgs, axis=1))
        grid = np.concatenate(tiles, axis=0)
        out_path = OUT_DIR / "leaf_species_compare.png"
        Image.fromarray(grid).save(out_path)
        print(f"\n  wrote grid -> {out_path.relative_to(Path.cwd())}")
    except ImportError:
        # fallback without labels
        rows = []
        for row in grid_order:
            rows.append(np.concatenate([rendered[sp] for sp in row], axis=1))
        grid = np.concatenate(rows, axis=0)
        from PIL import Image
        out_path = OUT_DIR / "leaf_species_compare.png"
        Image.fromarray(grid).save(out_path)
        print(f"\n  wrote grid (no labels) -> {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
