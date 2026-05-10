"""29: Prototype the procedural Apiaceae leaf — visual eyeball validation.

Generates one Anthriscus sylvestris leaf (bipinnate) with the new procedural
leaf model, samples a point cloud, renders a 2D image, and writes both an
interactive 3D HTML and the rendered PNG to notebooks/output/ for visual
comparison against reference photos.

This is intentionally a single-leaf, single-species prototype. Once the
output looks plausible we'll extend to all six species and integrate the
leaf branch into the training pipeline.

Usage:
    .venv/Scripts/python.exe notebooks/29_leaf_prototype.py
"""

from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic.leaf import ANTHRISCUS_LEAF, generate_apiaceae_leaf  # noqa: E402
from src.geometry import (                                              # noqa: E402
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    camera_around,
    render_pointcloud,
    role_aware_color_callable,
    role_aware_radius_callable,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def render_skeleton_html(skel, out_path: Path) -> None:
    """3D scatter of skeleton nodes coloured by role, plus edges as line traces."""
    role_colors = {
        "leaf-petiole": "#557d3c",
        "leaf-rachis": "#5f8c46",
        "leaf-pinna-rachis": "#69964b",
        "leaf-midrib": "#73a555",
        "leaf-edge": "#73a555",
    }
    traces = []
    for r, color in role_colors.items():
        mask = np.array([role == r for role in skel.node_role])
        if not mask.any():
            continue
        pts = skel.nodes[mask]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=dict(size=2.5, color=color),
            name=r,
        ))
    # edges
    edge_x, edge_y, edge_z = [], [], []
    for a, b in skel.edges:
        edge_x.extend([skel.nodes[a, 0], skel.nodes[b, 0], None])
        edge_y.extend([skel.nodes[a, 1], skel.nodes[b, 1], None])
        edge_z.extend([skel.nodes[a, 2], skel.nodes[b, 2], None])
    traces.append(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode="lines",
        line=dict(color="rgba(80,110,55,0.4)", width=1),
        name="edges", showlegend=False,
    ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Anthriscus sylvestris — bipinnate leaf prototype<br>"
              f"<sub>{skel.nodes.shape[0]} nodes, {len(skel.edges)} edges</sub>",
        scene=dict(aspectmode="data"),
    )
    fig.write_html(str(out_path))


def main() -> None:
    print("=== 29 — leaf prototype ===\n")

    # 1. generate the leaf skeleton + polygons
    params = ANTHRISCUS_LEAF
    params.seed = 0
    skel, polygons = generate_apiaceae_leaf(params)
    print(f"  generated skeleton: {skel.nodes.shape[0]} nodes, "
          f"{len(skel.edges)} edges, {len(polygons)} leaflet polygons")

    # 2. interactive 3D HTML for inspection
    html_path = OUT_DIR / "leaf_prototype_anthriscus.html"
    render_skeleton_html(skel, html_path)
    print(f"  wrote 3D HTML  -> {html_path.relative_to(Path.cwd())}")

    # 3. point cloud + 2D render (same pipeline as the inflorescence path)
    xyz, lbl = sample_skeleton_pointcloud(skel, points_per_mm2=2.5,
                                          noise_mm=0.15, seed=0)
    print(f"  sampled cloud: {xyz.shape[0]} points")

    target = xyz.mean(0).astype(np.float32)
    bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))

    # role-aware colors so the leaf renders green (uses ROLE_TO_RGB)
    organ_to_role = {}
    for o, r in zip(skel.node_organ, skel.node_role):
        organ_to_role.setdefault(o, r)
    color_fn = role_aware_color_callable(organ_to_role)
    radius_fn = role_aware_radius_callable(organ_to_role)

    # render multiple viewing angles for direct visual comparison against
    # reference photos. 90° elevation = top-down view (matches the typical
    # iNat leaf-on-ground reference shot).
    angles = [
        ("topdown", 0, 89),   # nearly straight down (avoid singularity at exact 90°)
        ("threequarter", 0, 45),
        ("side", 0, 10),
    ]
    rendered = {}
    for name, az, el in angles:
        cam = camera_around(target.astype(np.float64), bbox_diag,
                            azimuth_deg=az, elevation_deg=el,
                            distance_factor=1.4).astype(np.float32)
        visible = hpr_visible_indices(xyz, cam, radius_factor=1000.0)
        rgb, _, _ = render_pointcloud(
            xyz[visible], lbl[visible],
            camera_pos=cam, target=target,
            image_size=(2048, 2048), fov_deg=40.0,
            point_radius_px=2,
            label_to_color=color_fn,
            label_to_radius=radius_fn,
            polygons=polygons,
        )
        rendered[name] = rgb

    # save individual PNGs
    try:
        from PIL import Image
        for name, rgb in rendered.items():
            png_path = OUT_DIR / f"leaf_prototype_anthriscus_{name}.png"
            Image.fromarray(rgb).save(png_path)
            print(f"  wrote PNG ({name:>13s}) -> {png_path.relative_to(Path.cwd())}")
    except ImportError:
        print("  (PIL not installed — skipped raw PNGs)")

    # overlay comparison: synth leaf rotated/scaled onto reference photo
    # so the rachis directions visually align — gives a direct overlap check.
    ref_path = (ROOT / "data" / "leaf_images" / "Anthriscus_sylvestris"
                / "plingfactory_anthriscus.jpg")
    if ref_path.exists():
        try:
            from PIL import Image
            ref_pil = Image.open(ref_path).convert("RGB")
            ref_w, ref_h = ref_pil.size

            # Hand-calibrated alignment to the plingfactory photo.
            # The reference shows the leaf with petiole running from lower-left
            # to roughly the centre of the image, then leaf body fanning to
            # the right and upward. Rachis direction ≈ 25° above horizontal.
            # Synth is rendered with petiole at image-top (axis points +x in
            # world but image-top after the y-flip in render), so we rotate
            # the synth render by -65° to align rachis with the photo diagonal.
            rotate_deg = 131.0
            target_leaf_w_frac = 1.55
            # photo's leaf-body centre, in fractional coords
            center_xy_frac = (0.445, 0.515)

            synth_rgb = rendered["topdown"]
            synth_pil = Image.fromarray(synth_rgb).convert("RGBA")
            # background-key: pixels close to the bg color → fully transparent
            arr = np.array(synth_pil)
            bg = np.array([245, 245, 240])
            dist = np.linalg.norm(arr[..., :3].astype(np.int16) - bg, axis=-1)
            arr[..., 3] = np.where(dist < 25, 0, 255).astype(np.uint8)
            synth_pil = Image.fromarray(arr)

            # resize to match target leaf width on the reference
            target_w = int(ref_w * target_leaf_w_frac)
            scale = target_w / synth_pil.width
            new_size = (target_w, int(synth_pil.height * scale))
            synth_pil = synth_pil.resize(new_size, Image.LANCZOS)
            # rotate around the petiole anchor (top-centre of the synth render
            # since petiole points up). PIL rotates around image centre by
            # default → we shift so petiole is at the rotation centre.
            synth_pil = synth_pil.rotate(rotate_deg, resample=Image.BICUBIC,
                                         expand=True)

            # tint synth to a different shade so overlap is visible
            tinted = np.array(synth_pil)
            mask = tinted[..., 3] > 0
            tinted[mask, 0] = np.minimum(255, tinted[mask, 0] + 40)
            tinted[mask, 1] = np.maximum(0, tinted[mask, 1].astype(np.int16) - 30).astype(np.uint8)
            tinted[mask, 2] = np.maximum(0, tinted[mask, 2].astype(np.int16) - 20).astype(np.uint8)
            tinted[mask, 3] = 165                              # ~65% opacity
            synth_overlay = Image.fromarray(tinted)

            # paste centered on the photo's leaf-body centre
            cx_px = int(ref_w * center_xy_frac[0])
            cy_px = int(ref_h * center_xy_frac[1])
            paste_x = cx_px - synth_overlay.width // 2
            paste_y = cy_px - synth_overlay.height // 2
            base = ref_pil.convert("RGBA")
            base.paste(synth_overlay, (paste_x, paste_y), synth_overlay)
            combo_path = OUT_DIR / "leaf_prototype_anthriscus_overlay.png"
            base.convert("RGB").save(combo_path)
            print(f"  wrote overlay           -> {combo_path.relative_to(Path.cwd())}")

            # also a side-by-side for at-a-glance comparison
            h_target = synth_rgb.shape[0]
            ratio = h_target / ref_h
            ref_resized = np.array(ref_pil.resize(
                (int(ref_w * ratio), h_target), Image.LANCZOS))
            combo = np.concatenate([synth_rgb, ref_resized], axis=1)
            sbs_path = OUT_DIR / "leaf_prototype_anthriscus_compare.png"
            Image.fromarray(combo).save(sbs_path)
            print(f"  wrote side-by-side      -> {sbs_path.relative_to(Path.cwd())}")
        except Exception as e:
            print(f"  (comparison skipped: {e})")


if __name__ == "__main__":
    main()
