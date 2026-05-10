"""31: Interactive Apiaceae-leaf parameter tweaker.

Spins up a local Gradio app at http://127.0.0.1:7860/. Sliders on the right,
live overlay render against the reference photo on the left. Tuned for fast
visual calibration of LeafParams against a real specimen photo.

Usage:
    .venv/Scripts/python.exe notebooks/31_leaf_tweaker.py
"""

from pathlib import Path
import sys
import json
from dataclasses import asdict

import numpy as np
import gradio as gr
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic.leaf import ANTHRISCUS_LEAF, LeafParams, generate_apiaceae_leaf  # noqa: E402
from src.geometry import (                                              # noqa: E402
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    camera_around,
    render_pointcloud,
    role_aware_color_callable,
    role_aware_radius_callable,
)

ROOT = Path(__file__).resolve().parents[1]
# Drop a CC-licensed top-down reference photo here (Wikimedia Commons,
# iNat CC0/CC-BY, etc.). Falls die Datei nicht existiert, läuft die App
# trotzdem — der Overlay-Modus fällt auf Synth-only zurück.
REF_PATH = (ROOT / "data" / "leaf_images" / "Anthriscus_sylvestris"
            / "_tweaker_reference.jpg")
STATE_PATH = Path(__file__).parent / "output" / "leaf_tweaker_state.json"
STATE_PATH.parent.mkdir(exist_ok=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(*values) -> str:
    """Persist all slider values keyed by their internal name. Gradio passes
    each slider as a positional arg, so we capture them via ``*values``."""
    n_leaf = len(SLIDER_FIELDS)
    state = {}
    for (label, _mn, _mx, _st, name), v in zip(SLIDER_FIELDS, values[:n_leaf]):
        state[name] = float(v)
    for (label, _mn, _mx, _st, _d), v in zip(OVERLAY_FIELDS, values[n_leaf:]):
        state[f"_overlay::{label}"] = float(v)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return f"Saved -> {STATE_PATH.relative_to(ROOT)}"


def render_synth(params: LeafParams, image_size: int = 1024
                 ) -> tuple[np.ndarray, float]:
    """Returns (synth_rgb, render_bbox_diag_mm). The bbox is *fixed* (not
    derived from the actual leaf) so the camera distance and thus mm-per-
    pixel stay constant regardless of geometry params. That way every
    slider's mm effect translates 1:1 to overlay pixels — change a pinna
    length and the pinna actually grows/shrinks on the photo without
    re-zooming everything else."""
    skel, polygons = generate_apiaceae_leaf(params)
    xyz, lbl = sample_skeleton_pointcloud(skel, points_per_mm2=2.5,
                                          noise_mm=0.15, seed=0)
    target = xyz.mean(0).astype(np.float32)
    # Locked bbox tuned so a typical Anthriscus-class leaf (~500 mm)
    # fills ≈70% of the 1024×1024 render → good resolution AND headroom
    # for slightly larger geometries before clipping.
    bbox_diag = 700.0
    organ_to_role: dict[int, str] = {}
    for o, r in zip(skel.node_organ, skel.node_role):
        organ_to_role.setdefault(o, r)
    color_fn = role_aware_color_callable(organ_to_role)
    radius_fn = role_aware_radius_callable(organ_to_role)
    cam = camera_around(target.astype(np.float64), bbox_diag,
                        azimuth_deg=0, elevation_deg=89,
                        distance_factor=1.4).astype(np.float32)
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


def make_overlay(synth_rgb: np.ndarray, synth_bbox_diag: float,
                 rotate_deg: float, scale_frac: float,
                 cx: float, cy: float, alpha: int) -> Image.Image:
    if not REF_PATH.exists():
        return Image.fromarray(synth_rgb)
    ref_pil = Image.open(REF_PATH).convert("RGB")
    ref_w, ref_h = ref_pil.size

    # background-key the synth render
    synth_pil = Image.fromarray(synth_rgb).convert("RGBA")
    arr = np.array(synth_pil)
    bg = np.array([245, 245, 240])
    dist = np.linalg.norm(arr[..., :3].astype(np.int16) - bg, axis=-1)
    arr[..., 3] = np.where(dist < 25, 0, 255).astype(np.uint8)
    synth_pil = Image.fromarray(arr)

    # synth was rendered at a locked bbox (constant mm-per-pixel); just
    # scale to `scale_frac` of the photo width. Geometry changes show
    # their true mm-size differences relative to the photo.
    target_w = max(50, int(ref_w * scale_frac))
    s = target_w / synth_pil.width
    synth_pil = synth_pil.resize((target_w, int(synth_pil.height * s)),
                                 Image.LANCZOS)
    # rotate
    synth_pil = synth_pil.rotate(rotate_deg, resample=Image.BICUBIC,
                                 expand=True)
    # tint + alpha
    tinted = np.array(synth_pil)
    mask = tinted[..., 3] > 0
    tinted[mask, 0] = np.minimum(255, tinted[mask, 0].astype(np.int16) + 40).astype(np.uint8)
    tinted[mask, 1] = np.maximum(0, tinted[mask, 1].astype(np.int16) - 30).astype(np.uint8)
    tinted[mask, 2] = np.maximum(0, tinted[mask, 2].astype(np.int16) - 20).astype(np.uint8)
    tinted[mask, 3] = int(alpha)
    synth_overlay = Image.fromarray(tinted)
    cx_px = int(ref_w * cx)
    cy_px = int(ref_h * cy)
    paste_x = cx_px - synth_overlay.width // 2
    paste_y = cy_px - synth_overlay.height // 2
    base = ref_pil.convert("RGBA")
    base.paste(synth_overlay, (paste_x, paste_y), synth_overlay)
    return base.convert("RGB")


# Each slider corresponds to one field. Order MUST match the function signature.
SLIDER_FIELDS = [
    # (label, min, max, step, attr-name)
    ("Petiole length (mm)",       20.0, 200.0, 5.0, "petiole_length_mm"),
    ("Rachis length (mm)",       100.0, 1200.0, 10.0, "rachis_length_mm"),

    ("# pinna pairs",              1, 12, 1, "n_pinna_pairs"),
    ("Pinna angle base (°)",       10.0, 90.0, 1.0, "_pinna_angle_at_base"),
    ("Pinna angle tip (°)",        10.0, 90.0, 1.0, "_pinna_angle_at_tip"),
    ("Pinna length base (mm)",      5.0, 600.0, 5.0, "pinna_length_at_base"),
    ("Pinna length tip (mm)",       5.0, 400.0, 5.0, "pinna_length_at_tip"),
    ("Pinna spacing power",        0.2, 2.5, 0.05, "pinna_spacing_power"),
    ("Pinna petiolule frac",       0.0, 0.5, 0.01, "pinna_petiolule_frac"),

    ("Recursion depth",             0, 3, 1, "pinna_recursion_depth"),

    ("# pinnule pairs",             1, 10, 1, "n_pinnule_pairs"),
    ("Pinnule angle base (°)",     10.0, 90.0, 1.0, "_pinnule_angle_at_base"),
    ("Pinnule angle tip (°)",      10.0, 90.0, 1.0, "_pinnule_angle_at_tip"),
    ("Pinnule length base (mm)",    2.0, 200.0, 1.0, "pinnule_length_at_base"),
    ("Pinnule length tip (mm)",     1.0, 100.0, 0.5, "pinnule_length_at_tip"),
    ("Pinnule petiolule base",     0.0, 0.5, 0.01, "pinnule_petiolule_frac"),
    ("Pinnule petiolule tip",      0.0, 0.5, 0.01, "_pinnule_petiolule_frac_at_tip"),
    ("Pinnule spacing power",      0.2, 2.5, 0.05, "pinnule_spacing_power"),

    ("Recursive length factor",    0.3, 1.2, 0.05, "recursive_length_factor"),
    ("Pinnule scale with pinna",   0.0, 1.0, 0.05, "pinnule_scale_with_pinna"),
    ("Terminal leaflet scale",     0.5, 2.0, 0.05, "terminal_leaflet_scale"),

    ("Leaflet width (mm)",         0.3, 50.0, 0.2, "leaflet_width_mm"),
    ("Leaflet outline power",      0.3, 4.0, 0.05, "leaflet_outline_power"),
    ("Leaflet peak t",             0.05, 0.95, 0.02, "leaflet_peak_t"),
    ("Leaflet serration",          0.0, 0.6, 0.02, "leaflet_serration"),
    ("Leaflet serration periods",    1, 12, 1, "leaflet_serration_periods"),

    ("Randomness",                 0.0, 0.3, 0.01, "randomness"),
    ("Angle jitter (°)",           0.0, 20.0, 0.5, "angle_jitter_deg"),
]

OVERLAY_FIELDS = [
    ("Overlay rotation (°)",     -180.0, 180.0, 1.0, 131.0),
    ("Overlay scale (× ref width)",  0.3, 2.5, 0.02, 1.55),
    ("Overlay center X",             0.0, 1.0, 0.005, 0.445),
    ("Overlay center Y",             0.0, 1.0, 0.005, 0.515),
    ("Overlay alpha (0-255)",        0, 255, 5, 165),
]


_STATE = load_state()


def field_default(name: str):
    """Read default value: persisted state if present, else ANTHRISCUS_LEAF.
    Fields prefixed with '_' are Optional → coerce None → numeric default the
    slider can display."""
    if name in _STATE:
        return _STATE[name]
    raw_name = name.lstrip("_")
    val = getattr(ANTHRISCUS_LEAF, raw_name, None)
    if val is None:
        # for optional angle/petiolule fields fall back to fixed-value field
        if raw_name == "pinna_angle_at_base":
            return getattr(ANTHRISCUS_LEAF, "pinna_angle_deg", 60.0)
        if raw_name == "pinna_angle_at_tip":
            return getattr(ANTHRISCUS_LEAF, "pinna_angle_deg", 60.0)
        if raw_name == "pinnule_angle_at_base":
            return getattr(ANTHRISCUS_LEAF, "pinnule_angle_deg", 60.0)
        if raw_name == "pinnule_angle_at_tip":
            return getattr(ANTHRISCUS_LEAF, "pinnule_angle_deg", 60.0)
        if raw_name == "pinnule_petiolule_frac_at_tip":
            return getattr(ANTHRISCUS_LEAF, "pinnule_petiolule_frac", 0.0)
        return 0.0
    return val


def to_plotly(img: Image.Image) -> go.Figure:
    """Wrap a PIL image into a plotly Figure with zoom/pan-persistent UI."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    fig = go.Figure(go.Image(z=arr))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=900,
        xaxis=dict(visible=False, range=[0, w], scaleanchor="y",
                   scaleratio=1, constrain="domain"),
        yaxis=dict(visible=False, range=[h, 0], constrain="domain"),
        # uirevision keeps the user's zoom/pan state across re-renders. As
        # long as this string stays the same the camera/axes don't reset
        # when the figure is replaced.
        uirevision="lock",
        dragmode="pan",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_with_mode(values, mode: str = "overlay") -> go.Figure:
    """`mode = "overlay"` → synth tinted+blended onto the reference photo.
       `mode = "synth"`   → bare synth render (light background, no photo)."""
    n_leaf = len(SLIDER_FIELDS)
    leaf_vals = values[:n_leaf]
    overlay_vals = values[n_leaf:]

    p = LeafParams(**asdict(ANTHRISCUS_LEAF))
    p.seed = 0
    for (label, _mn, _mx, _st, name), v in zip(SLIDER_FIELDS, leaf_vals):
        raw_name = name.lstrip("_")
        if raw_name in {"n_pinna_pairs", "n_pinnule_pairs",
                        "pinna_recursion_depth",
                        "leaflet_serration_periods"}:
            v = int(round(v))
        setattr(p, raw_name, v)

    rgb, bbox_diag = render_synth(p)
    if mode == "synth":
        return to_plotly(Image.fromarray(rgb))
    rotate_deg, scale_frac, cx, cy, alpha = overlay_vals
    img = make_overlay(rgb, bbox_diag, rotate_deg, scale_frac, cx, cy,
                       int(alpha))
    return to_plotly(img)


def update(*values_and_mode) -> go.Figure:
    """Slider-driven update: last positional arg is the current view mode
    (gr.State), so changing a slider doesn't reset the user's selected
    view back to overlay."""
    *values, mode = values_and_mode
    return _render_with_mode(values, mode if mode in {"overlay", "synth"}
                             else "overlay")


def render_overlay(*values) -> tuple[go.Figure, str]:
    return _render_with_mode(values, "overlay"), "overlay"


def render_synth_only(*values) -> tuple[go.Figure, str]:
    return _render_with_mode(values, "synth"), "synth"


def main() -> None:
    with gr.Blocks(title="Anthriscus leaf tweaker") as app:
        gr.Markdown("# Apiaceae leaf parameter tweaker")
        gr.Markdown(
            "Schiebe Slider rechts, Overlay aktualisiert sich live. "
            "Default-Werte = `ANTHRISCUS_LEAF` aus `src/synthetic/leaf.py`."
        )
        # current view mode — persists across slider updates so that
        # changing a value while in Synth-only doesn't snap back to Overlay
        view_mode = gr.State("overlay")
        with gr.Row():
            with gr.Column(scale=3, min_width=600):
                out = gr.Plot(label=("Overlay — Plotly toolbar (top-right of "
                                     "the chart) has zoom/pan/reset buttons; "
                                     "zoom state survives across renders"),
                              show_label=True, elem_id="overlay-plot")
            with gr.Column(scale=1):
                sliders: list[gr.Slider] = []
                with gr.Accordion("Geometry — pinna level", open=True):
                    for label, mn, mx, st, name in SLIDER_FIELDS[:9]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name), label=label)
                        sliders.append(s)
                with gr.Accordion("Geometry — pinnule level", open=False):
                    for label, mn, mx, st, name in SLIDER_FIELDS[9:20]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name), label=label)
                        sliders.append(s)
                with gr.Accordion("Leaflet shape & noise", open=False):
                    for label, mn, mx, st, name in SLIDER_FIELDS[20:]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name), label=label)
                        sliders.append(s)
                with gr.Accordion("Overlay placement", open=True):
                    for label, mn, mx, st, default in OVERLAY_FIELDS:
                        v0 = _STATE.get(f"_overlay::{label}", default)
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=v0, label=label)
                        sliders.append(s)
                with gr.Row():
                    overlay_btn = gr.Button("🖼 Overlay", scale=1)
                    synth_btn = gr.Button("🍃 Synth only", scale=1)
                with gr.Row():
                    save_btn = gr.Button("💾 Save preset",
                                         variant="primary", scale=2)
                    reset_btn = gr.Button("↺ Reset to defaults", scale=1)
                status = gr.Textbox(label="Status", interactive=False,
                                    show_label=False)

        # initial render: fire `update` when the page finishes loading so the
        # overlay is populated without the user having to nudge a slider.
        app.load(update, inputs=[*sliders, view_mode], outputs=out)

        for s in sliders:
            s.release(update, inputs=[*sliders, view_mode], outputs=out)

        overlay_btn.click(render_overlay, inputs=sliders,
                          outputs=[out, view_mode])
        synth_btn.click(render_synth_only, inputs=sliders,
                        outputs=[out, view_mode])
        save_btn.click(save_state, inputs=sliders, outputs=status)

        def _reset() -> tuple:
            # delete state file and return slider defaults from ANTHRISCUS_LEAF
            if STATE_PATH.exists():
                STATE_PATH.unlink()
            _STATE.clear()
            vals = [field_default(n) for *_, n in SLIDER_FIELDS]
            vals += [d for *_, d in OVERLAY_FIELDS]
            return (*vals, "Reset to ANTHRISCUS_LEAF defaults")

        reset_btn.click(_reset, inputs=None, outputs=[*sliders, status])

    app.launch(server_name="127.0.0.1", inbrowser=True, share=False)


if __name__ == "__main__":
    main()
