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

from src.synthetic.leaf import (                                        # noqa: E402
    ANTHRISCUS_LEAF, CONIUM_LEAF, DAUCUS_LEAF, AETHUSA_LEAF,
    HERACLEUM_LEAF, PASTINACA_LEAF,
    LeafParams, generate_apiaceae_leaf,
)
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

# Each species: (display name, LeafParams preset, ref folder, state-file key).
# The ref folder is under data/leaf_images/<folder>/_tweaker_reference.jpg;
# the state file is notebooks/output/leaf_tweaker_<key>.json.
SPECIES: dict[str, tuple[LeafParams, str, str]] = {
    "Anthriscus sylvestris": (ANTHRISCUS_LEAF, "Anthriscus_sylvestris",
                              "anthriscus_sylvestris"),
    "Conium maculatum":      (CONIUM_LEAF,     "Conium_maculatum",
                              "conium_maculatum"),
    "Daucus carota":         (DAUCUS_LEAF,     "Daucus_carota",
                              "daucus_carota"),
    "Aethusa cynapium":      (AETHUSA_LEAF,    "Aethusa_cynapium",
                              "aethusa_cynapium"),
    "Heracleum sphondylium": (HERACLEUM_LEAF,  "Heracleum_sphondylium",
                              "heracleum_sphondylium"),
    "Pastinaca sativa":      (PASTINACA_LEAF,  "Pastinaca_sativa",
                              "pastinaca_sativa"),
}
DEFAULT_SPECIES = "Anthriscus sylvestris"


def species_paths(species: str) -> tuple[Path, Path]:
    _, folder, key = SPECIES[species]
    ref = ROOT / "data" / "leaf_images" / folder / "_tweaker_reference.jpg"
    state = OUT_DIR / f"leaf_tweaker_{key}.json"
    return ref, state


# Display cap: high-res reference plates (Köhler ~2833×3973) blow up the
# Plotly payload to >100 MB per render and crash the browser tab after a
# few slider tweaks. The synth render is 1024 px, so downsampling the
# composed image to this cap loses no calibration precision.
DISPLAY_MAX_DIM = 1200

# Mutable per-process current state. Switching species mutates these in
# place so the existing render functions don't need a species arg.
_REF_FULL: dict[str, Image.Image | None] = {"img": None}
_CURRENT_SPECIES: dict[str, str] = {"name": DEFAULT_SPECIES}


def load_state(species: str) -> dict:
    _, state_path = species_paths(species)
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(*values) -> str:
    """Persist all slider values keyed by their internal name. Gradio passes
    each slider as a positional arg, so we capture them via ``*values``.
    Last positional arg is the current species (passed via gr.State)."""
    *slider_vals, species = values
    n_leaf = len(SLIDER_FIELDS)
    state = {}
    for (label, _mn, _mx, _st, name), v in zip(SLIDER_FIELDS,
                                                slider_vals[:n_leaf]):
        state[name] = float(v)
    for (label, _mn, _mx, _st, _d), v in zip(OVERLAY_FIELDS,
                                              slider_vals[n_leaf:]):
        state[f"_overlay::{label}"] = float(v)
    _, state_path = species_paths(species)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    msg = f"✓ Saved [{species}] preset -> {state_path.relative_to(ROOT)}"
    # gr.Info pops a toast notification at the top of the page that stays
    # visible for several seconds — more noticeable than the status box
    # flashing.
    gr.Info(msg, duration=4)
    return msg


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
                 cx: float, cy: float, alpha: int,
                 crop_l: float = 0.0, crop_t: float = 0.0,
                 crop_r: float = 1.0, crop_b: float = 1.0) -> Image.Image:
    ref_pil = _REF_FULL["img"]
    if ref_pil is None:
        return Image.fromarray(synth_rgb)
    full_w, full_h = ref_pil.size
    # crop to the user-defined sub-rectangle (fractions of full image).
    # Clamp to valid order so a slider mid-drag never hits a 0-area crop.
    l = max(0.0, min(crop_l, crop_r - 0.02))
    r = max(l + 0.02, min(crop_r, 1.0))
    t = max(0.0, min(crop_t, crop_b - 0.02))
    b = max(t + 0.02, min(crop_b, 1.0))
    box = (int(full_w * l), int(full_h * t),
           int(full_w * r), int(full_h * b))
    ref_pil = ref_pil.crop(box)
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
    ("Pinna apex extension",       0.0, 1.0, 0.05, "pinna_apex_extension_frac"),
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
    ("Pinnule apex extension",     0.0, 1.0, 0.05, "pinnule_apex_extension_frac"),

    ("Recursive length factor",    0.3, 1.2, 0.05, "recursive_length_factor"),
    ("Pinnule scale with pinna",   0.0, 1.0, 0.05, "pinnule_scale_with_pinna"),
    ("Terminal leaflet scale",     0.0, 2.0, 0.05, "terminal_leaflet_scale"),
    # 0 = off (use Terminal leaflet scale everywhere). > 0 = use this value
    # ONLY for the outermost rachis apex (= whole-leaf terminal pinna).
    # Lets species like Conium have a big terminal pinna without inflating
    # every lateral pinna's own apex by the same factor.
    ("Rachis apex scale (0=off)",  0.0, 20.0, 0.1, "rachis_apex_scale_override"),
    # 0 = recursive sub-pinnate apex on main rachis (legacy); 1 = single
    # pointed terminal leaflet polygon (botanically typical).
    ("Simplify rachis apex (0/1)", 0, 1, 1, "simplify_rachis_apex"),

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
    # Crop fractions on the reference image (0=left/top edge, 1=right/bottom).
    # Default = full image. Tighten to focus on a single leaf within a
    # composite illustration plate.
    ("Crop left",                    0.0, 0.98, 0.005, 0.0),
    ("Crop top",                     0.0, 0.98, 0.005, 0.0),
    ("Crop right",                   0.02, 1.0, 0.005, 1.0),
    ("Crop bottom",                  0.02, 1.0, 0.005, 1.0),
]


def field_default(name: str, species: str = DEFAULT_SPECIES,
                  state: dict | None = None):
    """Read default value: persisted state if present, else species preset.
    Fields prefixed with '_' are Optional → coerce None → numeric default the
    slider can display."""
    if state is None:
        state = load_state(species)
    if name in state:
        return state[name]
    raw_name = name.lstrip("_")
    preset = SPECIES[species][0]
    val = getattr(preset, raw_name, None)
    if val is None:
        # optional angle/petiolule fields → fall back to fixed-value field
        if raw_name in {"pinna_angle_at_base", "pinna_angle_at_tip"}:
            return getattr(preset, "pinna_angle_deg", 60.0)
        if raw_name in {"pinnule_angle_at_base", "pinnule_angle_at_tip"}:
            return getattr(preset, "pinnule_angle_deg", 60.0)
        if raw_name == "pinnule_petiolule_frac_at_tip":
            return getattr(preset, "pinnule_petiolule_frac", 0.0)
        return 0.0
    return val


def collect_slider_defaults(species: str) -> list[float]:
    state = load_state(species)
    vals = [field_default(n, species, state) for *_, n in SLIDER_FIELDS]
    for label, _mn, _mx, _st, default in OVERLAY_FIELDS:
        vals.append(state.get(f"_overlay::{label}", default))
    return vals


def load_ref_for(species: str) -> None:
    """(Re)load the reference image for the given species into the global
    cache. No-op if the file doesn't exist (overlay then degrades gracefully
    to synth-only)."""
    ref_path, _ = species_paths(species)
    if ref_path.exists():
        _REF_FULL["img"] = Image.open(ref_path).convert("RGB")
    else:
        _REF_FULL["img"] = None
    _CURRENT_SPECIES["name"] = species


# initial load — module top so the first render has a reference image
load_ref_for(DEFAULT_SPECIES)


def to_plotly(img: Image.Image) -> go.Figure:
    """Wrap a PIL image into a plotly Figure with zoom/pan-persistent UI.
    Downsamples to ``DISPLAY_MAX_DIM`` first — without this the high-res
    reference plate (~33 MP) gets serialized into Plotly's JSON payload
    every render and the browser tab OOMs after a handful of edits."""
    if max(img.size) > DISPLAY_MAX_DIM:
        s = DISPLAY_MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)),
                         Image.LANCZOS)
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


def _render_with_mode(values, mode: str = "overlay",
                      species: str = DEFAULT_SPECIES) -> go.Figure:
    """`mode = "overlay"` → synth tinted+blended onto the reference photo.
       `mode = "synth"`   → bare synth render (light background, no photo)."""
    n_leaf = len(SLIDER_FIELDS)
    leaf_vals = values[:n_leaf]
    overlay_vals = values[n_leaf:]

    p = LeafParams(**asdict(SPECIES[species][0]))
    p.seed = 0
    for (label, _mn, _mx, _st, name), v in zip(SLIDER_FIELDS, leaf_vals):
        raw_name = name.lstrip("_")
        if raw_name in {"n_pinna_pairs", "n_pinnule_pairs",
                        "pinna_recursion_depth",
                        "leaflet_serration_periods"}:
            v = int(round(v))
        if raw_name == "simplify_rachis_apex":
            v = bool(round(v))
        setattr(p, raw_name, v)

    rgb, bbox_diag = render_synth(p)
    if mode == "synth":
        return to_plotly(Image.fromarray(rgb))
    (rotate_deg, scale_frac, cx, cy, alpha,
     crop_l, crop_t, crop_r, crop_b) = overlay_vals
    img = make_overlay(rgb, bbox_diag, rotate_deg, scale_frac, cx, cy,
                       int(alpha), crop_l, crop_t, crop_r, crop_b)
    return to_plotly(img)


def update(*values_mode_species) -> go.Figure:
    """Slider-driven update: last two positional args are (mode, species)
    from gr.State. Changing a slider preserves both."""
    *values, mode, species = values_mode_species
    return _render_with_mode(values,
                             mode if mode in {"overlay", "synth"} else "overlay",
                             species if species in SPECIES else DEFAULT_SPECIES)


def render_overlay(*values_species) -> tuple[go.Figure, str]:
    *values, species = values_species
    return _render_with_mode(values, "overlay", species), "overlay"


def render_synth_only(*values_species) -> tuple[go.Figure, str]:
    *values, species = values_species
    return _render_with_mode(values, "synth", species), "synth"


def main() -> None:
    initial_state = load_state(DEFAULT_SPECIES)

    with gr.Blocks(title="Apiaceae leaf tweaker") as app:
        gr.Markdown("# Apiaceae leaf parameter tweaker")
        gr.Markdown(
            "Species-Dropdown wechselt Preset, Referenzbild und State-Datei. "
            "Slider-Werte werden pro Art unter "
            "`notebooks/output/leaf_tweaker_<species>.json` gespeichert."
        )
        view_mode = gr.State("overlay")
        species_state = gr.State(DEFAULT_SPECIES)

        with gr.Row():
            species_dd = gr.Dropdown(choices=list(SPECIES.keys()),
                                     value=DEFAULT_SPECIES,
                                     label="Species",
                                     scale=2, interactive=True)

        with gr.Row():
            with gr.Column(scale=3, min_width=600):
                out = gr.Plot(label=("Overlay — Plotly toolbar (top-right of "
                                     "the chart) has zoom/pan/reset buttons; "
                                     "zoom state survives across renders"),
                              show_label=True, elem_id="overlay-plot")
            with gr.Column(scale=1):
                sliders: list[gr.Slider] = []
                with gr.Accordion("Geometry — pinna level", open=True):
                    for label, mn, mx, st, name in SLIDER_FIELDS[:10]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name, DEFAULT_SPECIES,
                                                          initial_state),
                                      label=label)
                        sliders.append(s)
                with gr.Accordion("Geometry — pinnule level", open=False):
                    for label, mn, mx, st, name in SLIDER_FIELDS[10:25]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name, DEFAULT_SPECIES,
                                                          initial_state),
                                      label=label)
                        sliders.append(s)
                with gr.Accordion("Leaflet shape & noise", open=False):
                    for label, mn, mx, st, name in SLIDER_FIELDS[25:]:
                        s = gr.Slider(minimum=mn, maximum=mx, step=st,
                                      value=field_default(name, DEFAULT_SPECIES,
                                                          initial_state),
                                      label=label)
                        sliders.append(s)
                with gr.Accordion("Overlay placement", open=True):
                    for label, mn, mx, st, default in OVERLAY_FIELDS:
                        v0 = initial_state.get(f"_overlay::{label}", default)
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

        # initial render
        app.load(update, inputs=[*sliders, view_mode, species_state],
                 outputs=out)

        # Listen on `input` (not `change`) so renders fire on direct user
        # interaction — drag, type, arrow-button click — but NOT when the
        # species dropdown programmatically writes new values into the
        # sliders. Otherwise switching species would cascade ~30 renders
        # (one per slider value being applied).
        # `trigger_mode="always_last"` debounces bursts: when the user
        # clicks an arrow 5× quickly, only the latest value triggers a
        # render — older in-flight invocations are cancelled and queued
        # duplicates are dropped.
        for s in sliders:
            s.input(update, inputs=[*sliders, view_mode, species_state],
                    outputs=out, trigger_mode="always_last",
                    show_progress="hidden")

        overlay_btn.click(render_overlay,
                          inputs=[*sliders, species_state],
                          outputs=[out, view_mode])
        synth_btn.click(render_synth_only,
                        inputs=[*sliders, species_state],
                        outputs=[out, view_mode])
        save_btn.click(save_state,
                       inputs=[*sliders, species_state], outputs=status)

        def _switch_species(new_species: str):
            """Dropdown change: reload reference image, return new slider
            values for the new species (from its state file or its preset),
            and update the species gr.State."""
            if new_species not in SPECIES:
                new_species = DEFAULT_SPECIES
            load_ref_for(new_species)
            vals = collect_slider_defaults(new_species)
            return (*vals, new_species,
                    f"Loaded preset & state for [{new_species}]")

        species_dd.change(_switch_species, inputs=species_dd,
                          outputs=[*sliders, species_state, status])
        # also re-render after the slider values were swapped in
        species_dd.change(update,
                          inputs=[*sliders, view_mode, species_state],
                          outputs=out)

        def _reset(species: str) -> tuple:
            # delete *this species's* state file and return its preset defaults
            _, state_path = species_paths(species)
            if state_path.exists():
                state_path.unlink()
            vals = collect_slider_defaults(species)
            return (*vals, f"Reset [{species}] to preset defaults")

        reset_btn.click(_reset, inputs=species_state,
                        outputs=[*sliders, status])

    app.launch(server_name="127.0.0.1", inbrowser=True, share=False)


if __name__ == "__main__":
    main()
