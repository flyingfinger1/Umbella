"""Procedural Apiaceae leaf generator (first prototype).

Apiaceae leaves are recursively compound (typically 2- to 3-pinnate). The
diagnostic signal between species sits in the dissection topology and the
shape of the terminal leaflets, not in fine textural detail — both of which
an L-system-style skeleton + parametric outline can capture.

Model:
    petiole (stalk)
    └── rachis (main axis, with N pairs of pinnae)
        └── pinna (recursive: itself a pinnate axis at depth-1)
            └── ... eventually a terminal leaflet:
                     a short midrib + an outline curve at offset distance,
                     describing an ovate-lanceolate silhouette around the midrib

Output uses the same `Skeleton` data structure as the inflorescence
generator so the existing render / point-cloud / training pipeline can
consume it without changes.

Conventions:
    - units: millimeters
    - leaf base at origin; the leaf extends along a configurable axis
    - roles: "leaf-petiole", "leaf-rachis", "leaf-pinna-rachis",
             "leaf-midrib", "leaf-edge"

This is a first prototype calibrated visually against Anthriscus sylvestris.
Numbers are educated guesses from ID-key descriptions and reference photos,
not measured ground truth — same caveat as the early inflorescence params.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.geometry.skeleton import Skeleton


@dataclass
class LeafParams:
    # petiole (leaf stalk before the first pair of leaflets)
    petiole_length_mm: float = 100.0

    # main rachis (leaf axis after the petiole)
    rachis_length_mm: float = 150.0

    # pinnae — the first-order leaflets attached to the rachis
    n_pinna_pairs: int = 6
    pinna_angle_deg: float = 60.0           # off-axis angle (0 = along rachis, 90 = perpendicular)
    pinna_angle_at_base: float | None = None  # if set, angle tapers along the
    pinna_angle_at_tip: float | None = None   # rachis: base steep (~85°),
                                              # tip swept-forward (~30°). Both
                                              # None → constant pinna_angle_deg.
    pinna_length_at_base: float = 60.0      # mm at the proximal end of the rachis
    pinna_length_at_tip: float = 25.0       # mm at the distal end (taper)
    pinna_length_peak: float | None = None  # if set, overall envelope is ovate:
                                            # length curves from base → peak (at
                                            # `pinna_length_peak_t`) → tip via a
                                            # smooth bell, giving the leaf a
                                            # heart/ovate silhouette instead of
                                            # a straight-edged trapezoid.
    pinna_length_peak_t: float = 0.30       # rachis fraction at which the peak sits

    # recursion depth controls overall leaf compound-ness:
    #   0 = simple-pinnate (each pinna IS a terminal leaflet)
    #   1 = bipinnate
    #   2 = tripinnate
    pinna_recursion_depth: int = 1

    # pinnules — the leaflets within each pinna (only used at depth >= 1)
    n_pinnule_pairs: int = 4
    pinnule_angle_deg: float = 50.0
    pinnule_angle_at_base: float | None = None
    pinnule_angle_at_tip: float | None = None
    pinnule_length_at_base: float = 14.0
    pinnule_length_at_tip: float = 7.0
    pinnule_length_peak: float | None = None
    pinnule_length_peak_t: float = 0.30

    # at each recursion deeper, pinnule lengths are multiplied by this factor.
    # captures the "branches get smaller as you go further into the dissection"
    # pattern visible in real fern-like Apiaceae leaves (Anthriscus, Conium).
    recursive_length_factor: float = 0.55

    # couple pinnule (sub-leaflet) size to the parent pinna's relative length.
    # 0.0 = pinnules are the same size on every pinna (default behaviour).
    # 1.0 = pinnules scale exactly with the parent pinna's length / base length;
    #       a half-as-long distal pinna carries pinnules half the size of basal.
    # Real Anthriscus / Conium / Daucus leaves do scale this way — distal
    # pinnae are not just shorter axes, the whole compound shrinks.
    pinnule_scale_with_pinna: float = 0.0

    # bare-stem fractions: each pinna / pinnule sits on a leafless stalk
    # (petiolule). 0.0 → leaflets attach right at the rachis junction;
    # 0.25 → first 25% of the pinna-axis is bare before pinnules start.
    pinna_petiolule_frac: float = 0.0
    pinnule_petiolule_frac: float = 0.0
    pinnule_petiolule_frac_at_tip: float | None = None  # if set, the per-pinna
                                                        # petiolule fraction
                                                        # tapers along the
                                                        # parent rachis.

    # rachis pair-spacing taper. 1.0 = uniform spacing along the rachis;
    # >1 = pinnae cluster near the base, gaps grow toward the tip;
    # <1 = pinnae cluster near the tip, gaps grow toward the base.
    pinna_spacing_power: float = 1.0
    pinnule_spacing_power: float = 1.0

    # how far past the last lateral pair the terminal apex sits, expressed
    # as a fraction of the remaining rachis. 1.0 = one full inter-pair
    # step (legacy behavior; visually fine when n_pairs ≥ 5). 0.0 = apex
    # sits directly on the last pair (no bare rachis after it). Lowering
    # this is the easiest way to clean up the "extra stem" past the last
    # pair when n_pinna_pairs is small.
    pinna_apex_extension_frac: float = 1.0
    pinnule_apex_extension_frac: float = 1.0

    # For tripinnate leaves (recursion_depth >= 2), the recursive apex of the
    # OUTERMOST rachis would otherwise produce a mini sub-pinnate cluster
    # (10+ sub-leaflets fanning out from a tiny apex stem). Botanically most
    # Apiaceae have a single pointed terminal leaflet at the rachis tip
    # instead. When True (default), collapse the outermost apex to a single
    # terminal leaflet polygon. No effect when recursion_depth <= 1 (the
    # apex was already a single polygon at depth=0).
    simplify_rachis_apex: bool = True

    # If > 0, overrides terminal_leaflet_scale ONLY for the outermost rachis
    # apex (= the terminal pinna of the whole leaf). All inner apexes
    # (per-pinna, per-pinnule) keep using terminal_leaflet_scale. Use this
    # for species like Conium where the rachis terminates in a full-size
    # terminal pinna rather than a small terminal leaflet, without blowing
    # up every lateral pinna's own apex by the same factor.
    rachis_apex_scale_override: float = 0.0

    # at the very tip of every pinnate axis, attach a single terminal leaflet
    # along the axis direction (instead of the axis ending in a naked tip).
    # `terminal_leaflet_scale` scales the terminal's length relative to the
    # last pair's length — typically slightly >1 so the terminal looks like
    # the "endblatt" of the pinna.
    terminal_leaflet_scale: float = 1.15

    # ultimate (terminal) leaflet — midrib + outline + blade fill
    leaflet_width_mm: float = 3.0           # max half-width of the ovate-lanceolate outline
    leaflet_outline_power: float = 0.7      # exponent on sin(pi*t) for the
                                            # outline curve. 0.5 = wide bulgy
                                            # ovate; 0.7 = current default;
                                            # 1.0 = pure sine (slimmer);
                                            # 2.0+ = lanceolate with nearly
                                            # straight sides; >>1 = very
                                            # pointy / spear-tip.
    leaflet_peak_t: float = 0.5             # fraction along the leaflet at
                                            # which the maximum half-width
                                            # sits. 0.5 = symmetric ovate
                                            # (current); <0.5 = broad at
                                            # base + pointed tip
                                            # (Anthriscus-typical, ≈0.25);
                                            # >0.5 = obovate (broad tip).
    leaflet_serration: float = 0.0          # 0 = smooth ovate; 0.15 = lightly notched edges
    leaflet_serration_periods: int = 5      # number of edge notches along one side
    leaflet_n_outline_points: int = 32      # outline resolution (edge-connected silhouette)
    leaflet_n_midrib_nodes: int = 4
    leaflet_blade_fill_points: int = 60     # interior surface points (isolated nodes,
                                            # picked up by sample_skeleton_pointcloud
                                            # as bare points so the leaflet looks solid
                                            # rather than a wireframe)

    # leaf orientation in space (the leaf is held away from origin along this axis)
    leaf_yaw_deg: float = 0.0               # rotation around the vertical (z) axis
    leaf_pitch_deg: float = 15.0            # downward tilt below horizontal

    # randomness
    randomness: float = 0.10                # ±fraction applied to lengths
    angle_jitter_deg: float = 7.0           # ±degrees applied to angles

    seed: int | None = None


def _orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to `direction` (and to each other)."""
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    helper = (np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9
              else np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(direction, e1)
    return e1, e2


def generate_apiaceae_leaf(
    params: LeafParams | None = None,
) -> tuple[Skeleton, list[tuple[np.ndarray, int]]]:
    """Generate a single procedural Apiaceae leaf.

    Returns:
        skeleton: midrib + rachis structure (no outline / no fill — the leaf
                  blades are returned separately as polygons so the renderer
                  can rasterize them as continuous filled surfaces).
        polygons: list of (vertices_xyz, organ_id) — one entry per terminal
                  leaflet. `vertices_xyz` is an ordered (N, 3) array tracing
                  the leaflet outline; `organ_id` is the role-aware color
                  key (matches the "leaf-blade" role in ROLE_TO_RGB).

    The leaf is rooted at the origin and extends along the axis defined by
    `leaf_yaw_deg` / `leaf_pitch_deg`. Caller is responsible for translating
    / rotating the result into a final scene.
    """
    p = params or LeafParams()
    rng = np.random.default_rng(p.seed)

    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    organ: list[int] = []
    role: list[str] = []
    polygons: list[tuple[np.ndarray, int]] = []
    next_organ = [1]

    def add_node(xyz, oid: int, r: str) -> int:
        nodes.append(np.asarray(xyz, dtype=np.float32))
        organ.append(oid)
        role.append(r)
        return len(nodes) - 1

    def jitter(value: float) -> float:
        return float(value) * (1.0 + rng.uniform(-p.randomness, p.randomness))

    def jitter_angle(deg: float) -> float:
        return deg + rng.uniform(-p.angle_jitter_deg, p.angle_jitter_deg)

    # leaf axis: predominantly horizontal, slight downward pitch
    yaw = np.radians(p.leaf_yaw_deg)
    pitch = np.radians(p.leaf_pitch_deg)
    leaf_axis = np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
    ])

    # 1. Petiole — straight chain from origin to first leaflet pair
    petiole_organ = next_organ[0]; next_organ[0] += 1
    base_idx = add_node([0.0, 0.0, 0.0], petiole_organ, "leaf-petiole")
    petiole_end_xyz = leaf_axis * p.petiole_length_mm
    petiole_end_idx = add_node(petiole_end_xyz, petiole_organ, "leaf-petiole")
    edges.append((base_idx, petiole_end_idx))

    def add_terminal_leaflet(attach_idx: int, axis_dir: np.ndarray,
                             length: float, max_width: float) -> None:
        """Attach a terminal leaflet: short midrib + an outline polygon.

        The midrib goes into the skeleton (as a thin chain of cylinder edges,
        for vein detail). The outline is added as a polygon vertex list which
        the renderer will rasterize as a single filled surface — no point
        fill is generated, the polygon IS the leaflet body.
        """
        midrib_organ = next_organ[0]; next_organ[0] += 1
        attach_xyz = nodes[attach_idx]

        # midrib chain (skeleton — gives a faint vein line through the leaflet)
        n_mid = max(2, p.leaflet_n_midrib_nodes)
        prev = attach_idx
        for k in range(1, n_mid + 1):
            t = k / n_mid
            xyz = attach_xyz + axis_dir * (length * t)
            idx = add_node(xyz, midrib_organ, "leaf-midrib")
            edges.append((prev, idx))
            prev = idx

        # outline polygon: ordered vertices around the ovate-lanceolate shape.
        # First sweep one side base→tip, then the other side tip→base, then
        # closes back at the base. width(t) = max_width * sin(pi*t)^0.7.
        e1, _ = _orthonormal_frame(axis_dir)
        n_out = max(6, p.leaflet_n_outline_points)
        half = n_out // 2
        # serration: a sinusoidal modulation that shrinks/grows the half-width
        # along t, giving a notched outline like real Apiaceae leaflets.
        serr = float(p.leaflet_serration)
        per = max(1, int(p.leaflet_serration_periods))
        outline_power = float(p.leaflet_outline_power)
        peak_t = max(1e-3, min(1 - 1e-3, float(p.leaflet_peak_t)))
        def width_at(t: float) -> float:
            # warp t so sin(pi*s) peaks at the desired peak_t along the leaflet.
            # Below the peak we compress [0, peak_t] → [0, 0.5]; above the peak
            # we compress [peak_t, 1] → [0.5, 1]. Result is asymmetric ovate
            # whenever peak_t ≠ 0.5.
            if t <= peak_t:
                s = 0.5 * (t / peak_t)
            else:
                s = 0.5 + 0.5 * (t - peak_t) / (1.0 - peak_t)
            base = max_width * np.sin(np.pi * s) ** outline_power
            if serr <= 0.0:
                return base
            # cosine modulation in [-1, +1] → notch depth `serr`; nodes per
            # side ≈ per (so per=5 gives ~5 lobes per edge).
            mod = 1.0 - serr * (0.5 - 0.5 * np.cos(2 * np.pi * per * t))
            return base * mod
        verts: list[np.ndarray] = [attach_xyz.copy()]
        for k in range(1, half + 1):                       # side +1, base→tip
            t = k / half
            verts.append(attach_xyz + axis_dir * (length * t) + e1 * width_at(t))
        for k in range(half - 1, 0, -1):                   # side -1, tip→base
            t = k / half
            verts.append(attach_xyz + axis_dir * (length * t) - e1 * width_at(t))
        # the polygon closes implicitly back to verts[0]

        blade_organ = next_organ[0]; next_organ[0] += 1
        polygons.append((np.stack(verts).astype(np.float32), blade_organ))
        # also register the blade_organ in the skeleton via a single sentinel
        # node at the leaflet centroid. This keeps role-aware color lookup
        # working without a parallel polygon-role map; the sentinel is
        # rendered as one isolated point but the polygon fill covers it.
        centroid = (attach_xyz + axis_dir * (length * 0.5))
        add_node(centroid, blade_organ, "leaf-blade")

    def build_pinnate_axis(start_idx: int, axis_dir: np.ndarray,
                           length: float, depth: int,
                           n_pairs: int, attach_angle_deg: float,
                           leaflet_len_base: float, leaflet_len_tip: float,
                           organ_role: str,
                           terminal_max_width: float,
                           leaflet_len_peak: float | None = None,
                           leaflet_len_peak_t: float = 0.3,
                           attach_angle_at_base: float | None = None,
                           attach_angle_at_tip: float | None = None,
                           petiolule_frac: float = 0.0,
                           terminal_scale: float = 1.0,
                           spacing_power: float = 1.0,
                           apex_extension_frac: float = 1.0,
                           parent_couple: float = 1.0) -> None:
        """Recursively build a pinnate axis with leaflet pairs along it.

        depth == 0  →  each leaflet is a terminal (midrib + outline)
        depth >= 1  →  each leaflet is itself a pinnate axis at depth-1
        """
        rachis_organ = next_organ[0]; next_organ[0] += 1
        start_xyz = nodes[start_idx]

        # Rachis layout in t-space (t = fraction of rachis `length`):
        #   t = 0                : axis start (= start_idx)
        #   t = t_first          : pair 0 (=  petiolule_frac, continuous bare
        #                          stalk before the first pair)
        #   t = t_first + (1-t_first) · ((j/n)^p)
        #                        : pair j (j = 0 … n-1)
        #   t = t_last + apex_ext · (1-t_last)
        #                        : terminal apex
        # spacing_power < 1 → gaps shrink toward the tip; > 1 → grow.
        # apex_extension_frac < 1 → reduce bare stalk between last pair and apex.
        # petiolule_frac > 0 → continuous bare stalk BEFORE the first pair,
        # remaining pairs scale into [petiolule_frac, t_last] proportionally.
        n_pairs_eff = max(1, n_pairs)
        t_first = max(0.0, min(0.99, petiolule_frac))

        def _remap(t_raw: float) -> float:
            return t_first + (1.0 - t_first) * t_raw

        t_last_raw = (((n_pairs_eff - 1) / n_pairs_eff) ** spacing_power
                      if n_pairs_eff > 0 else 0.0)
        t_last_pair = _remap(t_last_raw)
        t_apex = t_last_pair + apex_extension_frac * (1.0 - t_last_pair)

        # Build the rachis chain: start → pair-0 → pair-1 → … → pair-(n-1)
        # → apex. Always create new nodes for each pair so the spatial
        # positions reflect petiolule_frac continuously (instead of drop-
        # thresholding pairs out, which was discrete and felt frozen
        # between consecutive pair t-values).
        rachis_indices = [start_idx]
        for j in range(0, n_pairs_eff + 1):
            if j < n_pairs_eff:
                t_pos = _remap((j / n_pairs_eff) ** spacing_power)
            else:
                t_pos = t_apex
            xyz = start_xyz + axis_dir * (length * t_pos)
            idx = add_node(xyz, rachis_organ, organ_role)
            edges.append((rachis_indices[-1], idx))
            rachis_indices.append(idx)
        # rachis_indices: [start_idx, pair_0, pair_1, …, pair_(n-1), apex]
        # → pair j attaches at rachis_indices[j + 1]
        # → apex attaches at rachis_indices[n_pairs_eff + 1]

        # axis_dir's perpendicular plane — leaflets attach on +e1 / -e1 sides
        e1, _ = _orthonormal_frame(axis_dir)

        # attach a pair of leaflets at each pair-position. The RAW t-value
        # (= position normalized into [0, 1] without the petiolule offset)
        # is used for length / angle interpolation so that `pinna_length_at_base`
        # really means "the length of the first lateral pair" regardless of
        # petiolule_frac, and `at_tip` likewise for the last pair.
        for j in range(0, n_pairs_eff):
            t = (j / n_pairs_eff) ** spacing_power
            attach_idx = rachis_indices[j + 1]
            if leaflet_len_peak is None:
                # legacy linear taper — straight-edged trapezoid envelope
                leaflet_len = leaflet_len_base + t * (
                    leaflet_len_tip - leaflet_len_base)
            else:
                # ovate envelope: piecewise-quadratic through three control
                # points (0, base) → (peak_t, peak) → (1, tip), giving the
                # whole leaf a curved heart-shaped silhouette instead of
                # straight diagonals.
                tp = max(1e-3, min(1 - 1e-3, leaflet_len_peak_t))
                if t <= tp:
                    s = t / tp                            # 0..1
                    # smooth ease-in to the peak
                    s = np.sin(0.5 * np.pi * s)
                    leaflet_len = leaflet_len_base + s * (
                        leaflet_len_peak - leaflet_len_base)
                else:
                    s = (t - tp) / (1.0 - tp)             # 0..1
                    s = np.sin(0.5 * np.pi * s)
                    leaflet_len = leaflet_len_peak + s * (
                        leaflet_len_tip - leaflet_len_peak)
            # Keep the un-jittered taper value so that each side can get its
            # OWN length jitter — real Apiaceae leaves often have slightly
            # different left vs right pinna lengths (the leaf in Thomé's
            # Conium plate shows this clearly). Applying jitter once per
            # pair before the side loop locked the two sides to identical
            # lengths.
            leaflet_len_taper = leaflet_len

            # angle taper along the rachis: real Apiaceae leaves have steep
            # basal pinnae (~80-85°, almost perpendicular to the rachis) and
            # forward-swept distal pinnae (~30°, leaning toward the leaf tip).
            if attach_angle_at_base is not None and attach_angle_at_tip is not None:
                angle_here = attach_angle_at_base + t * (
                    attach_angle_at_tip - attach_angle_at_base)
            else:
                angle_here = attach_angle_deg

            for side in (+1, -1):
                # per-side independent jitter (length AND angle)
                leaflet_len = jitter(leaflet_len_taper)
                ang = np.radians(jitter_angle(angle_here))
                leaflet_dir = (np.cos(ang) * axis_dir
                               + np.sin(ang) * (side * e1))
                leaflet_dir /= np.linalg.norm(leaflet_dir) + 1e-12

                if depth > 0:
                    # at each level deeper, pinnule LENGTHS shrink by
                    # `recursive_length_factor`. Per-pinna `couple` then
                    # additionally scales BOTH length and width together so
                    # leaflets on a smaller pinna stay proportional rather
                    # than getting chubby relative to their length.
                    scale = p.recursive_length_factor ** (
                        p.pinna_recursion_depth - depth + 1
                    )
                    # optional: couple sub-leaflet size to this pinna's
                    # relative length, so distal (shorter) pinnae carry
                    # smaller pinnules. coupling = 1 → full proportional;
                    # 0 → constant pinnule size on every pinna.
                    if p.pinnule_scale_with_pinna > 0 and leaflet_len_base > 0:
                        rel = leaflet_len / leaflet_len_base
                        local_couple = (1.0 - p.pinnule_scale_with_pinna) \
                                       + p.pinnule_scale_with_pinna * rel
                    else:
                        local_couple = 1.0
                    # cum_couple propagates all coupling factors from outer
                    # recursion levels. Without this, the deepest level
                    # always used `p.pinnule_length_at_base * scale * local_couple`
                    # — which dropped every ancestor's `local_couple` and
                    # left e.g. terminal leaflets on a small distal pinna at
                    # full size, sticking out past their parent pinnule.
                    cum_couple = parent_couple * local_couple
                    build_pinnate_axis(
                        start_idx=attach_idx,
                        axis_dir=leaflet_dir,
                        length=leaflet_len,
                        depth=depth - 1,
                        n_pairs=p.n_pinnule_pairs,
                        attach_angle_deg=p.pinnule_angle_deg,
                        leaflet_len_base=p.pinnule_length_at_base * scale * cum_couple,
                        leaflet_len_tip=p.pinnule_length_at_tip * scale * cum_couple,
                        # width tracks length so smaller (coupled) leaflets
                        # stay proportional, not chubby:
                        leaflet_len_peak=(
                            p.pinnule_length_peak * scale
                            if p.pinnule_length_peak is not None else None),
                        leaflet_len_peak_t=p.pinnule_length_peak_t,
                        attach_angle_at_base=p.pinnule_angle_at_base,
                        attach_angle_at_tip=p.pinnule_angle_at_tip,
                        petiolule_frac=(
                            p.pinnule_petiolule_frac
                            if p.pinnule_petiolule_frac_at_tip is None
                            else p.pinnule_petiolule_frac + t * (
                                p.pinnule_petiolule_frac_at_tip
                                - p.pinnule_petiolule_frac)),
                        terminal_scale=p.terminal_leaflet_scale,
                        spacing_power=p.pinnule_spacing_power,
                        apex_extension_frac=p.pinnule_apex_extension_frac,
                        organ_role="leaf-pinna-rachis",
                        terminal_max_width=terminal_max_width * local_couple,
                        parent_couple=cum_couple,
                    )
                else:
                    # depth==0: this attachment IS a leaflet polygon. Scale
                    # its width by the same coupling factor used for length
                    # (computed from the parent pinna's relative size).
                    if p.pinnule_scale_with_pinna > 0 and leaflet_len_base > 0:
                        rel = leaflet_len / leaflet_len_base
                        couple_w = (1.0 - p.pinnule_scale_with_pinna) \
                                   + p.pinnule_scale_with_pinna * rel
                    else:
                        couple_w = 1.0
                    add_terminal_leaflet(
                        attach_idx=attach_idx,
                        axis_dir=leaflet_dir,
                        length=leaflet_len,
                        max_width=terminal_max_width * couple_w,
                    )

        # apex: single terminal leaflet (or sub-axis) along axis_dir,
        # attached at the very end of the rachis chain.
        if terminal_scale > 0:
            # Apex size derives from the LAST LATERAL PAIR's length, not
            # from leaflet_len_tip directly. The taper from base to tip is
            # only fully reached at t=1 (the apex position), so reading
            # leaflet_len_tip would underestimate the size of the structures
            # the apex sits between. terminal_scale is now a fraction of
            # the last pair (1.0 = same size, 0.5 = half, etc.). For a
            # depth=2 inner apex with last pair ~16 mm, this gives a 14 mm
            # apex stem at terminal_scale=0.85 instead of the previous 3 mm.
            if n_pairs_eff > 0:
                t_last = ((n_pairs_eff - 1) / n_pairs_eff) ** spacing_power
                if leaflet_len_peak is None:
                    last_pair_len = (leaflet_len_base
                                     + t_last * (leaflet_len_tip
                                                 - leaflet_len_base))
                else:
                    tp = max(1e-3, min(1 - 1e-3, leaflet_len_peak_t))
                    if t_last <= tp:
                        s = np.sin(0.5 * np.pi * (t_last / tp))
                        last_pair_len = (leaflet_len_base
                                         + s * (leaflet_len_peak
                                                - leaflet_len_base))
                    else:
                        s = np.sin(0.5 * np.pi
                                   * ((t_last - tp) / (1.0 - tp)))
                        last_pair_len = (leaflet_len_peak
                                         + s * (leaflet_len_tip
                                                - leaflet_len_peak))
            else:
                last_pair_len = leaflet_len_tip
            term_len = jitter(last_pair_len) * terminal_scale
            apex_idx = rachis_indices[n_pairs_eff + 1]
            # At the outermost rachis (depth == recursion_depth), and only
            # for tripinnate-or-deeper leaves (depth >= 2), cap the apex's
            # internal recursion at one level — i.e. the apex looks like a
            # bipinnate apex (mini-pinnate fan of simple polygons) instead
            # of recursively sub-pinnate (which produced a rectangular
            # cluster of 100+ polygons on a 4 mm stem).
            is_outermost = (depth == p.pinna_recursion_depth)
            collapse_one_level = (is_outermost and p.simplify_rachis_apex
                                  and depth >= 2)
            if depth > 0:
                # depth used for the apex recursion. With collapse_one_level
                # we go straight to depth=0 (simple polygons in the pair
                # loop) instead of depth-1.
                apex_depth = 0 if collapse_one_level else (depth - 1)
                scale = p.recursive_length_factor ** (
                    p.pinna_recursion_depth - depth + 1)
                # At EVERY apex (outermost rachis OR per-pinna), use pure
                # proportional scaling for sub-structure size — i.e.
                # local_couple = term_len / leaflet_len_base, without the
                # 15 % floor that a partial pinnule_scale_with_pinna
                # produces for lateral pairs. Otherwise a tiny apex stem
                # (a few mm) inherits sub-leaflets at 15 % of
                # pinnule_length_at_base (~10 mm), which stick way out past
                # the stem and look deformed. For Anthriscus
                # (pinnule_scale_with_pinna = 1.0) this matches the previous
                # behavior exactly (the 15 % floor was already 0 %).
                if leaflet_len_base > 0:
                    local_couple = term_len / leaflet_len_base
                else:
                    local_couple = 1.0
                cum_couple = parent_couple * local_couple
                build_pinnate_axis(
                    start_idx=apex_idx,
                    axis_dir=axis_dir,
                    length=term_len,
                    depth=apex_depth,
                    n_pairs=p.n_pinnule_pairs,
                    attach_angle_deg=p.pinnule_angle_deg,
                    leaflet_len_base=p.pinnule_length_at_base * scale * cum_couple,
                    leaflet_len_tip=p.pinnule_length_at_tip * scale * cum_couple,
                    leaflet_len_peak=(
                        p.pinnule_length_peak * scale
                        if p.pinnule_length_peak is not None else None),
                    leaflet_len_peak_t=p.pinnule_length_peak_t,
                    attach_angle_at_base=p.pinnule_angle_at_base,
                    attach_angle_at_tip=p.pinnule_angle_at_tip,
                    petiolule_frac=p.pinnule_petiolule_frac,
                    terminal_scale=p.terminal_leaflet_scale,
                    spacing_power=p.pinnule_spacing_power,
                    apex_extension_frac=p.pinnule_apex_extension_frac,
                    organ_role="leaf-pinna-rachis",
                    terminal_max_width=terminal_max_width * local_couple,
                    parent_couple=cum_couple,
                )
            else:
                # depth==0 apex: same width-scaling as the depth==0 pair case
                if p.pinnule_scale_with_pinna > 0 and leaflet_len_base > 0:
                    rel = term_len / leaflet_len_base
                    couple_w = (1.0 - p.pinnule_scale_with_pinna) \
                               + p.pinnule_scale_with_pinna * rel
                else:
                    couple_w = 1.0
                add_terminal_leaflet(
                    attach_idx=apex_idx,
                    axis_dir=axis_dir,
                    length=term_len,
                    max_width=terminal_max_width * couple_w,
                )

    # 2. Build the rachis from the petiole end
    build_pinnate_axis(
        start_idx=petiole_end_idx,
        axis_dir=leaf_axis,
        length=p.rachis_length_mm,
        depth=p.pinna_recursion_depth,
        n_pairs=p.n_pinna_pairs,
        attach_angle_deg=p.pinna_angle_deg,
        leaflet_len_base=p.pinna_length_at_base,
        leaflet_len_tip=p.pinna_length_at_tip,
        leaflet_len_peak=p.pinna_length_peak,
        leaflet_len_peak_t=p.pinna_length_peak_t,
        attach_angle_at_base=p.pinna_angle_at_base,
        attach_angle_at_tip=p.pinna_angle_at_tip,
        petiolule_frac=p.pinna_petiolule_frac,
        # outermost rachis apex can use a separate override scale so the
        # whole-leaf terminal pinna can be sized independently of the
        # smaller per-pinna apexes that all use terminal_leaflet_scale.
        terminal_scale=(p.rachis_apex_scale_override
                        if p.rachis_apex_scale_override > 0
                        else p.terminal_leaflet_scale),
        spacing_power=p.pinna_spacing_power,
        apex_extension_frac=p.pinna_apex_extension_frac,
        organ_role="leaf-rachis",
        terminal_max_width=p.leaflet_width_mm,
    )

    skel = Skeleton(
        nodes=np.stack(nodes).astype(np.float32),
        edges=edges,
        node_organ=organ,
        node_role=role,
        metadata={"synthetic": True, "kind": "leaf", "params": asdict(p)},
    )
    return skel, polygons


# Per-species leaf parameter presets — visually calibrated against ID-key
# descriptions and reference photos, not measured. Calibration against an
# expert reviewer (e.g. PH-Ökogarten visit, Birgit Nordt at BGBM) is a
# planned next step.
#
# The 6-species lineup is ordered roughly from "finely dissected" to "broad
# simple-pinnate", matching the botanical spectrum within Apiaceae.

# Anthriscus sylvestris — bipinnate. Initial calibration on 2026-05-10
# via the interactive tweaker (notebooks/31_leaf_tweaker.py); reference
# photo source has since been retracted (license incompatible with ML
# use). Re-validation against a CC-licensed top-down photo is pending.
ANTHRISCUS_LEAF = LeafParams(
    petiole_length_mm=80.0,
    rachis_length_mm=370.0,
    n_pinna_pairs=5,
    pinna_angle_deg=55.0,                   # fallback (per-position taper below)
    pinna_angle_at_base=75.0,               # basal pinnae nearly perpendicular
    pinna_angle_at_tip=27.0,                # apical pinnae sharply forward-swept
    pinna_length_at_base=205.0,
    pinna_length_at_tip=65.0,
    pinna_recursion_depth=1,                # bipinnate (real Anthriscus structure)
    n_pinnule_pairs=6,
    pinnule_angle_deg=60.0,
    pinnule_angle_at_base=74.0,
    pinnule_angle_at_tip=18.0,
    pinnule_length_at_base=133.0,
    pinnule_length_at_tip=54.0,
    recursive_length_factor=0.8,
    pinnule_scale_with_pinna=1.0,           # smaller pinnae carry smaller leaflets
    pinna_petiolule_frac=0.0,
    pinnule_petiolule_frac=0.03,
    pinnule_petiolule_frac_at_tip=0.02,
    pinna_spacing_power=0.65,               # gaps shrink toward tip
    pinnule_spacing_power=0.9,
    # Apex size formula changed 2026-05-11 from leaflet_len_tip to
    # last_pair_len based. Old 1.4 × tip (91 mm) → new 1.08 × last-pair (84 mm).
    terminal_leaflet_scale=1.08,
    leaflet_width_mm=19.7,
    leaflet_outline_power=0.7,              # rounded ovate base curve
    leaflet_peak_t=0.2,                     # asymmetric: broad at base, pointed tip
    leaflet_serration=0.3,
    leaflet_serration_periods=7,
    randomness=0.13,
    angle_jitter_deg=7.0,
)

# Conium maculatum — tripinnate, calibrated 2026-05-11 via tweaker against
# Thomé Flora plate 385. Architecture: 2 lateral pinna pairs + 1 full-size
# terminal pinna (= "imparipinnate" reading), see research diary entry.
CONIUM_LEAF = LeafParams(
    petiole_length_mm=100.0,
    rachis_length_mm=400.0,
    n_pinna_pairs=2,
    pinna_angle_deg=60.0,
    pinna_angle_at_base=61.0,
    pinna_angle_at_tip=30.0,
    pinna_length_at_base=265.0,
    pinna_length_at_tip=65.0,
    pinna_spacing_power=0.9,
    pinna_apex_extension_frac=1.0,
    pinna_recursion_depth=2,                # tripinnate
    n_pinnule_pairs=6,
    pinnule_angle_deg=60.0,
    pinnule_angle_at_base=62.0,
    pinnule_angle_at_tip=23.0,
    pinnule_length_at_base=200.0,
    pinnule_length_at_tip=55.0,
    pinnule_petiolule_frac=0.22,
    pinnule_spacing_power=1.0,
    pinnule_apex_extension_frac=0.85,
    recursive_length_factor=0.4,            # tight shrinkage between levels
    pinnule_scale_with_pinna=0.55,
    terminal_leaflet_scale=0.85,
    rachis_apex_scale_override=2.5,         # full-size terminal pinna
    simplify_rachis_apex=False,             # keep recursive structure on apex
    leaflet_width_mm=4.5,
    leaflet_outline_power=1.5,
    leaflet_peak_t=0.5,
    leaflet_serration=0.6,
    leaflet_serration_periods=10,
    randomness=0.3,
    angle_jitter_deg=7.0,
    leaflet_blade_fill_points=35,
)

# Daucus carota — smaller, ultimate segments very narrow (almost thread-like)
DAUCUS_LEAF = LeafParams(
    petiole_length_mm=50.0,
    rachis_length_mm=100.0,
    n_pinna_pairs=5,
    pinna_angle_deg=60.0,
    pinna_length_at_base=50.0,
    pinna_length_at_tip=12.0,
    pinna_recursion_depth=2,
    n_pinnule_pairs=4,
    pinnule_angle_deg=70.0,
    pinnule_length_at_base=12.0,
    pinnule_length_at_tip=3.0,
    recursive_length_factor=1.0,
    leaflet_width_mm=0.7,               # very narrow — thread-like
    leaflet_blade_fill_points=20,
)

# Aethusa cynapium — parsley-like, less dissected, ovate leaflets
AETHUSA_LEAF = LeafParams(
    petiole_length_mm=40.0,
    rachis_length_mm=80.0,
    n_pinna_pairs=4,
    pinna_angle_deg=55.0,
    pinna_length_at_base=40.0,
    pinna_length_at_tip=15.0,
    pinna_recursion_depth=1,            # simple-pinnate — each pinna is a leaflet
    leaflet_width_mm=4.0,               # broad ovate
    leaflet_blade_fill_points=70,
)

# Heracleum sphondylium — large with broad palmately-lobed leaflets
HERACLEUM_LEAF = LeafParams(
    petiole_length_mm=180.0,
    rachis_length_mm=300.0,
    n_pinna_pairs=2,                    # very few large lobed leaflets
    pinna_angle_deg=55.0,
    pinna_length_at_base=130.0,
    pinna_length_at_tip=80.0,
    pinna_recursion_depth=0,            # each pinna IS the terminal leaflet
    leaflet_width_mm=35.0,              # very broad
    leaflet_blade_fill_points=400,      # large area, more fill points
)

# Pastinaca sativa — pinnate with ovate-toothed leaflets
PASTINACA_LEAF = LeafParams(
    petiole_length_mm=80.0,
    rachis_length_mm=180.0,
    n_pinna_pairs=4,
    pinna_angle_deg=60.0,
    pinna_length_at_base=70.0,
    pinna_length_at_tip=30.0,
    pinna_recursion_depth=0,            # simple-pinnate
    leaflet_width_mm=15.0,               # broad ovate
    leaflet_blade_fill_points=150,
)


SPECIES_LEAVES: dict[str, LeafParams] = {
    "Anthriscus_sylvestris": ANTHRISCUS_LEAF,
    "Conium_maculatum": CONIUM_LEAF,
    "Daucus_carota": DAUCUS_LEAF,
    "Aethusa_cynapium": AETHUSA_LEAF,
    "Heracleum_sphondylium": HERACLEUM_LEAF,
    "Pastinaca_sativa": PASTINACA_LEAF,
}
