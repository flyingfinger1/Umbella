"""First L-system-style generator for Apiaceae compound umbels.

Models the structurally diagnostic part of a Doldenblütler:
    main stem -> terminal compound umbel
        compound umbel = primary rays radiating from the stem tip
            each primary ray ends in an umbellet
                umbellet = pedicels radiating from the ray tip

Optional: lateral branches off the main stem, each ending in its own (smaller)
compound umbel. Leaves are intentionally not modeled yet — the umbel topology
is the key diagnostic feature for species discrimination within the family.

Output uses the same `Skeleton` data structure as the Pheno4D extractor, so
synthetic and real plants can flow through the same downstream code.

Conventions:
    - units: millimeters
    - z is up, plant base at origin
    - organ id 1 = main stem (incl. junctions); each ray, lateral and pedicel
      gets its own incrementing organ id
    - roles: "stem", "stem-junction", "lateral", "ray-base/mid/tip",
             "umbellet-center", "pedicel-tip"
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from src.geometry.skeleton import Skeleton


@dataclass
class ApiaceaeParams:
    # main stem
    stem_height_mm: float = 800.0
    n_internodes: int = 5
    stem_jitter_mm: float = 3.0          # x/y wiggle per internode

    # lateral branches off the main stem
    n_laterals: int = 0
    lateral_angle_deg: float = 45.0      # angle from vertical
    lateral_length_factor: float = 0.5   # fraction of remaining stem above branching point
    lateral_umbel_scale: float = 0.6     # how much smaller the lateral umbel is

    # primary umbel (terminal compound umbel at top of stem)
    n_primary_rays: int = 15             # diagnostic: e.g. Schierling 10-20, Bärenklau ~50
    primary_ray_length_mm: float = 70.0
    primary_ray_half_angle_deg: float = 50.0  # cone half-angle from vertical (0=straight up, 90=horizontal)
    primary_ray_nodes: int = 4

    # umbellet (secondary umbel at the tip of each primary ray)
    n_pedicels: int = 12                 # pedicels per umbellet
    pedicel_length_mm: float = 12.0
    pedicel_half_angle_deg: float = 70.0 # half-angle around the ray's own axis

    # phyllotaxis (rotation around stem axis between successive rays)
    phyllotaxis_deg: float = 137.5

    # involucre = ring of bracts at the base of a (compound) umbel
    n_involucre_bracts: int = 0
    involucre_bract_length_mm: float = 8.0
    involucre_bract_angle_deg: float = 80.0   # 0 = forward along parent axis; 90 = perpendicular; >90 = reflexed

    # involucel = ring of bracteoles at the base of an umbellet
    n_involucel_bracts: int = 0
    involucel_bract_length_mm: float = 5.0
    involucel_bract_angle_deg: float = 80.0   # >90 reflexed (Aethusa diagnostic)

    # multiplicative randomness applied to lengths and angles (0 = perfectly geometric)
    randomness: float = 0.08
    angle_jitter_deg: float = 6.0

    seed: int | None = None


def _orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to `direction` (and to each other)."""
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    return e1, e2


def _direction_in_cone(axis: np.ndarray, half_angle_rad: float, azimuth_rad: float) -> np.ndarray:
    e1, e2 = _orthonormal_frame(axis)
    return (np.cos(half_angle_rad) * axis
            + np.sin(half_angle_rad) * (np.cos(azimuth_rad) * e1
                                        + np.sin(azimuth_rad) * e2))


def generate_apiaceae(params: ApiaceaeParams | None = None) -> Skeleton:
    p = params or ApiaceaeParams()
    rng = np.random.default_rng(p.seed)

    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    organ: list[int] = []
    role: list[str] = []
    next_organ = 2  # 1 reserved for main stem

    def add_node(xyz, oid, r) -> int:
        nodes.append(np.asarray(xyz, dtype=np.float32))
        organ.append(oid)
        role.append(r)
        return len(nodes) - 1

    def jitter(value: float) -> float:
        return float(value) * (1.0 + rng.uniform(-p.randomness, p.randomness))

    def jitter_angle(deg: float) -> float:
        return deg + rng.uniform(-p.angle_jitter_deg, p.angle_jitter_deg)

    # 1. main stem — vertical chain with slight wiggle
    stem_indices = [add_node([0.0, 0.0, 0.0], 1, "stem")]
    z = 0.0
    internode = p.stem_height_mm / p.n_internodes
    for _ in range(p.n_internodes):
        z += jitter(internode)
        x = rng.normal(0, p.stem_jitter_mm)
        y = rng.normal(0, p.stem_jitter_mm)
        idx = add_node([x, y, z], 1, "stem")
        edges.append((stem_indices[-1], idx))
        stem_indices.append(idx)

    def add_bracts(
        attach_idx: int,
        parent_axis: np.ndarray,
        n_bracts: int,
        bract_length_mm: float,
        angle_deg: float,
        role_name: str,
        length_scale: float = 1.0,
    ) -> None:
        """Add a ring of bracts radiating around `parent_axis` from `attach_idx`.

        `angle_deg` is the half-angle from `parent_axis` to each bract:
            0   -> forward along axis
            90  -> perpendicular (typical involucre/involucel)
            >90 -> reflexed (points opposite to parent_axis; Aethusa diagnostic)
        """
        nonlocal next_organ
        if n_bracts <= 0 or bract_length_mm <= 0:
            return
        attach_xyz = nodes[attach_idx]
        for k in range(n_bracts):
            azimuth = np.radians(k * 360.0 / n_bracts + rng.uniform(-15, 15))
            half_angle = np.radians(jitter_angle(angle_deg))
            direction = _direction_in_cone(parent_axis, half_angle, azimuth)
            length = jitter(bract_length_mm) * length_scale
            tip = attach_xyz + direction * length
            organ_id = next_organ
            next_organ += 1
            tip_idx = add_node(tip, organ_id, role_name)
            edges.append((attach_idx, tip_idx))

    def build_compound_umbel(attach_idx: int, axis: np.ndarray, length_scale: float = 1.0) -> None:
        """Attach a compound umbel (primary rays + umbellets) to node `attach_idx`."""
        nonlocal next_organ
        # involucre at the base of the compound umbel
        add_bracts(
            attach_idx, axis,
            p.n_involucre_bracts,
            p.involucre_bract_length_mm,
            p.involucre_bract_angle_deg,
            role_name="bract",
            length_scale=length_scale,
        )

        attach_xyz = nodes[attach_idx]
        n_rays = max(3, int(round(jitter(p.n_primary_rays))))
        for k in range(n_rays):
            azimuth = np.radians((k * p.phyllotaxis_deg + rng.uniform(-5, 5)) % 360)
            half_angle = np.radians(jitter_angle(p.primary_ray_half_angle_deg))
            ray_dir = _direction_in_cone(axis, half_angle, azimuth)
            ray_len = jitter(p.primary_ray_length_mm) * length_scale
            ray_organ = next_organ
            next_organ += 1

            prev = attach_idx
            tip_xyz = attach_xyz
            for j in range(1, p.primary_ray_nodes + 1):
                t = j / p.primary_ray_nodes
                xyz = attach_xyz + ray_dir * (ray_len * t)
                rname = "ray-tip" if j == p.primary_ray_nodes else (
                    "ray-base" if j == 1 else "ray-mid")
                cur = add_node(xyz, ray_organ, rname)
                edges.append((prev, cur))
                prev = cur
                if j == p.primary_ray_nodes:
                    tip_xyz = xyz
            umbellet_idx = prev
            # mark the ray tip role-wise for downstream filtering
            role[umbellet_idx] = "umbellet-center"

            # involucel at the base of this umbellet
            add_bracts(
                umbellet_idx, ray_dir,
                p.n_involucel_bracts,
                p.involucel_bract_length_mm,
                p.involucel_bract_angle_deg,
                role_name="bracteole",
                length_scale=length_scale,
            )

            # umbellet — pedicels radiating around the ray's own axis
            n_ped = max(3, int(round(jitter(p.n_pedicels))))
            for q in range(n_ped):
                ped_az = np.radians(q * 360.0 / n_ped + rng.uniform(-10, 10))
                ped_half = np.radians(jitter_angle(p.pedicel_half_angle_deg))
                ped_dir = _direction_in_cone(ray_dir, ped_half, ped_az)
                ped_len = jitter(p.pedicel_length_mm) * length_scale
                ped_organ = next_organ
                next_organ += 1
                ped_tip = tip_xyz + ped_dir * ped_len
                ped_idx = add_node(ped_tip, ped_organ, "pedicel-tip")
                edges.append((umbellet_idx, ped_idx))

    # 2. lateral branches off the main stem (optional)
    if p.n_laterals > 0 and len(stem_indices) >= 3:
        # pick attachment internodes (skip base + tip)
        candidates = list(range(1, len(stem_indices) - 1))
        n_lat = min(p.n_laterals, len(candidates))
        chosen = rng.choice(candidates, size=n_lat, replace=False)
        for c in chosen:
            base_idx = int(c)
            base_xyz = nodes[base_idx]
            lateral_organ = next_organ
            next_organ += 1
            # direction: tilted from vertical, random azimuth
            az = np.radians(rng.uniform(0, 360))
            half = np.radians(jitter_angle(p.lateral_angle_deg))
            lat_dir = _direction_in_cone(np.array([0.0, 0.0, 1.0]), half, az)
            remaining = p.stem_height_mm - base_xyz[2]
            lat_len = remaining * p.lateral_length_factor * (1 + rng.uniform(-p.randomness, p.randomness))
            n_lat_nodes = 3
            prev = base_idx
            tip_xyz = base_xyz
            for j in range(1, n_lat_nodes + 1):
                t = j / n_lat_nodes
                xyz = base_xyz + lat_dir * (lat_len * t)
                cur = add_node(xyz, lateral_organ, "lateral")
                edges.append((prev, cur))
                prev = cur
                if j == n_lat_nodes:
                    tip_xyz = xyz
            # lateral umbel
            build_compound_umbel(prev, lat_dir, length_scale=p.lateral_umbel_scale)

    # 3. terminal compound umbel at top of main stem
    build_compound_umbel(stem_indices[-1], np.array([0.0, 0.0, 1.0]))

    return Skeleton(
        nodes=np.stack(nodes).astype(np.float32),
        edges=edges,
        node_organ=organ,
        node_role=role,
        metadata={"synthetic": True, "params": asdict(p)},
    )
