"""Sample a point cloud on the cylindrical surface around a skeleton.

Treats each edge as a thin cylinder (radius depends on the role of the edge's
endpoints). Samples uniformly on the cylinder surface, with optional Gaussian
noise to emulate scanner artifacts. Output matches the Pheno4D format
convention: (N, 3) float32 xyz + (N,) int32 organ labels.

Works on any Skeleton — synthetic or extracted from real data.
"""

from __future__ import annotations

import numpy as np

from .skeleton import Skeleton


# Reasonable starting radii (mm) for Apiaceae-scale plants. Tunable per call.
DEFAULT_RADIUS_BY_ROLE: dict[str, float] = {
    "stem": 7.0,
    "stem-junction": 7.0,
    "lateral": 4.0,
    "ray-base": 1.8,
    "ray-mid": 1.5,
    "ray-tip": 1.2,
    "umbellet-center": 1.5,
    "pedicel-tip": 0.5,
    "bract": 1.0,
    "bracteole": 0.7,
    # leaf roles for Pheno4D-derived skeletons
    "leaf-base": 1.0,
    "leaf-mid": 1.0,
    "leaf-tip": 0.6,
    # synthetic Apiaceae leaf roles
    "leaf-petiole":      1.6,
    "leaf-rachis":       1.0,
    "leaf-pinna-rachis": 0.6,
    "leaf-midrib":       0.3,
    "leaf-edge":         0.2,
}


def _orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(direction, e1)
    return e1, e2


def sample_skeleton_pointcloud(
    skel: Skeleton,
    points_per_mm2: float = 0.5,
    noise_mm: float = 0.3,
    radius_by_role: dict[str, float] | None = None,
    default_radius_mm: float = 1.5,
    min_points_per_edge: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (xyz, organ_label) points on cylinder surfaces around skeleton edges.

    `points_per_mm2` is density on the cylinder lateral surface, so longer or
    thicker edges naturally get more points. Each point is labeled with the
    organ id of its edge (stem edges use organ 1; for cross-organ edges, e.g.
    leaf↔stem joins, the non-stem organ wins).
    """
    rng = np.random.default_rng(seed)
    rmap = {**DEFAULT_RADIUS_BY_ROLE, **(radius_by_role or {})}

    all_xyz: list[np.ndarray] = []
    all_lbl: list[np.ndarray] = []

    for a, b in skel.edges:
        pa = skel.nodes[a]
        pb = skel.nodes[b]
        ab = pb - pa
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            continue
        axis = ab / L

        ra = rmap.get(skel.node_role[a], default_radius_mm)
        rb = rmap.get(skel.node_role[b], default_radius_mm)
        r = 0.5 * (ra + rb)

        # cylinder lateral surface area
        area = 2.0 * np.pi * r * L
        n = max(min_points_per_edge, int(np.ceil(area * points_per_mm2)))

        # parametric position along axis [0..L] and azimuth [0..2pi]
        t = rng.uniform(0.0, L, size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        e1, e2 = _orthonormal_frame(axis)
        radial = (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)
        # taper radius linearly between endpoints
        local_r = ra + (rb - ra) * (t / L)
        pts = pa[None, :] + t[:, None] * axis[None, :] + radial * local_r[:, None]
        if noise_mm > 0:
            pts += rng.normal(0.0, noise_mm, size=pts.shape)

        # label: prefer the non-stem organ if the edge crosses (e.g. join edge)
        oa, ob = skel.node_organ[a], skel.node_organ[b]
        if oa == ob:
            label = oa
        elif oa == 1:
            label = ob
        elif ob == 1:
            label = oa
        else:
            label = max(oa, ob)
        all_xyz.append(pts.astype(np.float32))
        all_lbl.append(np.full(n, label, dtype=np.int32))

    # Isolated nodes (those not on any edge) are emitted as bare points at
    # their own location. This is how the procedural leaf generator passes
    # surface-fill points for leaflet blades into the cloud — the leaflet
    # silhouette is described by edges, the interior by a cluster of
    # isolated nodes with role "leaf-blade".
    if skel.edges:
        on_edge = set()
        for a, b in skel.edges:
            on_edge.add(int(a))
            on_edge.add(int(b))
        isolated = [i for i in range(len(skel.node_role)) if i not in on_edge]
    else:
        isolated = list(range(len(skel.node_role)))
    if isolated:
        pts = skel.nodes[isolated].astype(np.float32)
        if noise_mm > 0:
            pts = pts + rng.normal(0.0, noise_mm, size=pts.shape).astype(np.float32)
        labels = np.array([skel.node_organ[i] for i in isolated], dtype=np.int32)
        all_xyz.append(pts)
        all_lbl.append(labels)

    if not all_xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int32)
    return np.concatenate(all_xyz, axis=0), np.concatenate(all_lbl, axis=0)
