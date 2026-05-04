"""Structural features extracted from a Skeleton.

Designed to be cheap to compute, scale-aware (most features are scale-invariant
or normalized), and topology-aware (branching counts, degree statistics).
Used for downstream classification / clustering experiments.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.geometry.skeleton import Skeleton


FEATURE_NAMES = [
    "n_nodes",
    "n_edges",
    "n_leaves",
    "n_junctions",
    "n_stem_nodes",
    "n_branch_nodes",          # nodes with degree >= 3
    "max_degree",
    "mean_degree",
    "total_length",            # sum of all edge lengths
    "stem_length",
    "leaf_length_total",
    "leaf_length_mean",
    "leaf_length_std",
    "leaves_per_stem_unit",    # n_leaves / stem_length
    "branching_factor",        # n_branch_nodes / n_stem_nodes
    "bbox_width",              # x extent
    "bbox_depth",              # y extent
    "bbox_height",             # z extent
    "aspect_ratio_hw",         # height / max(width, depth)
    "stem_height_fraction",    # stem_z_extent / bbox_height
    # bract-aware features (specific to Apiaceae diagnostics)
    "n_bracts",                       # involucre bracts at compound umbel base
    "n_bracteoles",                   # involucel bracts at umbellet base
    "bracteoles_per_umbellet",
    "mean_bract_length",
    "mean_bracteole_length",
    "mean_bracteole_reflex_angle",    # >90 = reflexed (Aethusa diagnostic)
    "bracteole_to_pedicel_length_ratio",  # >1 in Aethusa, <1 elsewhere
]


def _edge_length(skel: Skeleton, i: int, j: int) -> float:
    return float(np.linalg.norm(skel.nodes[i] - skel.nodes[j]))


def skeleton_features(skel: Skeleton) -> np.ndarray:
    nodes = skel.nodes
    edges = skel.edges
    organ = skel.node_organ
    role = skel.node_role

    n_nodes = len(nodes)
    n_edges = len(edges)

    # degree per node
    deg = np.zeros(n_nodes, dtype=np.int32)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1

    # role / organ counts
    n_leaves = sum(1 for r in role if r == "leaf-base")
    n_junctions = sum(1 for r in role if r == "stem-junction")
    n_stem_nodes = sum(1 for o in organ if o == 1)
    stem_node_idx_set = {i for i, o in enumerate(organ) if o == 1}
    n_branch_nodes = sum(1 for i in stem_node_idx_set if deg[i] >= 3)
    max_degree = int(deg.max()) if n_nodes else 0
    mean_degree = float(deg.mean()) if n_nodes else 0.0

    # edge lengths split by edge category (stem-stem, leaf-leaf, leaf-stem)
    total_length = 0.0
    stem_length = 0.0
    leaf_lengths = defaultdict(float)  # per leaf-id, sum of internal edge lengths
    for a, b in edges:
        L = _edge_length(skel, a, b)
        total_length += L
        oa, ob = organ[a], organ[b]
        if oa == 1 and ob == 1:
            stem_length += L
        elif oa == ob and oa >= 2:
            leaf_lengths[oa] += L
        # cross edges (leaf-stem joins) deliberately excluded from leaf length

    leaf_lens = np.array(list(leaf_lengths.values()), dtype=np.float32) if leaf_lengths else np.zeros(0, dtype=np.float32)
    leaf_length_total = float(leaf_lens.sum())
    leaf_length_mean = float(leaf_lens.mean()) if len(leaf_lens) else 0.0
    leaf_length_std = float(leaf_lens.std()) if len(leaf_lens) else 0.0

    leaves_per_stem_unit = (n_leaves / stem_length) if stem_length > 0 else 0.0
    branching_factor = (n_branch_nodes / n_stem_nodes) if n_stem_nodes > 0 else 0.0

    # bounding box
    if n_nodes:
        ext = nodes.max(axis=0) - nodes.min(axis=0)
        bbox_width, bbox_depth, bbox_height = float(ext[0]), float(ext[1]), float(ext[2])
    else:
        bbox_width = bbox_depth = bbox_height = 0.0
    aspect_ratio_hw = (bbox_height / max(bbox_width, bbox_depth)) if max(bbox_width, bbox_depth) > 0 else 0.0

    # stem z-extent fraction
    if stem_node_idx_set:
        stem_z = nodes[sorted(stem_node_idx_set), 2]
        stem_z_extent = float(stem_z.max() - stem_z.min())
    else:
        stem_z_extent = 0.0
    stem_height_fraction = (stem_z_extent / bbox_height) if bbox_height > 0 else 0.0

    # ---- bract / bracteole features ---------------------------------------
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    bract_idx = [i for i, r in enumerate(role) if r == "bract"]
    bracteole_idx = [i for i, r in enumerate(role) if r == "bracteole"]
    umbellet_idx = [i for i, r in enumerate(role) if r == "umbellet-center"]

    n_bracts = len(bract_idx)
    n_bracteoles = len(bracteole_idx)
    bracteoles_per_umbellet = (n_bracteoles / len(umbellet_idx)) if umbellet_idx else 0.0

    # bract length: simple distance to its single neighbor (the attachment point)
    bract_lens: list[float] = []
    for bi in bract_idx:
        if not adj[bi]:
            continue
        parent = next(iter(adj[bi]))
        bract_lens.append(_edge_length(skel, bi, parent))
    mean_bract_length = float(np.mean(bract_lens)) if bract_lens else 0.0

    # bracteole length + reflex angle relative to its parent ray's direction
    bracteole_lens: list[float] = []
    bracteole_angles: list[float] = []
    for bi in bracteole_idx:
        if not adj[bi]:
            continue
        ucenter = next(iter(adj[bi]))
        bracteole_lens.append(_edge_length(skel, bi, ucenter))
        # find ray direction: previous ray node (same organ as umbellet center)
        u_organ = organ[ucenter]
        ray_prev = None
        for nb in adj[ucenter]:
            if nb != bi and organ[nb] == u_organ:
                ray_prev = nb
                break
        if ray_prev is None:
            continue
        ray_dir = nodes[ucenter] - nodes[ray_prev]
        rn = float(np.linalg.norm(ray_dir))
        if rn < 1e-9:
            continue
        ray_dir = ray_dir / rn
        brc_dir = nodes[bi] - nodes[ucenter]
        bn = float(np.linalg.norm(brc_dir))
        if bn < 1e-9:
            continue
        brc_dir = brc_dir / bn
        cos_a = float(np.clip(ray_dir @ brc_dir, -1.0, 1.0))
        bracteole_angles.append(float(np.degrees(np.arccos(cos_a))))
    mean_bracteole_length = float(np.mean(bracteole_lens)) if bracteole_lens else 0.0
    mean_bracteole_reflex_angle = float(np.mean(bracteole_angles)) if bracteole_angles else 0.0

    # mean pedicel length (for ratio against bracteoles)
    pedicel_lens: list[float] = []
    for i, r in enumerate(role):
        if r != "pedicel-tip" or not adj[i]:
            continue
        parent = next(iter(adj[i]))
        pedicel_lens.append(_edge_length(skel, i, parent))
    mean_pedicel_length = float(np.mean(pedicel_lens)) if pedicel_lens else 0.0
    bracteole_to_pedicel_length_ratio = (
        mean_bracteole_length / mean_pedicel_length
        if mean_pedicel_length > 0 and mean_bracteole_length > 0 else 0.0
    )

    return np.array([
        n_nodes, n_edges, n_leaves, n_junctions, n_stem_nodes,
        n_branch_nodes, max_degree, mean_degree,
        total_length, stem_length, leaf_length_total, leaf_length_mean, leaf_length_std,
        leaves_per_stem_unit, branching_factor,
        bbox_width, bbox_depth, bbox_height, aspect_ratio_hw, stem_height_fraction,
        n_bracts, n_bracteoles, bracteoles_per_umbellet,
        mean_bract_length, mean_bracteole_length, mean_bracteole_reflex_angle,
        bracteole_to_pedicel_length_ratio,
    ], dtype=np.float32)
