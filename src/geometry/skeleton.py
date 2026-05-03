"""Lightweight skeleton extraction from labeled plant point clouds.

Approach (single-pass, no external solvers):
  - For each organ point cloud (stem or leaf), compute PCA. The first
    principal component is the dominant axis (length).
  - Project the points onto that axis, split into K equal bins, take the
    median of the points in each bin. Those medians form a polyline.
  - For a plant: extract one polyline per organ, then connect each leaf's
    base node (the bin endpoint nearest the stem polyline) to its closest
    stem node.

This is intentionally simple — it works well for tube-like organs
(maize/tomato leaves and stems) and gives us a concrete graph to feed
into downstream visualization, classifier inputs, or evaluation against
synthetic ground truth. More sophisticated methods (L1-medial skeleton,
graph contraction) can replace this later without changing the data
structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class Skeleton:
    nodes: np.ndarray                       # (N, 3) float32 xyz
    edges: list[tuple[int, int]]            # undirected, (i, j) with i<j
    node_organ: list[int]                   # per-node organ id (1=stem, >=2=leaf instance)
    node_role: list[str] = field(default_factory=list)  # "stem", "leaf-base", "leaf-tip", "leaf-mid"

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """Average points within voxel-sized cubes. Cheap, deterministic, no deps."""
    keys = np.floor(points / voxel).astype(np.int64)
    # encode (i,j,k) -> single int via lex order; since the cloud is bounded
    # this fits comfortably in int64
    mins = keys.min(axis=0)
    keys -= mins
    span = keys.max(axis=0) + 1
    flat = keys[:, 0] * span[1] * span[2] + keys[:, 1] * span[2] + keys[:, 2]
    order = np.argsort(flat)
    flat_sorted = flat[order]
    pts_sorted = points[order]
    # boundaries between groups
    breaks = np.flatnonzero(np.diff(flat_sorted)) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(flat_sorted)]
    means = np.add.reduceat(pts_sorted, starts, axis=0) / (ends - starts)[:, None]
    return means.astype(np.float32)


def _farthest_extremes(points: np.ndarray) -> tuple[int, int]:
    """Two endpoint indices: take min/max projection along PCA's first axis."""
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh[0]
    return int(np.argmin(proj)), int(np.argmax(proj))


def _knn_graph(points: np.ndarray, k: int) -> csr_matrix:
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k + 1)            # +1 because self is included
    rows = np.repeat(np.arange(points.shape[0]), k)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel()
    n = points.shape[0]
    g = csr_matrix((data, (rows, cols)), shape=(n, n))
    # symmetrize (kNN is asymmetric in general)
    return g.maximum(g.T)


def _resample_polyline(path_xyz: np.ndarray, n_nodes: int) -> np.ndarray:
    """Resample a polyline at n_nodes equidistant arc-length positions."""
    if len(path_xyz) <= n_nodes:
        return path_xyz
    seg = np.linalg.norm(np.diff(path_xyz, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    targets = np.linspace(0, arc[-1], n_nodes)
    out = np.empty((n_nodes, 3), dtype=path_xyz.dtype)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(arc) and arc[j + 1] < t:
            j += 1
        if j + 1 >= len(arc):
            out[i] = path_xyz[-1]
            continue
        span = arc[j + 1] - arc[j]
        u = 0.0 if span == 0 else (t - arc[j]) / span
        out[i] = path_xyz[j] * (1 - u) + path_xyz[j + 1] * u
    return out


def _smooth_polyline(nodes: np.ndarray, passes: int = 2) -> np.ndarray:
    """Endpoint-preserving 3-tap moving average."""
    if len(nodes) < 3:
        return nodes
    out = nodes.copy()
    for _ in range(passes):
        nxt = out.copy()
        nxt[1:-1] = (out[:-2] + out[1:-1] + out[2:]) / 3.0
        out = nxt
    return out


def extract_polyline(
    points: np.ndarray,
    n_nodes: int = 10,
    voxel: float | None = None,
    k_neighbors: int = 8,
) -> np.ndarray:
    """Geodesic mid-axis polyline via shortest path on a kNN graph.

    Robust to curvature (a bent leaf no longer collapses onto a straight PCA
    axis). Steps:
      1. voxel-downsample (auto-pick voxel size from organ bbox if not given)
      2. find two extremes via PCA min/max projection
      3. build kNN graph + Dijkstra between the extremes
      4. resample the path to `n_nodes` equidistant nodes
      5. light smoothing
    """
    if points.shape[0] < max(n_nodes, 4):
        return points.astype(np.float32)

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if voxel is None:
        # ~50 voxels along the longest dim — gives ~2-5k surviving points for typical organs
        voxel = max(bbox_diag / 80.0, 1e-3)

    ds = _voxel_downsample(points, voxel)
    if ds.shape[0] < max(n_nodes, 4):
        # voxel too coarse — fall back to original
        ds = points.astype(np.float32)

    a, b = _farthest_extremes(ds)
    g = _knn_graph(ds, k=k_neighbors)
    dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)

    if not np.isfinite(dists[b]):
        # graph disconnected (organ split into pieces by downsampling) -> bigger k
        g = _knn_graph(ds, k=k_neighbors * 3)
        dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)
        if not np.isfinite(dists[b]):
            # give up, use raw extreme line
            return _resample_polyline(np.stack([ds[a], ds[b]]), n_nodes).astype(np.float32)

    path = [b]
    while path[-1] != a:
        path.append(int(predecessors[path[-1]]))
    path.reverse()
    path_xyz = ds[path]

    nodes = _resample_polyline(path_xyz, n_nodes)
    nodes = _smooth_polyline(nodes, passes=2)

    if nodes[0, 2] > nodes[-1, 2]:
        nodes = nodes[::-1]
    return nodes.astype(np.float32)


def extract_branched_skeleton(
    points: np.ndarray,
    n_branches: int = 4,
    voxel: float | None = None,
    k_neighbors: int = 8,
    min_branch_len_ratio: float = 0.05,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Greedy farthest-point Steiner tree on a kNN graph.

    Returns (nodes, edges) where:
      - nodes: (M, 3) float32, only points actually used in the tree
      - edges: list of (i, j) tuples, undirected tree edges

    Stops early if the next branch would add less than `min_branch_len_ratio`
    times the first (longest) branch length.
    """
    if points.shape[0] < 4:
        return points.astype(np.float32), []

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if voxel is None:
        voxel = max(bbox_diag / 80.0, 1e-3)

    ds = _voxel_downsample(points, voxel)
    if ds.shape[0] < 4:
        ds = points.astype(np.float32)

    g = _knn_graph(ds, k=k_neighbors)

    # ensure graph is connected enough; widen k if necessary
    root = int(np.argmin(ds[:, 2]))
    dists0, _ = dijkstra(g, indices=root, return_predecessors=True)
    if not np.isfinite(dists0).all():
        g = _knn_graph(ds, k=k_neighbors * 3)

    # greedy multi-source Dijkstra
    covered_mask = np.zeros(ds.shape[0], dtype=bool)
    covered_mask[root] = True
    tree_edges_local: set[tuple[int, int]] = set()

    first_branch_len = None
    for _ in range(n_branches):
        sources = np.flatnonzero(covered_mask)
        d, pred, _src_for = dijkstra(g, indices=sources, return_predecessors=True, min_only=True)
        # ignore already-covered points
        d_cand = np.where(covered_mask, -np.inf, d)
        d_cand = np.where(np.isfinite(d_cand), d_cand, -np.inf)
        new_end = int(np.argmax(d_cand))
        if d_cand[new_end] <= 0:
            break
        if first_branch_len is None:
            first_branch_len = d_cand[new_end]
        elif d_cand[new_end] < min_branch_len_ratio * first_branch_len:
            break

        # trace path from new_end back to its source via predecessor array
        # `pred` from min_only stores predecessor in the multi-source SP tree;
        # walking back lands on one of the sources.
        path = [new_end]
        while True:
            p = int(pred[path[-1]])
            if p < 0:
                break
            path.append(p)
            if covered_mask[p]:
                break
        # add edges and mark nodes
        for a, b in zip(path[:-1], path[1:]):
            i, j = (a, b) if a < b else (b, a)
            tree_edges_local.add((i, j))
            covered_mask[a] = True
            covered_mask[b] = True

    if not tree_edges_local:
        return ds[[root]].astype(np.float32), []

    # remap to compact node ids
    used = sorted({i for e in tree_edges_local for i in e})
    remap = {old: new for new, old in enumerate(used)}
    nodes = ds[used].astype(np.float32)
    edges = [(remap[a], remap[b]) for a, b in tree_edges_local]
    return nodes, edges


def extract_plant_skeleton(
    xyz: np.ndarray,
    instance: np.ndarray,
    stem_branches: int = 4,
    stem_voxel: float | None = None,
    leaf_nodes: int = 6,
    min_organ_points: int = 50,
    stem_nodes: int | None = None,        # legacy: if set, use single polyline
) -> Skeleton:
    """Build a skeleton for a full plant.

    `instance` follows Pheno4D convention: 0=soil, 1=stem, >=2=leaf instance.
    Soil points are ignored.
    """
    nodes_list: list[np.ndarray] = []
    organs: list[int] = []
    roles: list[str] = []
    edges: list[tuple[int, int]] = []

    # 1. stem skeleton — branched tree by default, fall back to polyline if requested
    stem_mask = instance == 1
    stem_idx_start = len(nodes_list)
    if stem_mask.sum() >= min_organ_points:
        if stem_nodes is not None:
            stem_pts = extract_polyline(xyz[stem_mask], n_nodes=stem_nodes)
            stem_eds = [(i, i + 1) for i in range(len(stem_pts) - 1)]
        else:
            stem_pts, stem_eds = extract_branched_skeleton(
                xyz[stem_mask], n_branches=stem_branches, voxel=stem_voxel,
            )
        for p in stem_pts:
            nodes_list.append(p)
            organs.append(1)
            roles.append("stem")
        for a, b in stem_eds:
            edges.append((stem_idx_start + a, stem_idx_start + b))
    stem_idx_end = len(nodes_list)

    stem_node_xyz = (
        np.stack(nodes_list[stem_idx_start:stem_idx_end], axis=0)
        if stem_idx_end > stem_idx_start else None
    )
    stem_poly_idx_range = (stem_idx_start, stem_idx_end)

    # 2. leaves
    for lid in sorted(int(x) for x in np.unique(instance) if x >= 2):
        leaf_mask = instance == lid
        if leaf_mask.sum() < min_organ_points:
            continue
        leaf_poly = extract_polyline(xyz[leaf_mask], n_nodes=leaf_nodes)

        # decide which end of the polyline is the base: closer to stem
        if stem_node_xyz is not None:
            d_first = np.min(np.linalg.norm(stem_node_xyz - leaf_poly[0], axis=1))
            d_last = np.min(np.linalg.norm(stem_node_xyz - leaf_poly[-1], axis=1))
            if d_last < d_first:
                leaf_poly = leaf_poly[::-1]

        start = len(nodes_list)
        for i, p in enumerate(leaf_poly):
            nodes_list.append(p)
            organs.append(lid)
            if i == 0:
                roles.append("leaf-base")
            elif i == len(leaf_poly) - 1:
                roles.append("leaf-tip")
            else:
                roles.append("leaf-mid")
        for i in range(start, start + len(leaf_poly) - 1):
            edges.append((i, i + 1))

        # connect base to nearest stem node
        if stem_node_xyz is not None:
            base_xyz = leaf_poly[0]
            nearest_stem_local = int(np.argmin(np.linalg.norm(stem_node_xyz - base_xyz, axis=1)))
            nearest_stem_global = stem_poly_idx_range[0] + nearest_stem_local
            edges.append((nearest_stem_global, start))

    nodes = np.stack(nodes_list, axis=0).astype(np.float32) if nodes_list else np.zeros((0, 3), np.float32)
    return Skeleton(nodes=nodes, edges=edges, node_organ=organs, node_role=roles)
