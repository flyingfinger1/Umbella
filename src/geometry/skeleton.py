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
    edges: list[tuple[int, int]]            # undirected
    node_organ: list[int]                   # per-node organ id (1=stem, >=2=leaf instance)
    node_role: list[str] = field(default_factory=list)
    # roles in current pipeline: "stem", "stem-junction", "leaf-base", "leaf-mid", "leaf-tip"
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    def to_dict(self) -> dict:
        return {
            "schema": "umbella.skeleton.v1",
            "metadata": self.metadata,
            "nodes": self.nodes.round(4).tolist(),
            "edges": [list(e) for e in self.edges],
            "node_organ": list(self.node_organ),
            "node_role": list(self.node_role),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        return cls(
            nodes=np.asarray(d["nodes"], dtype=np.float32),
            edges=[tuple(e) for e in d["edges"]],
            node_organ=list(d["node_organ"]),
            node_role=list(d["node_role"]),
            metadata=dict(d.get("metadata", {})),
        )

    def save_json(self, path) -> None:
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path) -> "Skeleton":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


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
    anchor_to: np.ndarray | None = None,
) -> np.ndarray:
    """Geodesic mid-axis polyline via shortest path on a kNN graph.

    Robust to curvature (a bent organ no longer collapses onto a straight PCA
    axis).

    Endpoint selection depends on `anchor_to`:
      - If None: both endpoints are PCA min/max projections of the cloud.
        Works fine for tube-like organs where the PCA axis aligns with the
        physical centerline (e.g. a stem segment).
      - If provided (e.g. the stem point cloud when extracting a leaf):
        the *base* endpoint is forced to be the cloud point closest to any
        anchor point. The *tip* endpoint is the geodesically farthest
        reachable point from the base. This handles broad/flat organs (leaves
        with thin petioles) where PCA would otherwise miss the petiole and
        place the base in the leaf's interior.
        Returned polyline is oriented base -> tip; nodes[0] is guaranteed
        to be on the cloud point closest to `anchor_to`.

    Steps:
      1. voxel-downsample (auto-pick voxel size from organ bbox if not given)
      2. choose endpoints (PCA or anchor-based, see above)
      3. build kNN graph + Dijkstra between the endpoints
      4. resample the path to `n_nodes` equidistant nodes
      5. light smoothing (endpoints preserved)
    """
    if points.shape[0] < max(n_nodes, 4):
        return points.astype(np.float32)

    bbox_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if voxel is None:
        # ~80 voxels along the longest dim — gives ~2-5k surviving points for typical organs
        voxel = max(bbox_diag / 80.0, 1e-3)

    ds = _voxel_downsample(points, voxel)
    if ds.shape[0] < max(n_nodes, 4):
        ds = points.astype(np.float32)

    g = _knn_graph(ds, k=k_neighbors)

    if anchor_to is not None and anchor_to.shape[0] > 0:
        # base = downsampled point with smallest distance to any anchor point
        anchor_tree = cKDTree(anchor_to)
        d_to_anchor, _ = anchor_tree.query(ds, k=1)
        a = int(np.argmin(d_to_anchor))
        # tip = farthest geodesically reachable point from base
        dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)
        if not np.isfinite(dists).any() or np.isfinite(dists).sum() < 2:
            g = _knn_graph(ds, k=k_neighbors * 3)
            dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)
        finite_dists = np.where(np.isfinite(dists), dists, -np.inf)
        b = int(np.argmax(finite_dists))
        if not np.isfinite(dists[b]) or dists[b] <= 0:
            return ds[[a]].astype(np.float32)
    else:
        a, b = _farthest_extremes(ds)
        dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)
        if not np.isfinite(dists[b]):
            g = _knn_graph(ds, k=k_neighbors * 3)
            dists, predecessors = dijkstra(g, indices=a, return_predecessors=True)
            if not np.isfinite(dists[b]):
                return _resample_polyline(np.stack([ds[a], ds[b]]), n_nodes).astype(np.float32)

    path = [b]
    while path[-1] != a:
        p = int(predecessors[path[-1]])
        if p < 0:
            break
        path.append(p)
    path.reverse()
    path_xyz = ds[path]

    nodes = _resample_polyline(path_xyz, n_nodes)
    nodes = _smooth_polyline(nodes, passes=2)

    if anchor_to is None and nodes[0, 2] > nodes[-1, 2]:
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


def _attach_point_to_tree(
    nodes_list: list[np.ndarray],
    organs: list[int],
    roles: list[str],
    edges: list[tuple[int, int]],
    candidate_node_indices: set[int],
    point: np.ndarray,
    junction_organ_id: int = 1,
    junction_role: str = "stem-junction",
    eps_mm: float = 0.5,
) -> tuple[int, bool]:
    """Find the closest point on any edge among `candidate_node_indices` to `point`.
    If it lies near an existing endpoint, return that node id (was_new=False).
    Otherwise, insert a new node at the projected position and split the edge
    into two; return the new node id (was_new=True).

    Mutates `nodes_list`, `organs`, `roles`, and `edges` in place.
    """
    cand = candidate_node_indices
    best_dist = float("inf")
    best_edge_idx = -1
    best_t = 0.0
    best_a = best_b = -1
    best_proj: np.ndarray | None = None

    for edge_idx, (a, b) in enumerate(edges):
        if a not in cand or b not in cand:
            continue
        pa = nodes_list[a]
        pb = nodes_list[b]
        ab = pb - pa
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            continue
        t = float((point - pa) @ ab) / L2
        t = max(0.0, min(1.0, t))
        proj = pa + t * ab
        d = float(np.linalg.norm(point - proj))
        if d < best_dist:
            best_dist = d
            best_t = t
            best_edge_idx = edge_idx
            best_a, best_b = a, b
            best_proj = proj

    if best_edge_idx < 0:
        # no eligible edge (e.g. stem skeleton degenerate) -> nearest candidate node
        cand_list = sorted(cand)
        cand_xyz = np.stack([nodes_list[i] for i in cand_list])
        return cand_list[int(np.argmin(np.linalg.norm(cand_xyz - point, axis=1)))], False

    pa = nodes_list[best_a]
    pb = nodes_list[best_b]
    seg_len = float(np.linalg.norm(pb - pa))
    if best_t * seg_len < eps_mm:
        return best_a, False
    if (1.0 - best_t) * seg_len < eps_mm:
        return best_b, False

    new_idx = len(nodes_list)
    nodes_list.append(best_proj.astype(np.float32))
    organs.append(junction_organ_id)
    roles.append(junction_role)
    edges.pop(best_edge_idx)
    edges.append((best_a, new_idx))
    edges.append((new_idx, best_b))
    return new_idx, True


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
    stem_xyz_full = xyz[stem_mask] if stem_mask.any() else None
    # live set of stem skeleton node indices (grows when we split edges to insert junctions)
    stem_node_set: set[int] = set(range(stem_idx_start, stem_idx_end))
    for lid in sorted(int(x) for x in np.unique(instance) if x >= 2):
        leaf_mask = instance == lid
        if leaf_mask.sum() < min_organ_points:
            continue
        # anchor base to the stem point cloud — handles broad/lobed leaves where
        # PCA would otherwise place the base inside the leaf blade
        leaf_poly = extract_polyline(
            xyz[leaf_mask],
            n_nodes=leaf_nodes,
            anchor_to=stem_xyz_full,
        )

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

        # connect base to the closest point ON the stem skeleton (split edge if needed)
        if stem_node_set:
            parent_idx, was_new = _attach_point_to_tree(
                nodes_list, organs, roles, edges,
                candidate_node_indices=stem_node_set,
                point=leaf_poly[0],
            )
            if was_new:
                stem_node_set.add(parent_idx)
            edges.append((parent_idx, start))

    nodes = np.stack(nodes_list, axis=0).astype(np.float32) if nodes_list else np.zeros((0, 3), np.float32)
    return Skeleton(nodes=nodes, edges=edges, node_organ=organs, node_role=roles)
