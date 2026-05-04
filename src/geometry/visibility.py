"""Hidden Point Removal (HPR) via spherical flipping (Katz, Tal, Basri 2007).

Given a point cloud and a virtual camera position, return only the points
that would be visible from that camera. Used to make synthetic full-cylinder
samples look like one-sided LiDAR / photogrammetry scans.

Algorithm:
  1. translate points so camera is at origin
  2. spherical flip:  p' = p + 2*(R - |p|) * (p / |p|), with R > max |p|
  3. compute convex hull of the flipped points + origin
  4. points whose flipped image is on the hull are visible
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def hpr_visible_indices(
    points: np.ndarray,
    camera: np.ndarray,
    radius_factor: float = 1000.0,
) -> np.ndarray:
    """Return indices of points visible from `camera` via Katz et al. HPR.

    `radius_factor` scales the inversion radius beyond the cloud extent
    (corresponds to 10^gamma in the paper). Empirical Heracleum sweep on
    synthetic Apiaceae:
        factor=10   ->  0.5%  retention  (very strict)
        factor=100  ->  2.0%
        factor=1000 ->  7.5%   <- closest to realistic single-view photogrammetry
        factor=10000-> 20%+
    Counter-intuitive: larger factor keeps MORE points (the inversion makes
    the cloud "smaller" relative to the camera, so more points end up as
    hull vertices). Default 1000 is a reasonable photogrammetry-like setting.
    """
    p = points.astype(np.float64) - np.asarray(camera, dtype=np.float64)
    norms = np.linalg.norm(p, axis=1)
    nonzero = norms > 1e-9
    if nonzero.sum() < 4:
        return np.flatnonzero(nonzero)

    R = float(norms[nonzero].max()) * radius_factor
    flipped = np.empty_like(p)
    flipped[nonzero] = p[nonzero] + 2.0 * (R - norms[nonzero])[:, None] * (
        p[nonzero] / norms[nonzero, None]
    )
    flipped[~nonzero] = 0.0  # at-camera points: keep visible by default

    # add origin so the hull "wraps around" the camera
    pts_aug = np.vstack([flipped, np.zeros((1, 3))])
    try:
        hull = ConvexHull(pts_aug, qhull_options="QJ")
    except Exception:
        return np.arange(points.shape[0])

    visible = np.unique(hull.vertices)
    visible = visible[visible < points.shape[0]]
    return visible


def hpr_multi_view(
    points: np.ndarray,
    cameras: list[np.ndarray] | np.ndarray,
    radius_factor: float = 100.0,
) -> np.ndarray:
    """Union of visibility from several cameras."""
    keep = np.zeros(points.shape[0], dtype=bool)
    for cam in cameras:
        idx = hpr_visible_indices(points, np.asarray(cam), radius_factor)
        keep[idx] = True
    return np.flatnonzero(keep)


def camera_around(
    target: np.ndarray,
    bbox_diag: float,
    azimuth_deg: float,
    elevation_deg: float = 10.0,
    distance_factor: float = 2.0,
) -> np.ndarray:
    """Place a camera on a sphere around `target` at the given direction.

    azimuth_deg: 0 = +x axis, increases counter-clockwise around z
    elevation_deg: angle above horizontal
    distance_factor: camera distance = factor * bbox_diag
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    dist = distance_factor * bbox_diag
    direction = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    return target + dist * direction
