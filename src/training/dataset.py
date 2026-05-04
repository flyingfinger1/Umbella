"""Build and load image-skeleton training pairs from synthetic Apiaceae.

Output layout under `out_root`:
    skeletons/<species>/seed{N:04d}.json    # one JSON per instance
    images/<species>/seed{N:04d}_view{V}.npz # rgb, label, depth, camera, target
    metadata.json                            # flat list of all (skel, image) pairs

`build_dataset` is deterministic for a given seed; re-running with the same
parameters reproduces byte-identical files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from src.synthetic import SPECIES, generate_apiaceae
from src.geometry import (
    sample_skeleton_pointcloud,
    hpr_visible_indices,
    camera_around,
    render_pointcloud,
    role_aware_color_callable,
    augment_render,
)


def build_dataset(
    out_root: Path | str,
    n_per_species: int = 30,
    n_views: int = 4,
    image_size: tuple[int, int] = (384, 384),
    points_per_mm2: float = 2.5,
    noise_mm: float = 0.3,
    fov_deg: float = 40.0,
    elevation_deg: float = 10.0,
    distance_factor: float = 1.5,
    point_radius_px: int = 2,
    augment: bool = False,
    species_keys: Iterable[str] | None = None,
    progress: bool = True,
) -> dict:
    """Generate a synthetic training dataset and write it to `out_root`.

    Returns a metadata dict (also written to <out_root>/metadata.json).
    """
    out_root = Path(out_root)
    skel_dir = out_root / "skeletons"
    img_dir = out_root / "images"
    skel_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    keys = list(species_keys) if species_keys else list(SPECIES.keys())

    examples = []
    t_start = time.time()
    total = len(keys) * n_per_species * n_views
    counter = 0

    for key in keys:
        spec = SPECIES[key]
        (skel_dir / key).mkdir(exist_ok=True)
        (img_dir / key).mkdir(exist_ok=True)

        for seed in range(n_per_species):
            # one skeleton + cloud per instance
            params = spec.sample(seed=seed)
            skel = generate_apiaceae(params)
            skel.metadata["species"] = key
            skel.metadata["seed"] = int(seed)
            skel_path = skel_dir / key / f"seed{seed:04d}.json"
            skel.save_json(skel_path)

            xyz, lbl = sample_skeleton_pointcloud(
                skel, points_per_mm2=points_per_mm2, noise_mm=noise_mm, seed=seed,
            )
            target = xyz.mean(0).astype(np.float32)
            bbox_diag = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
            # role-aware coloring so bracts/bracteoles/pedicels render with
            # distinct colors regardless of organ-id numbering
            organ_to_role = {}
            for o, r in zip(skel.node_organ, skel.node_role):
                if o not in organ_to_role:
                    organ_to_role[o] = r

            # species-specific stem speckles (Conium diagnostic)
            speckle_rng = np.random.default_rng(seed * 1000 + 7)
            density = float(spec.stem_speckle_density.sample(speckle_rng))
            if density > 0:
                stem_idx = np.flatnonzero(lbl == 1)
                n_speckle = int(round(density * len(stem_idx)))
                if n_speckle > 0:
                    chosen = speckle_rng.choice(stem_idx, n_speckle, replace=False)
                    speckle_label = int(lbl.max()) + 1
                    lbl = lbl.copy()
                    lbl[chosen] = speckle_label
                    organ_to_role[speckle_label] = "stem-speckle"

            color_fn = role_aware_color_callable(organ_to_role)

            for view in range(n_views):
                az = view * 360.0 / n_views
                cam = camera_around(target.astype(np.float64), bbox_diag,
                                    azimuth_deg=az, elevation_deg=elevation_deg,
                                    distance_factor=distance_factor).astype(np.float32)
                visible = hpr_visible_indices(xyz, cam, radius_factor=1000.0)
                rgb, label_map, depth = render_pointcloud(
                    xyz[visible], lbl[visible],
                    camera_pos=cam, target=target,
                    image_size=image_size, fov_deg=fov_deg,
                    point_radius_px=point_radius_px,
                    label_to_color=color_fn,
                )
                if augment:
                    aug_seed = seed * 100_000 + view * 17 + 31
                    rgb = augment_render(rgb, label_map, depth, seed=aug_seed)
                # store inf as nan so npz can compress + tag bg via label==0
                depth_to_save = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
                img_path = img_dir / key / f"seed{seed:04d}_view{view}.npz"
                np.savez_compressed(
                    img_path,
                    rgb=rgb,
                    label=label_map.astype(np.int32),
                    depth=depth_to_save,
                    camera=cam,
                    target=target,
                    azimuth_deg=np.float32(az),
                    elevation_deg=np.float32(elevation_deg),
                )

                examples.append({
                    "species": key,
                    "seed": int(seed),
                    "view": int(view),
                    "azimuth_deg": float(az),
                    "skeleton": str(skel_path.relative_to(out_root)).replace("\\", "/"),
                    "image": str(img_path.relative_to(out_root)).replace("\\", "/"),
                })

                counter += 1
                if progress and counter % 50 == 0:
                    pct = 100 * counter / total
                    elapsed = time.time() - t_start
                    eta = elapsed * (total - counter) / max(counter, 1)
                    print(f"  [{counter:>5d}/{total}]  {pct:5.1f}%  "
                          f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s")

    metadata = {
        "schema": "umbella.training.v1",
        "image_size": list(image_size),
        "n_views_per_instance": int(n_views),
        "n_per_species": int(n_per_species),
        "species_keys": keys,
        "render_params": {
            "points_per_mm2": points_per_mm2,
            "noise_mm": noise_mm,
            "fov_deg": fov_deg,
            "elevation_deg": elevation_deg,
            "distance_factor": distance_factor,
            "point_radius_px": point_radius_px,
            "augment": bool(augment),
        },
        "examples": examples,
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return metadata


def list_examples(out_root: Path | str) -> list[dict]:
    out_root = Path(out_root)
    with open(out_root / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)["examples"]


def load_example(out_root: Path | str, example: dict) -> dict:
    """Load one example's image-side data + return path to its skeleton."""
    out_root = Path(out_root)
    img_path = out_root / example["image"]
    skel_path = out_root / example["skeleton"]
    data = np.load(img_path)
    return {
        "rgb": data["rgb"],
        "label": data["label"],
        "depth": data["depth"],
        "camera": data["camera"],
        "target": data["target"],
        "azimuth_deg": float(data["azimuth_deg"]),
        "elevation_deg": float(data["elevation_deg"]),
        "skeleton_path": skel_path,
        "species": example["species"],
        "seed": example["seed"],
        "view": example["view"],
    }
