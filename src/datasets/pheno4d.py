"""Pheno4D loader.

Pheno4D: 7 maize + 7 tomato plants, scanned over 2-3 weeks.
Files are space-separated text:
  - "<id>_<date>.txt"     -> 3 cols (x y z), unannotated
  - "<id>_<date>_a.txt"   -> annotated, but column count differs by species:
        * Tomato: 4 cols  -> x y z combined_label
        * Maize:  5 cols  -> x y z aux combined_label

`combined_label` semantics (verified empirically on the data):
    0     = soil
    1     = stem
    >= 2  = leaf instance id (each individual leaf gets a unique id)

We expose this as `semantic` (mapped to {0=soil, 1=stem, 2=leaf}) plus
`instance` (the raw combined label, useful for per-leaf segmentation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "soil", 1: "stem", 2: "leaf"}


@dataclass
class Pheno4DCloud:
    plant_id: str          # e.g. "Maize01"
    date: str              # e.g. "0313"
    annotated: bool
    xyz: np.ndarray        # (N, 3) float32
    instance: np.ndarray | None  # (N,) int32 or None
    semantic: np.ndarray | None  # (N,) int32 or None
    source_path: Path

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]

    def filter_class(self, class_id: int) -> "Pheno4DCloud":
        if not self.annotated:
            raise ValueError("Cannot filter class on unannotated cloud")
        mask = self.semantic == class_id
        return Pheno4DCloud(
            plant_id=self.plant_id,
            date=self.date,
            annotated=True,
            xyz=self.xyz[mask],
            instance=self.instance[mask],
            semantic=self.semantic[mask],
            source_path=self.source_path,
        )


class Pheno4D:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Pheno4D root not found: {self.root}")

    def plants(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def files(self, plant_id: str, annotated_only: bool = False) -> list[Path]:
        plant_dir = self.root / plant_id
        if not plant_dir.is_dir():
            raise FileNotFoundError(plant_dir)
        files = sorted(plant_dir.glob("*.txt"))
        if annotated_only:
            files = [f for f in files if f.stem.endswith("_a")]
        return files

    def load(self, path: str | Path) -> Pheno4DCloud:
        path = Path(path)
        plant_id = path.parent.name
        # filename like "M01_0313_a" or "M01_0314"
        stem = path.stem
        annotated = stem.endswith("_a")
        date = stem.split("_")[1]

        data = np.loadtxt(path, dtype=np.float32)
        xyz = data[:, :3]
        if annotated:
            # last column is the combined label (works for both 4-col tomato and 5-col maize)
            combined = data[:, -1].astype(np.int32)
            instance = combined
            semantic = np.where(combined >= 2, np.int32(2), combined).astype(np.int32)
        else:
            instance = None
            semantic = None

        return Pheno4DCloud(
            plant_id=plant_id,
            date=date,
            annotated=annotated,
            xyz=xyz,
            instance=instance,
            semantic=semantic,
            source_path=path,
        )
