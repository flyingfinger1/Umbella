"""15: Build the first synthetic training dataset.

50 instances per species × 6 species × 4 views = 1200 image-skeleton pairs.
At 384x384 with current density that's ~250 MB on disk.

Usage:
    .venv/Scripts/python.exe notebooks/15_build_training_set.py
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import build_dataset, list_examples, load_example   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "training" / "v5"


def main() -> None:
    t0 = time.time()
    print(f"building dataset at {OUT.relative_to(ROOT)}/")
    meta = build_dataset(
        out_root=OUT,
        n_per_species=200,
        n_views=4,
        image_size=(384, 384),
    )
    dt = time.time() - t0
    print(f"\nfinished {len(meta['examples'])} examples in {dt:.1f}s")

    # disk size
    total_bytes = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    n_files = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"  disk: {total_bytes / 1e6:.1f} MB across {n_files} files")

    # sanity: load one example back and report shapes
    examples = list_examples(OUT)
    sample = load_example(OUT, examples[0])
    print(f"\nexample[0]: {examples[0]['species']} seed{examples[0]['seed']} view{examples[0]['view']}")
    print(f"  rgb:   {sample['rgb'].shape} {sample['rgb'].dtype}")
    print(f"  label: {sample['label'].shape} {sample['label'].dtype}  unique={len(set(sample['label'].ravel().tolist()))} ids")
    print(f"  depth: {sample['depth'].shape} {sample['depth'].dtype}  fg_px={(sample['depth'] > 0).sum()}")
    print(f"  camera: {sample['camera']}")
    print(f"  azimuth: {sample['azimuth_deg']}°")

    # per-species count check
    per_species = {}
    for e in examples:
        per_species[e["species"]] = per_species.get(e["species"], 0) + 1
    print("\nexamples per species:")
    for s, n in per_species.items():
        print(f"  {s:>22s}: {n}")


if __name__ == "__main__":
    main()
