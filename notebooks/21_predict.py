"""21: Single-image species prediction from a trained checkpoint.

Usage:
    .venv/Scripts/python.exe notebooks/21_predict.py path/to/photo.jpg
    .venv/Scripts/python.exe notebooks/21_predict.py path/to/photo.jpg --model data/models/v5_classifier.pt
    .venv/Scripts/python.exe notebooks/21_predict.py path/to/folder/    # batch over a directory

Default model: data/models/v6_classifier.pt (the augmented one — best for real photos).
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_model, predict_image_path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data" / "models" / "v6_classifier.pt"


def report(path: Path, result: dict) -> None:
    print(f"\n{path.name}")
    print(f"  top-1: {result['top1_german']:>22s} ({result['top1_species']:>22s})  "
          f"conf {result['top1_confidence'] * 100:5.1f}%")
    if len(result["top_k"]) > 1:
        for s, g, p in result["top_k"][1:]:
            print(f"     {g:>22s} ({s:>22s})  conf {p * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="image file or directory of images")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="model checkpoint path (default: data/models/v6_classifier.pt)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"checkpoint not found: {model_path}")
    print(f"loading model: {model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path}")
    model = load_model(model_path)

    p = Path(args.path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
        if not images:
            sys.exit(f"no images found in {p}")
        print(f"running on {len(images)} images")
        for img_path in images:
            try:
                result = predict_image_path(model, img_path, top_k=args.top_k)
                report(img_path, result)
            except Exception as e:
                print(f"\n{img_path.name}\n  ERROR: {e}")
    else:
        if not p.exists():
            sys.exit(f"image not found: {p}")
        result = predict_image_path(model, p, top_k=args.top_k)
        report(p, result)


if __name__ == "__main__":
    main()
