"""28: Evaluate v9 on the 9 real Apiaceae photos vs v7/v8/leaf/hybrid baselines.

Same photo set as notebook 26. Counts Anthriscus_sylvestris top-1 and top-3
hits per strategy, plus prints per-photo top-1 with Anthriscus rank.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import (
    load_model, predict_image_path,
    predict_leaf_image_path, predict_hybrid,
)
from src.leaf import load_leaf_classifier

ROOT = Path(__file__).resolve().parents[1]


def anth_rank(top_k):
    for i, (s, _, p) in enumerate(top_k, start=1):
        if s == "Anthriscus_sylvestris":
            return i, p
    return None, 0.0


def main() -> None:
    v7 = load_model(ROOT / "data" / "models" / "v7_classifier.pt")
    v8 = load_model(ROOT / "data" / "models" / "v8_classifier.pt")
    v9 = load_model(ROOT / "data" / "models" / "v9_classifier.pt")
    leaf = load_leaf_classifier(ROOT / "data" / "models" / "leaf_classifier.pt")

    photos = [
        ("Stockfoto",          r"C:\Users\teign\Downloads\test.jpg"),
        ("INDOOR  Übersicht",  r"C:\Users\teign\Downloads\photo_5460693769318504232_y.jpg"),
        ("INDOOR  Doldenkopf", r"C:\Users\teign\Downloads\photo_5460693769318504233_y.jpg"),
        ("INDOOR  Blatt",       r"C:\Users\teign\Downloads\photo_5460693769318504234_y.jpg"),
        ("INDOOR  Stamm",       r"C:\Users\teign\Downloads\photo_5460693769318504235_y.jpg"),
        ("OUTDOOR Übersicht",  r"C:\Users\teign\Downloads\photo_5460693769318504309_y.jpg"),
        ("OUTDOOR Doldenkopf", r"C:\Users\teign\Downloads\photo_5460693769318504307_y.jpg"),
        ("OUTDOOR Blatt",       r"C:\Users\teign\Downloads\photo_5460693769318504306_y.jpg"),
        ("OUTDOOR Stamm",       r"C:\Users\teign\Downloads\photo_5460693769318504308_y.jpg"),
    ]

    strategies = ["v7", "v8", "v9", "leaf",
                  "v9+leaf soft50", "v9+leaf soft30/70", "v9+leaf conf"]
    counts = {k: [0, 0] for k in strategies}

    def cell(res):
        t1 = res["top_k"][0]
        r, p = anth_rank(res["top_k"])
        anth = f"  Anth#{r} {p*100:.0f}%" if r else ""
        return f"{t1[1][:14]:>14s} {t1[2]*100:>4.0f}%{anth}"

    header = (f"{'photo':>20s}  {'v7':>30s}  {'v8':>30s}  {'v9':>30s}  "
              f"{'leaf':>30s}  {'v9+leaf 30/70':>30s}")
    print(header)
    for name, path in photos:
        r_v7 = predict_image_path(v7, path)
        r_v8 = predict_image_path(v8, path)
        r_v9 = predict_image_path(v9, path)
        r_leaf = predict_leaf_image_path(leaf, path)
        r_s50 = predict_hybrid(v9, leaf, path, strategy="soft")
        r_s37 = predict_hybrid(v9, leaf, path, strategy="soft",
                               synth_weight=0.3, leaf_weight=0.7)
        r_conf = predict_hybrid(v9, leaf, path, strategy="confidence")

        for key, res in [("v7", r_v7), ("v8", r_v8), ("v9", r_v9),
                         ("leaf", r_leaf),
                         ("v9+leaf soft50", r_s50),
                         ("v9+leaf soft30/70", r_s37),
                         ("v9+leaf conf", r_conf)]:
            r, _ = anth_rank(res["top_k"])
            if r == 1:
                counts[key][0] += 1
                counts[key][1] += 1
            elif r in (2, 3):
                counts[key][1] += 1

        print(f"{name:>20s}  {cell(r_v7):>30s}  {cell(r_v8):>30s}  "
              f"{cell(r_v9):>30s}  {cell(r_leaf):>30s}  {cell(r_s37):>30s}")

    print()
    print(f"{'strategy':>20s}  {'Anth Top-1':>11s}  {'Anth Top-3':>11s}")
    for k in strategies:
        t1, t3 = counts[k]
        print(f"{k:>20s}  {t1}/{len(photos)}        {t3}/{len(photos)}")


if __name__ == "__main__":
    main()
