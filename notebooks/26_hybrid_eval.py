"""26: Evaluate the hybrid (synth + leaf) classifier on real Apiaceae photos.

Compares:
  - Synth v7 alone
  - Leaf classifier alone
  - Hybrid soft-vote (50/50)
  - Hybrid max-confidence
  - Hybrid vote
on all 8 of the user's Wiesen-Kerbel photos (4 indoor + 4 outdoor) plus the
Anthriscus stock photo. Reports top-1 plus Anthriscus rank and confidence.
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
    """Return (rank, conf) of Anthriscus_sylvestris in top_k, or (None, 0)."""
    for i, (s, _, p) in enumerate(top_k, start=1):
        if s == "Anthriscus_sylvestris":
            return i, p
    return None, 0.0


def main() -> None:
    synth = load_model(ROOT / "data" / "models" / "v7_classifier.pt")
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

    counts = {k: [0, 0] for k in
              ["synth", "leaf", "soft50", "soft30/70", "conf", "leaf_pri", "vote"]}
    # counts[name] = [top1_count, top3_count] for Anthriscus

    print(f"{'photo':>20s}  {'synth':>30s}  {'leaf':>30s}  {'soft':>30s}  {'max':>30s}")
    for name, path in photos:
        synth_res = predict_image_path(synth, path)
        leaf_res = predict_leaf_image_path(leaf, path)
        soft_res = predict_hybrid(synth, leaf, path, strategy="soft")
        soft_l = predict_hybrid(synth, leaf, path, strategy="soft",
                                synth_weight=0.3, leaf_weight=0.7)
        conf_res = predict_hybrid(synth, leaf, path, strategy="confidence")
        leaf_pri = predict_hybrid(synth, leaf, path, strategy="leaf_priority")
        vote_res = predict_hybrid(synth, leaf, path, strategy="vote")

        for key, res in [("synth", synth_res), ("leaf", leaf_res),
                         ("soft50", soft_res), ("soft30/70", soft_l),
                         ("conf", conf_res), ("leaf_pri", leaf_pri),
                         ("vote", vote_res)]:
            r, p = anth_rank(res["top_k"])
            if r == 1:
                counts[key][0] += 1
                counts[key][1] += 1
            elif r in (2, 3):
                counts[key][1] += 1

        def cell(res):
            t1 = res["top_k"][0]
            r, p = anth_rank(res["top_k"])
            anth = f"  Anth#{r} {p*100:.0f}%" if r else ""
            return f"{t1[1][:14]:>14s} {t1[2]*100:>4.0f}%{anth}"

        print(f"{name:>20s}  {cell(synth_res):>30s}  {cell(leaf_res):>30s}  "
              f"{cell(soft_l):>30s}  {cell(leaf_pri):>30s}")

    print()
    print(f"{'strategy':>10s}  {'Anth Top-1':>11s}  {'Anth Top-3':>11s}")
    for k, (t1, t3) in counts.items():
        print(f"{k:>10s}  {t1}/{len(photos)}        {t3}/{len(photos)}")


if __name__ == "__main__":
    main()
