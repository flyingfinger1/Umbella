"""10: Classify the six calibrated Apiaceae species from skeleton features.

Generates a synthetic corpus (100 instances per species, 600 total), saves
each as JSON, then trains a classifier with stratified K-fold CV. Reports
overall accuracy, confusion matrix, and a focused look at the
Aethusa <-> Conium pair (the hard case the bract-aware features should help).

Usage:
    .venv/Scripts/python.exe notebooks/10_classify_apiaceae.py
"""

from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import SPECIES, generate_apiaceae        # noqa: E402
from src.geometry import Skeleton                            # noqa: E402
from src.eval import skeleton_features, FEATURE_NAMES        # noqa: E402

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.model_selection import StratifiedKFold          # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "skeletons" / "synthetic"

N_PER_SPECIES = 100
N_FOLDS = 5


def build_corpus() -> list[tuple[str, Path, np.ndarray]]:
    """Generate 100 instances per species, save each, extract features."""
    rows = []
    for key, spec in SPECIES.items():
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        for seed in range(N_PER_SPECIES):
            params = spec.sample(seed=seed)
            skel = generate_apiaceae(params)
            skel.metadata["species"] = key
            out_path = out_dir / f"seed{seed:03d}.json"
            skel.save_json(out_path)
            feats = skeleton_features(skel)
            rows.append((key, out_path, feats))
    return rows


def kfold_eval(rows, classifier_factory, scale: bool, name: str):
    species_names = list(SPECIES.keys())
    label_of = {s: i for i, s in enumerate(species_names)}

    X = np.stack([r[2] for r in rows])
    y = np.array([label_of[r[0]] for r in rows])

    pred = np.full(len(rows), -1, dtype=np.int32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr = sc.transform(Xtr)
            Xte = sc.transform(Xte)
        clf = classifier_factory()
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)

    acc = float((pred == y).mean())
    print(f"\n[{name}] overall accuracy: {acc * 100:.1f}%  ({(pred == y).sum()}/{len(y)})")

    # confusion matrix
    cm = np.zeros((len(species_names), len(species_names)), dtype=np.int32)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print(f"  confusion (rows=true, cols=pred):")
    head = "          " + " ".join(f"{s[:9]:>9s}" for s in species_names)
    print(head)
    for i, s in enumerate(species_names):
        row = " ".join(f"{cm[i, j]:>9d}" for j in range(len(species_names)))
        print(f"  {s[:9]:>9s} {row}  diag={cm[i, i]}/{cm[i].sum()}")

    # focused: Aethusa <-> Conium
    if "Aethusa_cynapium" in label_of and "Conium_maculatum" in label_of:
        ai = label_of["Aethusa_cynapium"]
        ci = label_of["Conium_maculatum"]
        a_as_c = int(cm[ai, ci])
        c_as_a = int(cm[ci, ai])
        print(f"  Aethusa->Conium errors: {a_as_c}/{cm[ai].sum()}  "
              f"Conium->Aethusa errors: {c_as_a}/{cm[ci].sum()}")
    return pred, y


def main():
    t0 = time.time()
    print("Building synthetic corpus...")
    rows = build_corpus()
    print(f"  generated {len(rows)} skeletons in {time.time() - t0:.1f}s")

    kfold_eval(rows, lambda: LogisticRegression(max_iter=2000),
               scale=True, name="LogReg")
    pred, y = kfold_eval(rows, lambda: RandomForestClassifier(n_estimators=300, random_state=0),
                         scale=False, name="RandomForest")

    # feature importance from a single RF on all data
    X = np.stack([r[2] for r in rows])
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\n[feature importance — RF, full data]")
    for i in order[:12]:
        print(f"  {FEATURE_NAMES[i]:>34s}  {rf.feature_importances_[i]:.3f}")

    # ablation: drop bract features and re-test
    bract_feature_idx = [
        i for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("n_bract") or "bracteole" in n or "bract_" in n
    ]
    keep = [i for i in range(len(FEATURE_NAMES)) if i not in bract_feature_idx]
    print(f"\n[ablation: dropping {len(bract_feature_idx)} bract features, "
          f"keeping {len(keep)}]")
    X_ablate = X[:, keep]
    pred_ab = np.full(len(rows), -1, dtype=np.int32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in skf.split(X_ablate, y):
        rf2 = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_ablate[tr], y[tr])
        pred_ab[te] = rf2.predict(X_ablate[te])
    acc_ab = float((pred_ab == y).mean())
    print(f"  RF accuracy without bract features: {acc_ab * 100:.1f}%  "
          f"(was {(pred == y).mean() * 100:.1f}% with them)")

    species_names = list(SPECIES.keys())
    label_of = {s: i for i, s in enumerate(species_names)}
    ai = label_of["Aethusa_cynapium"]
    ci = label_of["Conium_maculatum"]
    aethusa_mask = y == ai
    conium_mask = y == ci
    aethusa_acc_with = (pred[aethusa_mask] == ai).mean()
    aethusa_acc_without = (pred_ab[aethusa_mask] == ai).mean()
    conium_acc_with = (pred[conium_mask] == ci).mean()
    conium_acc_without = (pred_ab[conium_mask] == ci).mean()
    print(f"  Aethusa accuracy: with bracts {aethusa_acc_with * 100:.1f}% "
          f"-> without {aethusa_acc_without * 100:.1f}%")
    print(f"  Conium  accuracy: with bracts {conium_acc_with * 100:.1f}% "
          f"-> without {conium_acc_without * 100:.1f}%")


if __name__ == "__main__":
    main()
