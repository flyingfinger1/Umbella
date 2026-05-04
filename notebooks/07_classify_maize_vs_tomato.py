"""07: End-to-end sanity check — classify Maize vs. Tomato from skeleton features.

Loads all serialized skeletons in data/skeletons/, extracts structural feature
vectors, runs Leave-One-Plant-Out cross-validation with logistic regression
and a random forest. Prints per-fold accuracy, overall accuracy at scan and
plant level, plus per-feature importance.

This is a sanity check, not a real benchmark — Maize and Tomato have very
different topology, so accuracy should be near-perfect. If it isn't, we
have a bug somewhere upstream.

Usage:
    .venv/Scripts/python.exe notebooks/07_classify_maize_vs_tomato.py
"""

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import Skeleton                                # noqa: E402
from src.eval import skeleton_features, FEATURE_NAMES            # noqa: E402

from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.ensemble import RandomForestClassifier              # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SKEL_DIR = ROOT / "data" / "skeletons"


def load_corpus():
    rows = []
    for plant_dir in sorted(SKEL_DIR.iterdir()):
        if not plant_dir.is_dir():
            continue
        plant_id = plant_dir.name
        species = "Maize" if plant_id.startswith("Maize") else "Tomato"
        for json_path in sorted(plant_dir.glob("*.json")):
            skel = Skeleton.load_json(json_path)
            feats = skeleton_features(skel)
            rows.append((plant_id, species, json_path.stem, feats))
    return rows


def lopo_cv(rows, classifier_factory, scale: bool = True):
    """Leave-one-plant-out cross-validation. Returns per-row predictions."""
    plants = sorted({r[0] for r in rows})
    species = ["Maize", "Tomato"]
    label_of = {s: i for i, s in enumerate(species)}

    X = np.stack([r[3] for r in rows])
    y = np.array([label_of[r[1]] for r in rows])
    plant_of = np.array([r[0] for r in rows])

    pred = np.full(len(rows), -1, dtype=np.int32)

    for hold in plants:
        train = plant_of != hold
        test = plant_of == hold
        Xtr, ytr = X[train], y[train]
        Xte = X[test]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr = sc.transform(Xtr)
            Xte = sc.transform(Xte)
        clf = classifier_factory()
        clf.fit(Xtr, ytr)
        pred[test] = clf.predict(Xte)

    return X, y, pred, plant_of


def report(name, rows, y, pred, plant_of):
    species_names = ["Maize", "Tomato"]
    scan_acc = float((pred == y).mean())
    print(f"\n[{name}]")
    print(f"  scan-level accuracy: {scan_acc * 100:.1f}%  ({(pred == y).sum()}/{len(y)})")

    # plant-level: majority vote per plant
    plant_correct = 0
    plant_total = 0
    plant_results = {}
    for plant in sorted(set(plant_of)):
        mask = plant_of == plant
        votes = Counter(pred[mask].tolist())
        true_label = int(y[mask][0])
        majority = votes.most_common(1)[0][0]
        ok = majority == true_label
        plant_correct += ok
        plant_total += 1
        plant_results[plant] = (species_names[true_label], species_names[majority],
                                int(mask.sum()), int((pred[mask] == true_label).sum()))
    print(f"  plant-level accuracy (majority vote): {plant_correct}/{plant_total}")
    for plant, (true_lbl, pred_lbl, n_scans, n_correct_scans) in plant_results.items():
        marker = "  ok" if true_lbl == pred_lbl else "  MISCLASSIFIED"
        print(f"    {plant}: true={true_lbl}, pred={pred_lbl}, "
              f"{n_correct_scans}/{n_scans} scans correct{marker}")

    # confusion matrix at scan level
    cm = np.zeros((2, 2), dtype=np.int32)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print(f"  scan confusion (rows=true, cols=pred):")
    print(f"            {species_names[0]:>8s} {species_names[1]:>8s}")
    for i, name_i in enumerate(species_names):
        print(f"    {name_i:>6s}  {cm[i, 0]:>8d} {cm[i, 1]:>8d}")


def feature_importance(rows):
    """Train a single RF on all data + report Gini importance, just for ranking insight."""
    X = np.stack([r[3] for r in rows])
    y = np.array([0 if r[1] == "Maize" else 1 for r in rows])
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    print("\n[feature importance — RF Gini, full dataset, indicative only]")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:>22s}  {rf.feature_importances_[i]:.3f}")


def main():
    rows = load_corpus()
    n = len(rows)
    n_maize = sum(1 for r in rows if r[1] == "Maize")
    n_tomato = n - n_maize
    print(f"loaded {n} skeletons: {n_maize} Maize, {n_tomato} Tomato across "
          f"{len({r[0] for r in rows})} plants\n")

    X, y, pred, plant_of = lopo_cv(rows, lambda: LogisticRegression(max_iter=2000), scale=True)
    report("LogReg (LOPO CV)", rows, y, pred, plant_of)

    X, y, pred, plant_of = lopo_cv(rows, lambda: RandomForestClassifier(
        n_estimators=200, random_state=0), scale=False)
    report("RandomForest (LOPO CV)", rows, y, pred, plant_of)

    feature_importance(rows)

    # extra: how does accuracy depend on growth stage?
    print("\n[scan-level accuracy by date — RandomForest]")
    by_date = defaultdict(list)
    for (plant, species, date, _), t, p in zip(rows, y, pred):
        by_date[date].append(int(t == p))
    for date in sorted(by_date):
        acc = sum(by_date[date]) / len(by_date[date])
        print(f"  {date}: {acc * 100:>5.1f}%  ({sum(by_date[date])}/{len(by_date[date])})")


if __name__ == "__main__":
    main()
