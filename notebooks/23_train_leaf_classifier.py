"""23: Train the ResNet-18 leaf classifier on the iNaturalist data.

Should be run AFTER notebooks/15-style fetch is complete (see
src/leaf/fetch_inaturalist.py). Output checkpoint:
data/models/leaf_classifier.pt
"""

from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leaf import (
    LeafImageDataset, collect_image_paths, stratified_split,
    build_leaf_classifier, SPECIES_ORDER,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "leaf_images"
OUT_CKPT = ROOT / "data" / "models" / "leaf_classifier.pt"
OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

EPOCHS = 12
BATCH = 32
BASE_LR = 5e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    cm = np.zeros((6, 6), dtype=np.int32)
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += y.numel()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1
    return correct / max(total, 1), cm


def main():
    entries = collect_image_paths(DATA)
    train, val, test = stratified_split(entries, train_frac=0.8, val_frac=0.1, seed=0)
    print(f"loaded {len(entries)} images")
    print(f"split:  {len(train)} train, {len(val)} val, {len(test)} test")
    counts = {s: 0 for s in SPECIES_ORDER}
    for _, l in entries:
        counts[SPECIES_ORDER[l]] += 1
    print("per-species counts:")
    for k, v in counts.items():
        print(f"  {k:>22s}: {v}")

    train_ds = LeafImageDataset(DATA, train, size=IMAGE_SIZE, augment=True)
    val_ds = LeafImageDataset(DATA, val, size=IMAGE_SIZE)
    test_ds = LeafImageDataset(DATA, test, size=IMAGE_SIZE)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_leaf_classifier(n_classes=6, pretrained=True).to(device)
    print(f"device: {device}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=BASE_LR * 0.05)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in train_ld:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * y.numel()
            total_correct += int((logits.argmax(1) == y).sum())
            total += y.numel()
        sched.step()
        val_acc, _ = evaluate(model, val_ld, device)
        dt = time.time() - t0
        marker = "  *" if val_acc > best_val else ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), OUT_CKPT)
        print(f"epoch {epoch:>2d}/{EPOCHS}  lr={opt.param_groups[0]['lr']:.5f}  "
              f"train_loss={total_loss / total:.4f}  "
              f"train_acc={100 * total_correct / total:5.1f}%  "
              f"val_acc={val_acc * 100:5.1f}%  ({dt:.0f}s){marker}")

    model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
    test_acc, cm = evaluate(model, test_ld, device)
    print(f"\nbest val: {best_val * 100:.1f}%   test acc: {test_acc * 100:.1f}%")
    print("          " + "".join(f"{s[:9]:>10s}" for s in SPECIES_ORDER))
    for i, s in enumerate(SPECIES_ORDER):
        row = "".join(f"{cm[i, j]:>10d}" for j in range(6))
        print(f"  {s[:9]:>9s} {row}  diag={cm[i, i]}/{cm[i].sum()}")


if __name__ == "__main__":
    main()
