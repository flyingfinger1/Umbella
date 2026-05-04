"""19: Train classifier on v5 (with Conium stem speckles).

Same hyperparameters as v3-long for an apples-to-apples comparison: only
the stem speckles change between v3 and v5 datasets.

Usage:
    .venv/Scripts/python.exe notebooks/19_train_v5_long.py
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

from src.training import list_examples
from src.models import ApiaceaeCNN, ApiaceaeImageDataset, instance_stratified_split
from src.models.classifier import SPECIES_ORDER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training" / "v5"
OUT_CKPT = ROOT / "data" / "models" / "v5_classifier.pt"
OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)

EPOCHS = 20
BATCH = 16
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4


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
    examples = list_examples(DATA)
    train, val, test = instance_stratified_split(examples,
                                                 train_frac=0.7, val_frac=0.15, seed=0)
    print(f"split: {len(train)} train, {len(val)} val, {len(test)} test")

    train_ds = ApiaceaeImageDataset(DATA, train, augment=True)
    val_ds = ApiaceaeImageDataset(DATA, val)
    test_ds = ApiaceaeImageDataset(DATA, test)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ApiaceaeCNN(n_classes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params  device: {device}  epochs: {EPOCHS}")

    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=BASE_LR * 0.01)
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
    head = "          " + "".join(f"{s[:9]:>10s}" for s in SPECIES_ORDER)
    print(head)
    for i, s in enumerate(SPECIES_ORDER):
        row = "".join(f"{cm[i, j]:>10d}" for j in range(6))
        print(f"  {s[:9]:>9s} {row}  diag={cm[i, i]}/{cm[i].sum()}")
    print("\nper-class recall:")
    for i, s in enumerate(SPECIES_ORDER):
        print(f"  {s:>22s}: {100 * cm[i, i] / max(cm[i].sum(), 1):.1f}%")


if __name__ == "__main__":
    main()
