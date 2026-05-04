"""22: build v7 (corrected botanical params + Pastinaca yellow + augmentation) and train.

Same training hyperparameters as v6. Only species spec ranges and Pastinaca
pedicel_rgb change between v6 and v7 datasets.
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

from src.training import build_dataset, list_examples
from src.models import ApiaceaeCNN, ApiaceaeImageDataset, instance_stratified_split
from src.models.classifier import SPECIES_ORDER

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training" / "v7"
OUT_CKPT = ROOT / "data" / "models" / "v7_classifier.pt"
LOG = ROOT / "data" / "models" / "v7_log.txt"
LOG.parent.mkdir(parents=True, exist_ok=True)
log_handle = open(LOG, "w", encoding="utf-8", buffering=1)

EPOCHS = 20
BATCH = 16
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4


def log(msg: str = ""):
    print(msg)
    log_handle.write(msg + "\n")


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
    t_global = time.time()
    log(f"=== v7 overnight {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    log("=== STEP 1: building v7 (corrected botany + augmented) ===")
    t0 = time.time()
    build_dataset(out_root=DATA, n_per_species=200, n_views=4,
                  image_size=(384, 384), augment=True, progress=False)
    log(f"  build done in {(time.time() - t0) / 60:.1f} min\n")

    log("=== STEP 2: training v7 ===")
    examples = list_examples(DATA)
    train, val, test = instance_stratified_split(examples,
                                                 train_frac=0.7, val_frac=0.15, seed=0)
    log(f"  split: {len(train)} train, {len(val)} val, {len(test)} test")

    train_ds = ApiaceaeImageDataset(DATA, train, augment=True)
    val_ds = ApiaceaeImageDataset(DATA, val)
    test_ds = ApiaceaeImageDataset(DATA, test)
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ApiaceaeCNN(n_classes=6).to(device)
    log(f"  model: {sum(p.numel() for p in model.parameters()):,} params  device: {device}")

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
        log(f"  epoch {epoch:>2d}/{EPOCHS}  lr={opt.param_groups[0]['lr']:.5f}  "
            f"train_loss={total_loss / total:.4f}  "
            f"train_acc={100 * total_correct / total:5.1f}%  "
            f"val_acc={val_acc * 100:5.1f}%  ({dt:.0f}s){marker}")

    model.load_state_dict(torch.load(OUT_CKPT, map_location=device))
    test_acc, cm = evaluate(model, test_ld, device)
    log(f"\nbest val: {best_val * 100:.1f}%   test acc: {test_acc * 100:.1f}%")
    log("          " + "".join(f"{s[:9]:>10s}" for s in SPECIES_ORDER))
    for i, s in enumerate(SPECIES_ORDER):
        row = "".join(f"{cm[i, j]:>10d}" for j in range(6))
        log(f"  {s[:9]:>9s} {row}  diag={cm[i, i]}/{cm[i].sum()}")
    log("\nper-class recall:")
    for i, s in enumerate(SPECIES_ORDER):
        log(f"  {s:>22s}: {100 * cm[i, i] / max(cm[i].sum(), 1):.1f}%")

    log(f"\n=== total {(time.time() - t_global) / 60:.1f} min ===")
    log_handle.close()


if __name__ == "__main__":
    main()
