"""27: build v9 (clean RGB save) + train with online augmentation.

Architecture change vs v6/v7/v8:
  - build_dataset() saves CLEAN RGB (no augmentation) plus label and depth
  - ApiaceaeImageDataset applies augment_render() at load time, fresh per call
  - BG pool = 77 curated subject-free textures from data/bg_textures/

Hypothesis: online augmentation (12 epochs × 4800 examples = 57 600 unique
augmented views) plus subject-free real backgrounds will improve real-photo
generalization vs v8's "iNat with subjects + heavy blur" approach.
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
DATA = ROOT / "data" / "training" / "v9"
BG_DIR = ROOT / "data" / "bg_textures"
OUT_CKPT = ROOT / "data" / "models" / "v9_classifier.pt"
LOG = ROOT / "data" / "models" / "v9_log.txt"
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
    log(f"=== v9 overnight {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # Step 1: build dataset with CLEAN RGB (no offline augmentation)
    log("=== STEP 1: building v9 (clean RGB, no offline aug) ===")
    t0 = time.time()
    build_dataset(out_root=DATA, n_per_species=200, n_views=4,
                  image_size=(384, 384), augment=False, progress=False)
    log(f"  build done in {(time.time() - t0) / 60:.1f} min\n")

    # Step 2: load BG pool for online augmentation
    bg_pool = sorted(p for p in BG_DIR.rglob("*.jpg"))
    log(f"BG pool: {len(bg_pool)} subject-free textures from {BG_DIR.name}/\n")

    log("=== STEP 2: training v9 with online augmentation ===")
    examples = list_examples(DATA)
    train, val, test = instance_stratified_split(examples,
                                                 train_frac=0.7, val_frac=0.15, seed=0)
    log(f"  split: {len(train)} train, {len(val)} val, {len(test)} test")

    train_ds = ApiaceaeImageDataset(DATA, train, augment=True, online_aug_pool=bg_pool)
    val_ds = ApiaceaeImageDataset(DATA, val, online_aug_pool=bg_pool)
    test_ds = ApiaceaeImageDataset(DATA, test, online_aug_pool=bg_pool)
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
