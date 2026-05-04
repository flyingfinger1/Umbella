"""18: Overnight orchestrator.

Step 1: train on v3 with cosine LR schedule + 20 epochs (cheap).
Step 2: only if Conium recall < 0.85 OR Anthriscus recall < 0.85, build a
        higher-resolution dataset (768x768) and retrain.

All output captured to data/models/overnight_log.txt; final summary printed
at the end.

Usage:
    .venv/Scripts/python.exe notebooks/18_overnight.py
"""

from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import list_examples, build_dataset                # noqa: E402
from src.models import (                                                # noqa: E402
    ApiaceaeCNN, ApiaceaeImageDataset, instance_stratified_split,
)
from src.models.classifier import SPECIES_ORDER                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "models" / "overnight_log.txt"
LOG.parent.mkdir(parents=True, exist_ok=True)
log_handle = open(LOG, "w", encoding="utf-8", buffering=1)


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


def print_confusion(cm, header: str = ""):
    if header:
        log(header)
    head = "          " + "".join(f"{s[:9]:>10s}" for s in SPECIES_ORDER)
    log(head)
    for i, s in enumerate(SPECIES_ORDER):
        row = "".join(f"{cm[i, j]:>10d}" for j in range(6))
        log(f"  {s[:9]:>9s} {row}  diag={cm[i, i]}/{cm[i].sum()}")


def train_classifier(
    data_root: Path,
    ckpt_path: Path,
    epochs: int = 20,
    batch: int = 16,
    base_lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    examples = list_examples(data_root)
    train, val, test = instance_stratified_split(examples,
                                                 train_frac=0.7, val_frac=0.15, seed=0)
    log(f"  split: {len(train)} train, {len(val)} val, {len(test)} test")

    train_ds = ApiaceaeImageDataset(data_root, train, augment=True)
    val_ds = ApiaceaeImageDataset(data_root, val)
    test_ds = ApiaceaeImageDataset(data_root, test)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ApiaceaeCNN(n_classes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  model: {n_params:,} params  device: {device}  epochs: {epochs}")

    opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=base_lr * 0.01)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, epochs + 1):
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
        train_acc = total_correct / total
        train_loss = total_loss / total
        val_acc, _ = evaluate(model, val_ld, device)
        dt = time.time() - t0
        marker = "  *" if val_acc > best_val else ""
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), ckpt_path)
        log(f"  epoch {epoch:>2d}/{epochs}  lr={opt.param_groups[0]['lr']:.5f}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc * 100:5.1f}%  "
            f"val_acc={val_acc * 100:5.1f}%  ({dt:.0f}s){marker}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, cm = evaluate(model, test_ld, device)
    log(f"  best val: {best_val * 100:.1f}%   test acc: {test_acc * 100:.1f}%")
    print_confusion(cm)
    # per-class recall
    per_class = {SPECIES_ORDER[i]: float(cm[i, i] / max(cm[i].sum(), 1))
                 for i in range(6)}
    log("  per-class recall:")
    for k, v in per_class.items():
        log(f"    {k:>22s}: {v * 100:.1f}%")
    return test_acc, per_class


def main():
    t_global = time.time()
    log(f"=== overnight run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # -------- step 1: longer training on existing v3 -----------------------
    log("=== STEP 1: longer training on v3 (cosine LR, 20 epochs) ===")
    v3_test_acc, v3_per_class = train_classifier(
        data_root=ROOT / "data" / "training" / "v3",
        ckpt_path=ROOT / "data" / "models" / "v3_long_classifier.pt",
        epochs=20,
    )
    step1_dt = time.time() - t_global
    log(f"\n=== STEP 1 done in {step1_dt / 60:.1f} min ===\n")

    # decide whether step 2 is needed
    conium_ok = v3_per_class["Conium_maculatum"] >= 0.85
    anthriscus_ok = v3_per_class["Anthriscus_sylvestris"] >= 0.85
    if conium_ok and anthriscus_ok:
        log(f"hard pair both above 85% — skipping step 2.")
        log(f"final test acc: {v3_test_acc * 100:.1f}%")
        log_handle.close()
        return

    log(f"hard pair below 85% (Conium {v3_per_class['Conium_maculatum'] * 100:.1f}%, "
        f"Anthriscus {v3_per_class['Anthriscus_sylvestris'] * 100:.1f}%) — "
        f"proceeding to step 2.\n")

    # -------- step 2: build v4 at 768x768 + train --------------------------
    v4_root = ROOT / "data" / "training" / "v4"
    log(f"=== STEP 2a: building v4 at 768x768 -> {v4_root.relative_to(ROOT)}/ ===")
    t_build = time.time()
    build_dataset(
        out_root=v4_root,
        n_per_species=200,
        n_views=4,
        image_size=(768, 768),
        points_per_mm2=3.0,           # bump density slightly to fill more pixels
        point_radius_px=3,            # thicker splats keep visual feel similar to 384
        progress=False,
    )
    log(f"  v4 build done in {(time.time() - t_build) / 60:.1f} min")

    log(f"\n=== STEP 2b: training on v4 (cosine LR, 15 epochs at higher res) ===")
    v4_test_acc, v4_per_class = train_classifier(
        data_root=v4_root,
        ckpt_path=ROOT / "data" / "models" / "v4_classifier.pt",
        epochs=15,
        batch=8,                       # smaller batch since 768x768 uses more memory
    )

    log(f"\n=== overnight run done in {(time.time() - t_global) / 60:.1f} min total ===")
    log(f"v3-long  test_acc {v3_test_acc * 100:.1f}%   "
        f"Conium {v3_per_class['Conium_maculatum'] * 100:.1f}%  "
        f"Anthriscus {v3_per_class['Anthriscus_sylvestris'] * 100:.1f}%")
    log(f"v4-768   test_acc {v4_test_acc * 100:.1f}%   "
        f"Conium {v4_per_class['Conium_maculatum'] * 100:.1f}%  "
        f"Anthriscus {v4_per_class['Anthriscus_sylvestris'] * 100:.1f}%")
    log_handle.close()


if __name__ == "__main__":
    main()
