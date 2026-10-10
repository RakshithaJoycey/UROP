"""
train.py
--------
Stage 1: Train MST++ on RGB → 31-band HSI reconstruction (healthy face skin).
Uses physics-informed BioOpticalLoss (MSE + chromophore + smoothness + melanin).

Usage:
    python src/train.py --zip ../face_ms.zip
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from dataset      import build_datasets
from mst_plus_plus import MSTpp, count_parameters
from bio_loss     import BioOpticalLoss


def get_args():
    p = argparse.ArgumentParser(description="Train MST++ RGB→HSI with bio-optical loss")
    p.add_argument("--zip",           default="../face_ms.zip")
    p.add_argument("--patch_size",    type=int,   default=64)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--base_ch",       type=int,   default=32)
    p.add_argument("--num_blocks",    type=int,   default=2)
    p.add_argument("--num_heads",     type=int,   default=1)
    p.add_argument("--lambda_chrom",  type=float, default=0.05)
    p.add_argument("--lambda_smooth", type=float, default=0.01)
    p.add_argument("--lambda_mono",   type=float, default=0.01)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--checkpoints",   default="../checkpoints")
    p.add_argument("--outputs",       default="../outputs")
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_breakdown = {"mse": 0, "chrom": 0, "smooth": 0, "mono": 0}

    for rgb, hsi in loader:
        rgb, hsi = rgb.to(device), hsi.to(device)
        pred = model(rgb)
        loss, breakdown = criterion(pred, hsi)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n = rgb.size(0)
        total_loss += loss.item() * n
        for k in total_breakdown:
            total_breakdown[k] += breakdown[k] * n

    n_total = len(loader.dataset)
    avg_breakdown = {k: v / n_total for k, v in total_breakdown.items()}
    return total_loss / n_total, avg_breakdown


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rgb, hsi in loader:
            rgb, hsi = rgb.to(device), hsi.to(device)
            pred = model(rgb)
            loss, _ = criterion(pred, hsi)
            total_loss += loss.item() * rgb.size(0)
    return total_loss / len(loader.dataset)


def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── data ──
    print("\nLoading dataset …")
    train_ds, val_ds, hsi_stats, rgb_full, hsi_full = build_datasets(
        zip_path=args.zip, patch_size=args.patch_size, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── model ──
    model = MSTpp(
        in_channels=3, out_channels=31,
        base_ch=args.base_ch, num_blocks=args.num_blocks, num_heads=args.num_heads
    ).to(device)
    print(f"\nModel: MST++  |  Parameters: {count_parameters(model):,}")

    # ── loss + optimiser ──
    criterion = BioOpticalLoss(
        lambda_chrom=args.lambda_chrom,
        lambda_smooth=args.lambda_smooth,
        lambda_mono=args.lambda_mono,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── output dirs ──
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.outputs,     exist_ok=True)
    best_ckpt = os.path.join(args.checkpoints, "best_model.pth")

    # save stats for evaluate.py / adapt.py
    with open(os.path.join(args.checkpoints, "hsi_stats.json"), "w") as f:
        json.dump({str(k): list(v) for k, v in hsi_stats.items()}, f, indent=2)
    np.save(os.path.join(args.checkpoints, "rgb_full.npy"), rgb_full)
    np.save(os.path.join(args.checkpoints, "hsi_full.npy"), hsi_full)
    print(f"Stats + full arrays saved to {args.checkpoints}/")

    # ── training loop ──
    print(f"\nTraining MST++ for up to {args.epochs} epochs …\n")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'MSE':>8}  {'Chrom':>8}  {'Smooth':>8}  {'Mono':>8}  {'s':>6}")
    print("-" * 80)

    best_val   = float("inf")
    no_improve = 0
    history    = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, bd = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss       = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **bd
        })
        print(f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{bd['mse']:>8.5f}  {bd['chrom']:>8.5f}  {bd['smooth']:>8.5f}  "
              f"{bd['mono']:>8.5f}  {elapsed:>6.1f}s")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    best_val,
                "args":        vars(args),
                "model_type":  "MSTpp",
            }, best_ckpt)
            print(f"         ✓ Best saved (val={best_val:.5f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    with open(os.path.join(args.outputs, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone.  Best val loss: {best_val:.5f}  →  {best_ckpt}")


if __name__ == "__main__":
    main()
