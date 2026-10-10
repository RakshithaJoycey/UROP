"""
evaluate.py
-----------
Loads the best trained MST++ model, reconstructs the full image,
and computes RMSE, SAM and PSNR against the ground-truth HSI.

Metrics
-------
RMSE  : pixel-level reconstruction error in original units (lower is better)
SAM   : Spectral Angle Mapper in degrees (lower is better; 0° = perfect match)
PSNR  : Peak Signal-to-Noise Ratio in dB (higher is better; >30 dB = good)
SSIM  : per-band Structural Similarity (higher is better; 1.0 = perfect)

Usage:
    python src/evaluate.py --zip ../face_ms.zip
"""

import os
import sys
import argparse
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from dataset      import load_dataset_from_zip
from mst_plus_plus import MSTpp


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def compute_sam(pred: np.ndarray, gt: np.ndarray) -> float:
    dot     = np.sum(pred * gt, axis=-1)
    norm_p  = np.linalg.norm(pred, axis=-1)
    norm_g  = np.linalg.norm(gt,   axis=-1)
    cos_sim = np.clip(dot / (norm_p * norm_g + 1e-8), -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cos_sim))))


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    data_range = float(gt.max() - gt.min())
    if data_range < 1e-6:
        return float("inf")
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def compute_mean_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Per-band SSIM averaged across all 31 bands.
    Uses a simplified single-window formulation.
    pred, gt : (H, W, 31)
    """
    ssims = []
    for b in range(pred.shape[-1]):
        p = pred[..., b].astype(np.float64)
        g = gt[...,   b].astype(np.float64)
        mu_p, mu_g  = p.mean(), g.mean()
        sig_p  = p.std()
        sig_g  = g.std()
        sig_pg = np.mean((p - mu_p) * (g - mu_g))
        data_range = g.max() - g.min() + 1e-8
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2) / \
               ((mu_p**2 + mu_g**2 + c1) * (sig_p**2 + sig_g**2 + c2))
        ssims.append(ssim)
    return float(np.mean(ssims))


# ── full-image reconstruction ─────────────────────────────────────────────────

def predict_full_image(
    model:     torch.nn.Module,
    rgb:       np.ndarray,
    hsi_stats: dict,
    device,
    patch_size: int = 64,
) -> np.ndarray:
    """Patch-by-patch inference → de-normalised HSI (H, W, 31) float32."""
    model.eval()
    H, W, _ = rgb.shape
    pred_hsi = np.zeros((H, W, 31), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                patch = rgb[y:y+patch_size, x:x+patch_size, :].astype(np.float32) / 255.0
                t     = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
                out   = model(t).squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (P,P,31)
                for b in range(31):
                    m, s = hsi_stats[str(b)]
                    out[..., b] = out[..., b] * s + m
                pred_hsi[y:y+patch_size, x:x+patch_size, :] = out

    return pred_hsi


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip",         default="../face_ms.zip")
    p.add_argument("--checkpoints", default="../checkpoints")
    p.add_argument("--outputs",     default="../outputs")
    p.add_argument("--patch_size",  type=int, default=64)
    p.add_argument("--base_ch",     type=int, default=32)
    p.add_argument("--num_blocks",  type=int, default=2)
    p.add_argument("--num_heads",   type=int, default=1)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rgb, hsi_gt = load_dataset_from_zip(args.zip)

    stats_path = os.path.join(args.checkpoints, "hsi_stats.json")
    with open(stats_path) as f:
        hsi_stats = json.load(f)

    ckpt_path = os.path.join(args.checkpoints, "best_model.pth")
    model = MSTpp(
        in_channels=3, out_channels=31,
        base_ch=args.base_ch, num_blocks=args.num_blocks, num_heads=args.num_heads
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"Loaded {ckpt.get('model_type', 'model')} from epoch {ckpt['epoch']}  "
          f"(val_loss={ckpt['val_loss']:.5f})")

    print("Reconstructing HSI …")
    pred_hsi = predict_full_image(model, rgb, hsi_stats, device, patch_size=args.patch_size)

    H_crop = (rgb.shape[0] // args.patch_size) * args.patch_size
    W_crop = (rgb.shape[1] // args.patch_size) * args.patch_size
    pred_c = pred_hsi[:H_crop, :W_crop, :]
    gt_c   = hsi_gt[:H_crop, :W_crop, :].astype(np.float32)

    rmse = compute_rmse(pred_c, gt_c)
    sam  = compute_sam( pred_c, gt_c)
    psnr = compute_psnr(pred_c, gt_c)
    ssim = compute_mean_ssim(pred_c, gt_c)

    print("\n=== Evaluation Results (MST++) ===")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  SAM  : {sam:.4f} °")
    print(f"  PSNR : {psnr:.4f} dB")
    print(f"  SSIM : {ssim:.4f}")

    os.makedirs(args.outputs, exist_ok=True)
    results = {
        "RMSE": rmse, "SAM_degrees": sam, "PSNR_dB": psnr, "SSIM": ssim,
        "checkpoint_epoch": int(ckpt["epoch"]),
        "val_loss_at_best": float(ckpt["val_loss"]),
        "model_type": ckpt.get("model_type", "MSTpp"),
    }
    with open(os.path.join(args.outputs, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.outputs, "evaluation_results.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v}\n")

    np.save(os.path.join(args.checkpoints, "pred_hsi_full.npy"), pred_hsi)
    print(f"\nResults → {args.outputs}/evaluation_results.json")
    print(f"Predictions → {args.checkpoints}/pred_hsi_full.npy")


if __name__ == "__main__":
    main()
