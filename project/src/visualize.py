"""
visualize.py
------------
Generates all visualisation figures and saves them to outputs/.

Figures produced
----------------
1. original_rgb.png           – the input RGB image
2. band_comparison_XX.png     – side-by-side: GT band / Predicted band / Abs Error
   (for bands 1, 10, 20, 31 by default)
3. spectrum_pixel.png         – ground-truth vs predicted spectrum at 3 sample pixels
4. loss_curve.png             – training and validation loss curves

Dataset note
------------
The CAVE face dataset does not ship explicit wavelength labels in the ZIP,
so we label the x-axis as "Band index (1–31)" rather than inventing nm values.
If wavelength metadata becomes available it can be plugged in via --wavelengths.

Usage:
    python src/visualize.py --zip ../face_ms.zip
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_dataset_from_zip


# ── helpers ───────────────────────────────────────────────────────────────────

def normalise_for_display(arr: np.ndarray) -> np.ndarray:
    """Scales a 2-D array to [0,1] for display purposes only."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ── individual plot functions ─────────────────────────────────────────────────

def plot_original_rgb(rgb: np.ndarray, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title("Input RGB Image", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "original_rgb.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_band_comparison(
    gt_hsi:   np.ndarray,   # (H, W, 31)
    pred_hsi: np.ndarray,   # (H, W, 31)
    band_indices: list,     # 0-based
    out_dir:  str,
):
    """
    For each selected band: GT | Predicted | Absolute Error  (3 columns).
    """
    for b in band_indices:
        gt_b   = gt_hsi[..., b].astype(np.float32)
        pred_b = pred_hsi[..., b].astype(np.float32)
        err_b  = np.abs(gt_b - pred_b)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Spectral Band {b+1:02d}", fontsize=14, fontweight="bold")

        axes[0].imshow(normalise_for_display(gt_b),   cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(normalise_for_display(pred_b), cmap="gray")
        axes[1].set_title("Predicted")
        axes[1].axis("off")

        im = axes[2].imshow(err_b, cmap="hot")
        axes[2].set_title("Absolute Error")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.tight_layout()
        path = os.path.join(out_dir, f"band_comparison_{b+1:02d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_spectrum(
    gt_hsi:   np.ndarray,   # (H, W, 31)
    pred_hsi: np.ndarray,
    pixels:   list,         # list of (row, col) tuples
    out_dir:  str,
):
    """
    Ground-truth vs predicted spectrum for selected pixel locations.
    x-axis = band index (1-31); no wavelength assumed.
    """
    band_idx = np.arange(1, 32)
    n_pixels = len(pixels)

    fig, axes = plt.subplots(1, n_pixels, figsize=(6 * n_pixels, 4), squeeze=False)
    fig.suptitle("Spectral Profiles: Ground Truth vs Predicted", fontsize=13)

    for i, (r, c) in enumerate(pixels):
        ax = axes[0][i]
        ax.plot(band_idx, gt_hsi[r, c, :],   label="Ground Truth", color="steelblue",  lw=2)
        ax.plot(band_idx, pred_hsi[r, c, :], label="Predicted",    color="darkorange", lw=2, linestyle="--")
        ax.set_title(f"Pixel ({r}, {c})", fontsize=11)
        ax.set_xlabel("Band index")
        ax.set_ylabel("Intensity (raw units)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "spectrum_pixel.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_loss_curve(history_path: str, out_dir: str):
    if not os.path.exists(history_path):
        print(f"  Skipping loss curve (history not found: {history_path})")
        return
    with open(history_path) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax.plot(epochs, val_loss,   label="Val Loss",   color="darkorange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reconstructed_rgb(
    gt_hsi:   np.ndarray,   # (H, W, 31) ground truth
    pred_hsi: np.ndarray,   # (H, W, 31) predicted
    out_dir:  str,
):
    """
    Pseudo-RGB reconstruction by mapping bands 7, 15, 25 → R, G, B
    (rough visual approximation; exact mapping depends on sensor).
    """
    def make_pseudo_rgb(hsi, r_b, g_b, b_b):
        r = normalise_for_display(hsi[..., r_b])
        g = normalise_for_display(hsi[..., g_b])
        b = normalise_for_display(hsi[..., b_b])
        return np.stack([r, g, b], axis=-1)

    gt_prgb   = make_pseudo_rgb(gt_hsi,   6, 14, 24)
    pred_prgb = make_pseudo_rgb(pred_hsi, 6, 14, 24)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Pseudo-RGB (bands 7, 15, 25 → R, G, B)", fontsize=12)
    axes[0].imshow(gt_prgb)
    axes[0].set_title("Ground Truth pseudo-RGB")
    axes[0].axis("off")
    axes[1].imshow(pred_prgb)
    axes[1].set_title("Predicted pseudo-RGB")
    axes[1].axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "pseudo_rgb_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip",         default="../face_ms.zip")
    p.add_argument("--checkpoints", default="../checkpoints")
    p.add_argument("--outputs",     default="../outputs")
    p.add_argument("--bands",       default="0,9,19,30",
                   help="0-based band indices to visualise (comma-separated)")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.outputs, exist_ok=True)

    # ── load ground truth ──
    rgb, gt_hsi = load_dataset_from_zip(args.zip)
    gt_hsi = gt_hsi.astype(np.float32)

    # ── load predictions ──
    pred_path = os.path.join(args.checkpoints, "pred_hsi_full.npy")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}\n"
            "Run evaluate.py first to generate pred_hsi_full.npy"
        )
    pred_hsi = np.load(pred_path).astype(np.float32)
    print(f"Predictions loaded: {pred_hsi.shape}")

    # crop to the same spatial region
    H = min(gt_hsi.shape[0], pred_hsi.shape[0])
    W = min(gt_hsi.shape[1], pred_hsi.shape[1])
    gt_hsi   = gt_hsi[:H, :W, :]
    pred_hsi = pred_hsi[:H, :W, :]
    rgb_crop = rgb[:H, :W, :]

    print("\nGenerating visualisations …")

    # 1. Original RGB
    plot_original_rgb(rgb_crop, args.outputs)

    # 2. Band comparisons
    band_indices = [int(b) for b in args.bands.split(",")]
    plot_band_comparison(gt_hsi, pred_hsi, band_indices, args.outputs)

    # 3. Spectrum at a few pixel locations (centre, quarter-way, three-quarter)
    sample_pixels = [
        (H // 2,     W // 2),
        (H // 4,     W // 4),
        (3 * H // 4, 3 * W // 4),
    ]
    plot_spectrum(gt_hsi, pred_hsi, sample_pixels, args.outputs)

    # 4. Loss curve
    history_path = os.path.join(args.outputs, "training_history.json")
    plot_loss_curve(history_path, args.outputs)

    # 5. Pseudo-RGB comparison
    plot_reconstructed_rgb(gt_hsi, pred_hsi, args.outputs)

    print("\nAll visualisations saved to:", os.path.abspath(args.outputs))


if __name__ == "__main__":
    main()
