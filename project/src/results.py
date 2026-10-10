"""
results.py
----------
Complete results and visualisation pipeline for all 4 stages:

  Stage 1 – MST++ reconstruction results
  Stage 2 – Bio-optical constraint analysis
  Stage 3 – Domain adaptation (synthetic lesion generation + adaptation)
  Stage 4 – Melanoma classifier evaluation

Run:
    python src/results.py --zip ../face_ms.zip

All figures saved to outputs/
"""

import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(__file__))

from dataset        import load_dataset_from_zip, make_patches, compute_normalisation
from model          import SimpleHSINet
from mst_plus_plus  import MSTpp, count_parameters
from bio_loss       import BioOpticalLoss, _MEL_RAW, _HBO2_RAW, _HB_RAW, _build_basis
from lesion_dataset import build_lesion_patches, LesionDataset, apply_melanoma_perturbation, apply_benign_perturbation
from classifier     import MelanomaClassifier
from evaluate       import compute_rmse, compute_sam, compute_psnr, compute_mean_ssim


def load_model_from_ckpt(ckpt, args, device):
    """Detect architecture from saved keys and load the right model."""
    keys = list(ckpt["model_state"].keys())
    # SimpleHSINet has plain sequential keys like 'enc1.0.weight'
    # MSTpp has keys like 'stem.weight', 'enc1.downsample.weight'
    if any(k.startswith("stem.") for k in keys):
        model = MSTpp(
            in_channels=3, out_channels=31,
            base_ch=args.base_ch, num_blocks=args.num_blocks, num_heads=args.num_heads
        ).to(device)
        arch = "MSTpp"
    else:
        model = SimpleHSINet(in_channels=3, out_channels=31, base_ch=args.base_ch).to(device)
        arch = "SimpleHSINet"
    model.load_state_dict(ckpt["model_state"])
    return model, arch


# ── constants ─────────────────────────────────────────────────────────────────

WAVELENGTHS = np.arange(400, 710, 10)   # 31 bands, 400-700 nm
BAND_LABELS = [f"{w}nm" for w in WAVELENGTHS]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip",         default="../face_ms.zip")
    p.add_argument("--checkpoints", default="../checkpoints")
    p.add_argument("--outputs",     default="../outputs")
    p.add_argument("--base_ch",     type=int, default=32)
    p.add_argument("--num_blocks",  type=int, default=2)
    p.add_argument("--num_heads",   type=int, default=1)
    p.add_argument("--patch_size",  type=int, default=64)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n_melanoma",  type=int, default=100)
    p.add_argument("--n_benign",    type=int, default=100)
    p.add_argument("--adapt_epochs",type=int, default=25)
    p.add_argument("--clf_epochs",  type=int, default=40)
    return p.parse_args()


def norm_display(arr):
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1  – Load model & compute reconstruction metrics
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_data(args, device):
    print("\n" + "═"*60)
    print("STAGE 1: Loading MST++ model and reconstruction data")
    print("═"*60)

    rgb, hsi_gt = load_dataset_from_zip(args.zip)

    with open(os.path.join(args.checkpoints, "hsi_stats.json")) as f:
        hsi_stats = json.load(f)

    ckpt_path = os.path.join(args.checkpoints, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model, arch = load_model_from_ckpt(ckpt, args, device)
    model.eval()
    print(f"Loaded {arch} (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
    print(f"Parameters: {count_parameters(model):,}")

    pred_path = os.path.join(args.checkpoints, "pred_hsi_full.npy")
    pred_hsi = np.load(pred_path).astype(np.float32)

    H_crop = (rgb.shape[0] // args.patch_size) * args.patch_size
    W_crop = (rgb.shape[1] // args.patch_size) * args.patch_size
    pred_c = pred_hsi[:H_crop, :W_crop, :]
    gt_c   = hsi_gt[:H_crop, :W_crop, :].astype(np.float32)
    rgb_c  = rgb[:H_crop, :W_crop, :]

    rmse = compute_rmse(pred_c, gt_c)
    sam  = compute_sam(pred_c, gt_c)
    psnr = compute_psnr(pred_c, gt_c)
    ssim = compute_mean_ssim(pred_c, gt_c)

    print(f"\nReconstruction Metrics:")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  SAM  : {sam:.4f}°")
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")

    metrics = {"RMSE": rmse, "SAM_degrees": sam, "PSNR_dB": psnr, "SSIM": ssim,
               "checkpoint_epoch": int(ckpt["epoch"]),
               "val_loss_at_best": float(ckpt["val_loss"])}
    with open(os.path.join(args.outputs, "evaluation_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, rgb_c, gt_c, pred_c, hsi_stats, hsi_gt.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def viz_stage1_overview(rgb, gt_hsi, pred_hsi, metrics, out_dir):
    """Big overview: RGB | pseudo-RGB GT | pseudo-RGB Pred | metrics table"""
    def pseudo_rgb(hsi):
        r = norm_display(hsi[..., 24])   # ~640nm
        g = norm_display(hsi[..., 14])   # ~540nm
        b = norm_display(hsi[..., 6])    # ~460nm
        return np.stack([r, g, b], axis=-1)

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)
    fig.suptitle("Stage 1 – MST++ RGB → Hyperspectral Reconstruction", fontsize=15, fontweight="bold")

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(rgb)
    ax0.set_title("Input RGB", fontsize=12)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(pseudo_rgb(gt_hsi))
    ax1.set_title("Ground Truth\n(Pseudo-RGB from HSI)", fontsize=12)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(pseudo_rgb(pred_hsi))
    ax2.set_title("Reconstructed HSI\n(MST++ output)", fontsize=12)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    table_data = [
        ["RMSE",  f"{metrics['RMSE']:.2f}",    "lower ↓"],
        ["SAM",   f"{metrics['SAM_degrees']:.3f}°", "lower ↓"],
        ["PSNR",  f"{metrics['PSNR_dB']:.2f} dB", "higher ↑"],
        ["SSIM",  f"{metrics['SSIM']:.4f}",    "higher ↑"],
    ]
    tbl = ax3.table(
        cellText=table_data,
        colLabels=["Metric", "Value", "Direction"],
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c5f8a")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#eef4fb")
    ax3.set_title("Reconstruction Quality", fontsize=12, pad=20)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage1_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_band_grid(gt_hsi, pred_hsi, out_dir):
    """4×8 grid showing all 31 bands: GT row and Predicted row side-by-side"""
    n_bands = 31
    cols = 8
    rows_per = 2  # GT + Pred per pair
    # select 16 representative bands
    sel_bands = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle("Stage 1 – Band-by-Band Reconstruction (GT top / Pred bottom)", fontsize=14, fontweight="bold")

    for i, b in enumerate(sel_bands):
        r_gt   = (i // 8) * 2
        r_pred = r_gt + 1
        col    = i % 8

        wl = WAVELENGTHS[b]
        axes[r_gt][col].imshow(norm_display(gt_hsi[..., b]),   cmap="gray")
        axes[r_gt][col].set_title(f"{wl}nm GT",  fontsize=8)
        axes[r_gt][col].axis("off")

        axes[r_pred][col].imshow(norm_display(pred_hsi[..., b]), cmap="gray")
        axes[r_pred][col].set_title(f"{wl}nm Pred", fontsize=8)
        axes[r_pred][col].axis("off")

    fig.tight_layout()
    path = os.path.join(out_dir, "stage1_band_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_error_maps(gt_hsi, pred_hsi, out_dir):
    """Per-band absolute error heat maps + per-band RMSE bar chart"""
    sel_bands = [0, 6, 14, 19, 24, 30]
    wls = [WAVELENGTHS[b] for b in sel_bands]

    fig, axes = plt.subplots(2, len(sel_bands), figsize=(18, 6))
    fig.suptitle("Stage 1 – Reconstruction Error Analysis", fontsize=14, fontweight="bold")

    # row 0: error maps
    for i, b in enumerate(sel_bands):
        err = np.abs(gt_hsi[..., b] - pred_hsi[..., b])
        im = axes[0][i].imshow(err, cmap="hot")
        axes[0][i].set_title(f"{wls[i]}nm", fontsize=10)
        axes[0][i].axis("off")
        plt.colorbar(im, ax=axes[0][i], fraction=0.046, pad=0.04)

    # row 1: per-band RMSE bar chart (spanning full width)
    axes[1][0].remove()
    ax_bar = fig.add_subplot(2, 1, 2)
    band_rmses = [np.sqrt(np.mean((gt_hsi[..., b] - pred_hsi[..., b])**2)) for b in range(31)]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 31))
    ax_bar.bar(WAVELENGTHS, band_rmses, width=8, color=colors, edgecolor="none")
    ax_bar.set_xlabel("Wavelength (nm)", fontsize=11)
    ax_bar.set_ylabel("RMSE", fontsize=11)
    ax_bar.set_title("Per-Band RMSE Across All 31 Spectral Bands", fontsize=11)
    ax_bar.grid(axis="y", alpha=0.3)
    for i in range(1, len(sel_bands)):
        axes[1][i].remove()

    fig.tight_layout()
    path = os.path.join(out_dir, "stage1_error_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_spectral_profiles(gt_hsi, pred_hsi, rgb, out_dir):
    """Spectral profiles at 6 different skin regions"""
    H, W = gt_hsi.shape[:2]
    pixels = [
        (H//2,     W//2,     "Centre"),
        (H//4,     W//4,     "Top-Left"),
        (H//4,     3*W//4,   "Top-Right"),
        (3*H//4,   W//4,     "Bot-Left"),
        (3*H//4,   3*W//4,   "Bot-Right"),
        (H//2,     W//4,     "Mid-Left"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Stage 1 – Spectral Profiles: GT vs Reconstructed (6 Skin Regions)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, (r, c, label) in enumerate(pixels):
        ax = axes[i]
        ax.plot(WAVELENGTHS, gt_hsi[r, c, :],   label="Ground Truth", color="#1f77b4", lw=2.5)
        ax.plot(WAVELENGTHS, pred_hsi[r, c, :], label="Predicted",    color="#ff7f0e", lw=2, linestyle="--")
        sam_val = np.degrees(np.arccos(np.clip(
            np.dot(gt_hsi[r,c,:], pred_hsi[r,c,:]) /
            (np.linalg.norm(gt_hsi[r,c,:])*np.linalg.norm(pred_hsi[r,c,:])+1e-8), -1, 1
        )))
        ax.set_title(f"{label} ({r},{c})  SAM={sam_val:.2f}°", fontsize=10)
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.set_ylabel("Intensity", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # shade melanin absorption region (blue range)
        ax.axvspan(400, 500, alpha=0.07, color="blue", label="Melanin-dominant")
        # shade HbO2 Q-bands
        ax.axvspan(530, 580, alpha=0.07, color="red",  label="HbO2 Q-bands")

    fig.tight_layout()
    path = os.path.join(out_dir, "stage1_spectral_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_loss_curve(out_dir):
    history_path = os.path.join(out_dir, "training_history.json")
    if not os.path.exists(history_path):
        print("  Skipping loss curve: no history file")
        return
    with open(history_path) as f:
        h = json.load(f)

    epochs = [e["epoch"] for e in h]
    train  = [e["train_loss"] for e in h]
    val    = [e["val_loss"]   for e in h]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train, label="Train Loss", color="#1f77b4", lw=2)
    ax.plot(epochs, val,   label="Val Loss",   color="#ff7f0e", lw=2)
    best_epoch = epochs[int(np.argmin(val))]
    best_val   = min(val)
    ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    ax.scatter([best_epoch], [best_val], color="green", zorder=5, s=80)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Stage 1 – MST++ Training History (BioOpticalLoss)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "stage1_loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – Bio-optical analysis visualisations
# ═══════════════════════════════════════════════════════════════════════════════

def viz_bio_optical(gt_hsi, pred_hsi, out_dir):
    """Chromophore maps + extinction spectra + constraint analysis"""
    print("\n" + "═"*60)
    print("STAGE 2: Bio-Optical Constraint Analysis")
    print("═"*60)

    basis   = _build_basis().numpy()    # (31, 3): mel, HbO2, Hb
    pinv    = np.linalg.pinv(basis)     # (3, 31)

    H, W, C = pred_hsi.shape
    pred_flat = pred_hsi.reshape(-1, C)   # (N, 31)
    gt_flat   = gt_hsi.reshape(-1, C)

    # --- normalise to [0,1] for chromophore fitting
    pred_norm = pred_flat / (pred_flat.max() + 1e-8)
    gt_norm   = gt_flat   / (gt_flat.max()   + 1e-8)

    c_pred = (pinv @ pred_norm.T).T   # (N, 3)  chromophore weights
    c_gt   = (pinv @ gt_norm.T).T
    c_pred = np.clip(c_pred, 0, None)
    c_gt   = np.clip(c_gt,   0, None)

    mel_map_pred  = c_pred[:, 0].reshape(H, W)
    hbo2_map_pred = c_pred[:, 1].reshape(H, W)
    hb_map_pred   = c_pred[:, 2].reshape(H, W)
    mel_map_gt    = c_gt[:, 0].reshape(H, W)

    # ── Figure 1: Chromophore concentration maps ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Stage 2 – Bio-Optical Chromophore Maps (Physics-Informed Reconstruction)", fontsize=14, fontweight="bold")

    cmaps = ["YlOrBr", "Reds", "Blues"]
    titles_gt   = ["Melanin (GT)",    "HbO2 (GT)",      "Hb (GT)"]
    titles_pred = ["Melanin (Pred)",  "HbO2 (Pred)",    "Hb (Pred)"]
    maps_gt   = [c_gt[:,0].reshape(H,W),  c_gt[:,1].reshape(H,W),  c_gt[:,2].reshape(H,W)]
    maps_pred = [mel_map_pred, hbo2_map_pred, hb_map_pred]

    for i in range(3):
        im0 = axes[0][i].imshow(norm_display(maps_gt[i]),   cmap=cmaps[i])
        axes[0][i].set_title(titles_gt[i], fontsize=11)
        axes[0][i].axis("off")
        plt.colorbar(im0, ax=axes[0][i], fraction=0.046, pad=0.04)

        im1 = axes[1][i].imshow(norm_display(maps_pred[i]), cmap=cmaps[i])
        axes[1][i].set_title(titles_pred[i], fontsize=11)
        axes[1][i].axis("off")
        plt.colorbar(im1, ax=axes[1][i], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage2_chromophore_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Figure 2: Extinction basis spectra + example pixel fitting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stage 2 – Bio-Optical Basis Spectra and Pixel-Level Fitting", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(WAVELENGTHS, _MEL_RAW,  label="Melanin",          color="#8B4513", lw=2.5)
    ax.plot(WAVELENGTHS, _HBO2_RAW, label="OxyHaemoglobin",   color="#DC143C", lw=2.5)
    ax.plot(WAVELENGTHS, _HB_RAW,   label="DeoxyHaemoglobin", color="#00008B", lw=2.5)
    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Normalised Absorption", fontsize=11)
    ax.set_title("Chromophore Extinction Spectra (400-700nm)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(400, 500, alpha=0.06, color="blue",  label="Melanin dominant")
    ax.axvspan(530, 580, alpha=0.06, color="red",   label="HbO2 Q-bands")

    # example pixel fit
    ax2 = axes[1]
    r, c = H//2, W//2
    gt_px   = gt_norm[r*W + c]
    pred_px = pred_norm[r*W + c]
    c_gt_px   = np.clip(pinv @ gt_px,   0, None)
    c_pred_px = np.clip(pinv @ pred_px, 0, None)
    gt_fit    = basis @ c_gt_px
    pred_fit  = basis @ c_pred_px

    ax2.plot(WAVELENGTHS, gt_px,    label="GT spectrum",      color="#1f77b4", lw=2.5)
    ax2.plot(WAVELENGTHS, gt_fit,   label="GT chromophore fit",   color="#1f77b4", lw=1.5, linestyle=":")
    ax2.plot(WAVELENGTHS, pred_px,  label="Predicted spectrum",color="#ff7f0e", lw=2.5)
    ax2.plot(WAVELENGTHS, pred_fit, label="Pred chromophore fit", color="#ff7f0e", lw=1.5, linestyle=":")
    ax2.set_xlabel("Wavelength (nm)", fontsize=11)
    ax2.set_ylabel("Normalised Reflectance", fontsize=11)
    ax2.set_title(f"Centre Pixel Chromophore Decomposition\n"
                  f"GT: Mel={c_gt_px[0]:.3f} HbO2={c_gt_px[1]:.3f} Hb={c_gt_px[2]:.3f}\n"
                  f"Pr: Mel={c_pred_px[0]:.3f} HbO2={c_pred_px[1]:.3f} Hb={c_pred_px[2]:.3f}", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage2_bio_optical_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Figure 3: Melanin monotonicity check ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 2 – Melanin Monotonicity Constraint Analysis", fontsize=14, fontweight="bold")

    # sample 50 random pixels and check monotonicity
    rng = np.random.default_rng(42)
    idxs = rng.choice(H*W, 50, replace=False)
    violations_gt   = []
    violations_pred = []
    for idx in idxs:
        mel_c_gt   = c_gt[idx, 0]
        mel_c_pred = c_pred[idx, 0]
        mel_sp_gt   = mel_c_gt   * basis[:, 0]
        mel_sp_pred = mel_c_pred * basis[:, 0]
        viol_gt   = max(0, np.sum(np.maximum(0, np.diff(mel_sp_gt))))
        viol_pred = max(0, np.sum(np.maximum(0, np.diff(mel_sp_pred))))
        violations_gt.append(viol_gt)
        violations_pred.append(viol_pred)

    axes[0].bar(range(50), violations_gt,   color="#1f77b4", alpha=0.7, label="Ground Truth")
    axes[0].bar(range(50), violations_pred, color="#ff7f0e", alpha=0.7, label="Predicted")
    axes[0].set_xlabel("Pixel sample index", fontsize=10)
    axes[0].set_ylabel("Monotonicity violation (sum of positive diffs)", fontsize=9)
    axes[0].set_title("Melanin Monotonicity: positive = violation", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # average per-band spectral smoothness (second derivative)
    gt_mean   = gt_flat.mean(axis=0)
    pred_mean = pred_flat.mean(axis=0)
    d2_gt   = np.abs(gt_mean[2:]   - 2*gt_mean[1:-1]   + gt_mean[:-2])
    d2_pred = np.abs(pred_mean[2:] - 2*pred_mean[1:-1] + pred_mean[:-2])

    axes[1].plot(WAVELENGTHS[1:-1], d2_gt,   label="GT 2nd derivative",   color="#1f77b4", lw=2)
    axes[1].plot(WAVELENGTHS[1:-1], d2_pred, label="Pred 2nd derivative", color="#ff7f0e", lw=2)
    axes[1].set_xlabel("Wavelength (nm)", fontsize=10)
    axes[1].set_ylabel("|d²R/dλ²|  (spectral smoothness violation)", fontsize=9)
    axes[1].set_title("Spectral Smoothness: Mean Spectrum 2nd Derivative", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage2_constraints.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – Domain adaptation
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage3_adaptation(args, model, hsi_patches_all, hsi_stats, device):
    print("\n" + "═"*60)
    print("STAGE 3: Domain Adaptation – Healthy Skin → Lesion")
    print("═"*60)

    rng = np.random.default_rng(args.seed)
    hsi_p, labels = build_lesion_patches(
        hsi_patches_all, args.n_melanoma, args.n_benign, args.seed
    )

    dataset = LesionDataset(hsi_p, labels, hsi_stats)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=0)
    print(f"  Adaptation: {n_train} train / {n_val} val  ({args.n_melanoma} mel + {args.n_benign} benign)")

    criterion = BioOpticalLoss(lambda_chrom=0.05, lambda_smooth=0.01, lambda_mono=0.01).to(device)

    # freeze encoder layers (works for both SimpleHSINet and MSTpp)
    def set_freeze(m, frozen):
        for name, param in m.named_parameters():
            if any(tag in name for tag in ("stem", "enc1", "enc2", "enc3")):
                param.requires_grad = not frozen

    set_freeze(model, True)
    freeze_epochs = 5
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4
    )

    history = []
    best_val = float("inf")
    adapt_ckpt = os.path.join(args.checkpoints, "adapted_model.pth")

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'Encoder':>8}")
    print("-"*42)

    for epoch in range(1, args.adapt_epochs + 1):
        if epoch == freeze_epochs + 1:
            set_freeze(model, False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-4)

        frozen_str = "frozen" if epoch <= freeze_epochs else "unfrozen"
        model.train()
        t_loss = 0.0
        for rgb, hsi, _ in train_loader:
            rgb, hsi = rgb.to(device), hsi.to(device)
            pred = model(rgb)
            loss, _ = criterion(pred, hsi)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * rgb.size(0)
        t_loss /= n_train

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for rgb, hsi, _ in val_loader:
                rgb, hsi = rgb.to(device), hsi.to(device)
                v_loss += criterion(model(rgb), hsi)[0].item() * rgb.size(0)
        v_loss /= n_val

        history.append({"epoch": epoch, "train": t_loss, "val": v_loss})
        print(f"{epoch:>6}  {t_loss:>10.5f}  {v_loss:>10.5f}  {frozen_str:>8}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": best_val, "model_type": "MSTpp_adapted"}, adapt_ckpt)
            print(f"         ✓ Saved (val={best_val:.5f})")

    ckpt = torch.load(adapt_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    with open(os.path.join(args.outputs, "adapt_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nAdaptation done. Best val loss: {best_val:.5f}")

    return model, dataset, hsi_p, labels


def viz_stage3_lesion_synthesis(hsi_patches_all, out_dir):
    """Show example healthy vs melanoma vs benign HSI patches + spectra"""
    rng = np.random.default_rng(42)
    base_idx = rng.integers(0, len(hsi_patches_all))
    base     = hsi_patches_all[base_idx]

    mel_patch    = apply_melanoma_perturbation(base.copy(), np.random.default_rng(10))
    benign_patch = apply_benign_perturbation(base.copy(),  np.random.default_rng(20))

    def pseudo_rgb_hsi(patch):
        r = norm_display(patch[..., 24])
        g = norm_display(patch[..., 14])
        b = norm_display(patch[..., 6])
        return np.stack([r, g, b], axis=-1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Stage 3 – Synthetic Lesion HSI Generation (Domain Adaptation Source)", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # pseudo-RGB patches
    for col, (patch, title) in enumerate([
        (base,        "Healthy Skin (Face)"),
        (mel_patch,   "Synthetic Melanoma"),
        (benign_patch,"Synthetic Benign Naevus"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pseudo_rgb_hsi(patch))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    # spectral comparison
    cx, cy = base.shape[0]//2, base.shape[1]//2
    ax_sp = fig.add_subplot(gs[1, :])

    def norm_sp(patch):
        sp = patch[cx, cy, :].astype(np.float64)
        return sp / (sp.max() + 1e-8)

    ax_sp.plot(WAVELENGTHS, norm_sp(base),        label="Healthy Skin",       color="#2e8b57", lw=2.5)
    ax_sp.plot(WAVELENGTHS, norm_sp(mel_patch),   label="Synthetic Melanoma", color="#8b0000", lw=2.5)
    ax_sp.plot(WAVELENGTHS, norm_sp(benign_patch),label="Synthetic Benign",   color="#ff8c00", lw=2.5)

    # annotate key chromophore bands
    ax_sp.axvspan(400, 490, alpha=0.07, color="indigo",label="Melanin-dominant range")
    ax_sp.axvspan(530, 580, alpha=0.07, color="red",   label="HbO2 Q-bands")
    ax_sp.set_xlabel("Wavelength (nm)", fontsize=12)
    ax_sp.set_ylabel("Normalised Reflectance", fontsize=12)
    ax_sp.set_title("Centre-Pixel Spectral Signatures: Healthy vs Melanoma vs Benign", fontsize=12)
    ax_sp.legend(fontsize=10)
    ax_sp.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage3_lesion_synthesis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_stage3_adaptation_curve(out_dir):
    hist_path = os.path.join(out_dir, "adapt_history.json")
    if not os.path.exists(hist_path):
        print("  Skipping adaptation curve: no history file")
        return
    with open(hist_path) as f:
        h = json.load(f)

    epochs = [e["epoch"] for e in h]
    train  = [e["train"] for e in h]
    val    = [e["val"]   for e in h]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train, label="Train Loss (Adaptation)", color="#2e8b57", lw=2)
    ax.plot(epochs, val,   label="Val Loss (Adaptation)",   color="#8b0000", lw=2)
    ax.axvline(5, color="gray", linestyle="--", alpha=0.6, label="Encoder unfrozen (epoch 6)")
    best_e = epochs[int(np.argmin(val))]
    ax.axvline(best_e, color="green", linestyle=":", alpha=0.8, label=f"Best epoch {best_e}")
    ax.set_xlabel("Adaptation Epoch", fontsize=12)
    ax.set_ylabel("BioOpticalLoss", fontsize=12)
    ax.set_title("Stage 3 – Domain Adaptation Loss (Healthy Face → Lesion)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "stage3_adaptation_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_stage3_chromophore_shift(hsi_patches_all, out_dir):
    """Compare chromophore distributions: healthy vs melanoma vs benign"""
    rng = np.random.default_rng(42)
    basis = _build_basis().numpy()
    pinv  = np.linalg.pinv(basis)

    n = min(50, len(hsi_patches_all))
    idxs = rng.choice(len(hsi_patches_all), n, replace=False)

    mel_concs_h, hbo2_concs_h = [], []
    mel_concs_m, hbo2_concs_m = [], []
    mel_concs_b, hbo2_concs_b = [], []

    for idx in idxs:
        base  = hsi_patches_all[idx].astype(np.float32)
        bmax  = base.max() + 1e-8
        mel_p = apply_melanoma_perturbation(base.copy(), np.random.default_rng(int(idx)))
        ben_p = apply_benign_perturbation(base.copy(),   np.random.default_rng(int(idx)+100))

        for patch, mc, hc in [(base, mel_concs_h, hbo2_concs_h),
                               (mel_p, mel_concs_m, hbo2_concs_m),
                               (ben_p, mel_concs_b, hbo2_concs_b)]:
            cx, cy  = patch.shape[0]//2, patch.shape[1]//2
            sp_norm = patch[cx, cy, :] / (patch.max() + 1e-8)
            c       = np.clip(pinv @ sp_norm, 0, None)
            mc.append(c[0])
            hc.append(c[1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 3 – Chromophore Distribution Shift: Healthy vs Lesion Domains", fontsize=14, fontweight="bold")

    categories = ["Healthy\n(Face)", "Melanoma\n(Synthetic)", "Benign\n(Synthetic)"]
    mel_data  = [mel_concs_h,  mel_concs_m,  mel_concs_b]
    hbo2_data = [hbo2_concs_h, hbo2_concs_m, hbo2_concs_b]
    colors    = ["#2e8b57", "#8b0000", "#ff8c00"]

    for ax, data, title, ylab in [
        (axes[0], mel_data,  "Melanin Concentration",          "Estimated Melanin Conc."),
        (axes[1], hbo2_data, "Oxyhaemoglobin Concentration",   "Estimated HbO2 Conc."),
    ]:
        bp = ax.boxplot(data, patch_artist=True, labels=categories, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage3_chromophore_shift.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – Classifier
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage4_classifier(args, adapted_model, lesion_dataset, device):
    print("\n" + "═"*60)
    print("STAGE 4: Melanoma vs Benign Classifier Training")
    print("═"*60)

    for p in adapted_model.parameters():
        p.requires_grad = False
    adapted_model.eval()

    clf = MelanomaClassifier(in_bands=31, feature_dim=128, dropout=0.3).to(device)
    print(f"  Classifier params: {count_parameters(clf):,}")

    n_train = int(0.8 * len(lesion_dataset))
    n_val   = len(lesion_dataset) - n_train
    train_ds, val_ds = random_split(
        lesion_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.clf_epochs, eta_min=1e-5)

    history    = []
    best_acc   = 0.0
    clf_ckpt   = os.path.join(args.checkpoints, "classifier.pth")

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc':>10}")
    print("-"*35)

    for epoch in range(1, args.clf_epochs + 1):
        clf.train()
        t_loss = 0.0
        for rgb, hsi, labels in train_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            with torch.no_grad():
                pred_hsi = adapted_model(rgb)
            logits = clf(pred_hsi)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item() * rgb.size(0)
        t_loss /= n_train
        scheduler.step()

        clf.eval()
        correct, total = 0, 0
        all_probs, all_labels_list = [], []
        with torch.no_grad():
            for rgb, hsi, labels in val_loader:
                rgb, labels = rgb.to(device), labels.to(device)
                pred_hsi = adapted_model(rgb)
                logits   = clf(pred_hsi)
                probs    = torch.softmax(logits, dim=1)[:, 1]
                preds    = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels_list.extend(labels.cpu().numpy().tolist())

        val_acc = correct / total
        history.append({"epoch": epoch, "train_loss": t_loss, "val_acc": val_acc})
        print(f"{epoch:>6}  {t_loss:>12.5f}  {val_acc:>10.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"epoch": epoch, "classifier_state": clf.state_dict(),
                        "val_acc": best_acc}, clf_ckpt)
            print(f"         ✓ Best clf saved (acc={best_acc:.4f})")

    # final eval
    clf.eval()
    all_preds, all_probs2, all_true = [], [], []
    with torch.no_grad():
        for rgb, hsi, labels in val_loader:
            rgb, labels = rgb.to(device), labels.to(device)
            pred_hsi = adapted_model(rgb)
            logits   = clf(pred_hsi)
            probs    = torch.softmax(logits, dim=1)[:, 1]
            preds    = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs2.extend(probs.cpu().numpy().tolist())
            all_true.extend(labels.cpu().numpy().tolist())

    metrics = _clf_metrics(all_true, all_preds, all_probs2)
    with open(os.path.join(args.outputs, "classifier_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.outputs, "clf_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n=== Stage 4 Classifier Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:.4f}")

    return clf, metrics, all_true, all_probs2, history


def _clf_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    tn = int(((y_pred==0)&(y_true==0)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    acc  = (tp+tn)/(tp+tn+fp+fn+1e-8)
    sens = tp/(tp+fn+1e-8)
    spec = tn/(tn+fp+1e-8)
    prec = tp/(tp+fp+1e-8)
    f1   = 2*prec*sens/(prec+sens+1e-8)
    thrs = np.linspace(0,1,200)
    tprs, fprs = [], []
    for thr in thrs:
        p   = (y_prob>=thr).astype(int)
        tpr = ((p==1)&(y_true==1)).sum()/(y_true.sum()+1e-8)
        fpr = ((p==1)&(y_true==0)).sum()/((y_true==0).sum()+1e-8)
        tprs.append(float(tpr)); fprs.append(float(fpr))
    auc = float(np.trapz(tprs[::-1], fprs[::-1]))
    return {"accuracy": float(acc), "sensitivity": float(sens), "specificity": float(spec),
            "precision": float(prec), "f1_score": float(f1), "auc_roc": float(auc),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "_tprs": tprs, "_fprs": fprs}

# =============================================================================
# STAGE 4 VISUALISATIONS
# =============================================================================

def viz_stage4_roc_curve(metrics, out_dir):
    tprs = metrics["_tprs"]
    fprs = metrics["_fprs"]
    auc  = metrics["auc_roc"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fprs[::-1], tprs[::-1], color="#8b0000", lw=2.5, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Classifier")
    ax.fill_between(fprs[::-1], tprs[::-1], alpha=0.1, color="#8b0000")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("Stage 4 – ROC Curve: Melanoma vs Benign Classification", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    path = os.path.join(out_dir, "stage4_roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_stage4_confusion_matrix(metrics, out_dir):
    tp, tn, fp, fn = metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]
    cm = np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 4 – Classification Results", fontsize=14, fontweight="bold")

    # confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Benign", "Pred Melanoma"], fontsize=11)
    ax.set_yticklabels(["True Benign", "True Melanoma"], fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title("Confusion Matrix", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # metrics bar chart
    ax2 = axes[1]
    met_names = ["Accuracy", "Sensitivity\n(Recall)", "Specificity", "Precision", "F1 Score", "AUC-ROC"]
    met_vals  = [metrics["accuracy"], metrics["sensitivity"], metrics["specificity"],
                 metrics["precision"], metrics["f1_score"],   metrics["auc_roc"]]
    colors = ["#2c5f8a", "#8b0000", "#2e8b57", "#ff8c00", "#6a0dad", "#b8860b"]
    bars = ax2.barh(met_names, met_vals, color=colors, alpha=0.85, height=0.6)
    for bar, val in zip(bars, met_vals):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
    ax2.set_xlim([0, 1.15])
    ax2.set_xlabel("Score", fontsize=11)
    ax2.set_title("Classification Metrics", fontsize=12)
    ax2.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 baseline")
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage4_classification_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def viz_stage4_clf_training(history, out_dir):
    epochs    = [e["epoch"]      for e in history]
    t_loss    = [e["train_loss"] for e in history]
    val_acc   = [e["val_acc"]    for e in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stage 4 – Classifier Training Curves", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, t_loss, color="#2c5f8a", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_acc, color="#8b0000", lw=2)
    best_e = epochs[int(np.argmax(val_acc))]
    best_a = max(val_acc)
    axes[1].axhline(best_a, color="green", linestyle="--", alpha=0.7,
                    label=f"Best {best_a:.4f} @ epoch {best_e}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "stage4_clf_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def viz_summary_dashboard(recon_metrics, clf_metrics, out_dir):
    """Single-page pipeline summary figure"""
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor("#0f1923")
    fig.suptitle(
        "Physics-Informed RGB-to-HSI Reconstruction for Skin Lesion Analysis\n"
        "Full Pipeline Results Summary",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)

    # colour scheme
    card_bg  = "#1e2d3d"
    text_col = "white"
    val_col  = "#00d4ff"

    def metric_card(ax, title, value, unit, color, subtitle=""):
        ax.set_facecolor(card_bg)
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.72, title, ha="center", va="center",
                fontsize=12, color=text_col, transform=ax.transAxes)
        ax.text(0.5, 0.42, value, ha="center", va="center",
                fontsize=26, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.18, unit, ha="center", va="center",
                fontsize=10, color="#aaaaaa", transform=ax.transAxes)
        if subtitle:
            ax.text(0.5, 0.03, subtitle, ha="center", va="center",
                    fontsize=8, color="#888888", transform=ax.transAxes)

    # row 0: reconstruction metrics
    rc_data = [
        ("RMSE",  f"{recon_metrics['RMSE']:.1f}",         "raw units ↓",   "#ff6b6b", "lower is better"),
        ("SAM",   f"{recon_metrics['SAM_degrees']:.3f}",  "degrees ↓",     "#ffa500", "lower is better"),
        ("PSNR",  f"{recon_metrics['PSNR_dB']:.2f}",     "dB ↑",          "#4ecdc4", "higher is better"),
        ("SSIM",  f"{recon_metrics['SSIM']:.4f}",         "↑",             "#95e1d3", "higher is better"),
    ]
    for col, (title, val, unit, color, sub) in enumerate(rc_data):
        ax = fig.add_subplot(gs[0, col])
        metric_card(ax, title, val, unit, color, sub)

    # row 1 col 0-1: classification metrics  (2 cards)
    clf_data = [
        ("Accuracy",    f"{clf_metrics['accuracy']:.4f}",    "↑",       "#7bc8f6"),
        ("Sensitivity", f"{clf_metrics['sensitivity']:.4f}", "↑",       "#ff9ff3"),
        ("Specificity", f"{clf_metrics['specificity']:.4f}", "↑",       "#54a0ff"),
        ("AUC-ROC",     f"{clf_metrics['auc_roc']:.4f}",    "↑",       "#5f27cd"),
    ]
    for col, (title, val, unit, color) in enumerate(clf_data):
        ax = fig.add_subplot(gs[1, col])
        metric_card(ax, title, val, unit, color)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "SUMMARY_DASHBOARD.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    os.makedirs(args.outputs, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)

    # ── Stage 1: load model + compute metrics ──────────────────────────────
    model, rgb, gt_hsi, pred_hsi, hsi_stats, hsi_gt_full = load_model_and_data(args, device)

    print("\nGenerating Stage 1 visualisations …")
    viz_stage1_overview(rgb, gt_hsi, pred_hsi,
                        json.load(open(os.path.join(args.outputs, "evaluation_results.json"))),
                        args.outputs)
    viz_band_grid(gt_hsi, pred_hsi, args.outputs)
    viz_error_maps(gt_hsi, pred_hsi, args.outputs)
    viz_spectral_profiles(gt_hsi, pred_hsi, rgb, args.outputs)
    viz_loss_curve(args.outputs)

    # ── Stage 2: bio-optical analysis ──────────────────────────────────────
    viz_bio_optical(gt_hsi, pred_hsi, args.outputs)

    # ── Stage 3: lesion synthesis visualisation (before adaptation) ─────────
    print("\nGenerating Stage 3 pre-adaptation visualisations …")
    _, hsi_patches_all = make_patches(
        rgb, gt_hsi, patch_size=args.patch_size
    )
    viz_stage3_lesion_synthesis(hsi_patches_all, args.outputs)
    viz_stage3_chromophore_shift(hsi_patches_all, args.outputs)

    # ── Stage 3: run domain adaptation ─────────────────────────────────────
    adapted_model, lesion_dataset, hsi_p, labels = run_stage3_adaptation(
        args, model, hsi_patches_all, hsi_stats, device
    )
    viz_stage3_adaptation_curve(args.outputs)

    # ── Stage 4: train classifier ───────────────────────────────────────────
    clf, clf_metrics, all_true, all_probs, clf_history = run_stage4_classifier(
        args, adapted_model, lesion_dataset, device
    )

    print("\nGenerating Stage 4 visualisations …")
    viz_stage4_roc_curve(clf_metrics, args.outputs)
    viz_stage4_confusion_matrix(clf_metrics, args.outputs)
    viz_stage4_clf_training(clf_history, args.outputs)

    # ── Summary dashboard ───────────────────────────────────────────────────
    recon_metrics = json.load(open(os.path.join(args.outputs, "evaluation_results.json")))
    viz_summary_dashboard(recon_metrics, clf_metrics, args.outputs)

    # ── Print final summary ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nStage 1 – Reconstruction (MST++ epoch {recon_metrics['checkpoint_epoch']}):")
    print(f"  RMSE  : {recon_metrics['RMSE']:.2f}")
    print(f"  SAM   : {recon_metrics['SAM_degrees']:.4f}°")
    print(f"  PSNR  : {recon_metrics['PSNR_dB']:.2f} dB")
    print(f"  SSIM  : {recon_metrics['SSIM']:.4f}")
    print(f"\nStage 4 – Classification:")
    print(f"  Accuracy    : {clf_metrics['accuracy']:.4f}")
    print(f"  Sensitivity : {clf_metrics['sensitivity']:.4f}")
    print(f"  Specificity : {clf_metrics['specificity']:.4f}")
    print(f"  AUC-ROC     : {clf_metrics['auc_roc']:.4f}")
    print(f"\nAll outputs saved to: {os.path.abspath(args.outputs)}")
    print("="*60)


if __name__ == "__main__":
    main()
