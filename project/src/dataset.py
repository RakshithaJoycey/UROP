"""
dataset.py
----------
Loads the face_ms dataset (1 RGB image + 31 spectral bands),
extracts spatial patches, and returns a PyTorch Dataset.

Why patches?
  We only have one scene, so we cannot split by image.
  Instead we tile the image into non-overlapping patches and
  assign 80% to train and 20% to val.  This avoids data leakage
  while still giving the model diverse spatial samples.
"""

import os
import io
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset_from_zip(zip_path: str):
    """
    Opens face_ms.zip and returns:
        rgb  : np.ndarray  (H, W, 3)  uint8
        hsi  : np.ndarray  (H, W, 31) float32   (raw uint16 cast to float32)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

        # ── find RGB ──
        rgb_candidates = [n for n in names if n.lower().endswith(".bmp")]
        if not rgb_candidates:
            raise FileNotFoundError(f"No .bmp file found inside {zip_path}")
        rgb_name = rgb_candidates[0]
        with z.open(rgb_name) as f:
            rgb = np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))  # (H,W,3) uint8

        # ── find spectral bands ──
        band_names = sorted(
            [n for n in names if n.lower().endswith(".png") and "face_ms_" in n.lower()]
        )
        if len(band_names) != 31:
            raise ValueError(
                f"Expected 31 spectral bands, found {len(band_names)}. Files: {band_names}"
            )

        bands = []
        for bname in band_names:
            with z.open(bname) as f:
                img = Image.open(io.BytesIO(f.read()))
                arr = np.array(img).astype(np.float32)   # keep 16-bit range
                bands.append(arr)

        hsi = np.stack(bands, axis=-1)   # (H, W, 31)

    print("=== Dataset loaded ===")
    print(f"  RGB  : {rgb.shape}  dtype={rgb.dtype}  min={rgb.min()}  max={rgb.max()}")
    print(f"  HSI  : {hsi.shape}  dtype={hsi.dtype}  min={hsi.min():.1f}  max={hsi.max():.1f}")
    print(f"  NaN in HSI: {np.isnan(hsi).any()}   Inf in HSI: {np.isinf(hsi).any()}")
    return rgb, hsi


def make_patches(rgb: np.ndarray, hsi: np.ndarray, patch_size: int = 64):
    """
    Tiles (rgb, hsi) into non-overlapping square patches.
    Returns two lists of numpy arrays.
    """
    H, W, _ = rgb.shape
    patches_rgb, patches_hsi = [], []

    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patches_rgb.append(rgb[y:y+patch_size, x:x+patch_size, :])
            patches_hsi.append(hsi[y:y+patch_size, x:x+patch_size, :])

    print(f"  Patch size   : {patch_size}×{patch_size}")
    print(f"  Total patches: {len(patches_rgb)}")
    return patches_rgb, patches_hsi


# ── normalisation statistics ──────────────────────────────────────────────────

def compute_normalisation(hsi: np.ndarray):
    """
    Computes per-band mean and std from the FULL hsi array.
    We use per-band z-score for HSI so each band is treated fairly.
    Returns dicts: {band_idx: (mean, std)}
    """
    stats = {}
    for b in range(hsi.shape[-1]):
        m = hsi[..., b].mean()
        s = hsi[..., b].std()
        if s < 1e-6:
            s = 1.0   # avoid division by zero for flat bands
        stats[b] = (float(m), float(s))
    return stats


# ── Dataset ──────────────────────────────────────────────────────────────────

class FaceMSDataset(Dataset):
    """
    PyTorch Dataset that returns (rgb_patch, hsi_patch) tensors.

    rgb_patch : (3,  patch_size, patch_size)  float32  in [0, 1]
    hsi_patch : (31, patch_size, patch_size)  float32  z-score normalised
    """

    def __init__(
        self,
        patches_rgb: list,
        patches_hsi: list,
        hsi_stats: dict,
    ):
        self.patches_rgb = patches_rgb
        self.patches_hsi = patches_hsi
        self.hsi_stats   = hsi_stats

    def __len__(self):
        return len(self.patches_rgb)

    def __getitem__(self, idx):
        rgb = self.patches_rgb[idx].astype(np.float32) / 255.0   # [0,1]
        hsi = self.patches_hsi[idx].astype(np.float32)

        # per-band z-score on HSI
        hsi_norm = np.zeros_like(hsi)
        for b in range(hsi.shape[-1]):
            mean, std = self.hsi_stats[b]
            hsi_norm[..., b] = (hsi[..., b] - mean) / std

        # HWC → CHW  (PyTorch convention)
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))        # (3,  H, W)
        hsi_t = torch.from_numpy(hsi_norm.transpose(2, 0, 1))   # (31, H, W)

        return rgb_t, hsi_t


# ── factory function ──────────────────────────────────────────────────────────

def build_datasets(
    zip_path: str,
    patch_size: int = 64,
    train_split: float = 0.8,
    seed: int = 42,
):
    """
    Full pipeline:
      1. Load RGB + HSI from ZIP
      2. Tile into patches
      3. Shuffle and split 80/20
      4. Return (train_dataset, val_dataset, hsi_stats, rgb_full, hsi_full)
    """
    rgb, hsi = load_dataset_from_zip(zip_path)

    patches_rgb, patches_hsi = make_patches(rgb, hsi, patch_size)

    # compute stats on ALL patches (use full image stats for consistency)
    hsi_stats = compute_normalisation(hsi)

    # reproducible shuffle then split
    rng = np.random.default_rng(seed)
    indices = np.arange(len(patches_rgb))
    rng.shuffle(indices)

    n_train = int(len(indices) * train_split)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_rgb = [patches_rgb[i] for i in train_idx]
    train_hsi = [patches_hsi[i] for i in train_idx]
    val_rgb   = [patches_rgb[i] for i in val_idx]
    val_hsi   = [patches_hsi[i] for i in val_idx]

    print(f"  Train patches: {len(train_rgb)}   Val patches: {len(val_rgb)}")

    train_ds = FaceMSDataset(train_rgb, train_hsi, hsi_stats)
    val_ds   = FaceMSDataset(val_rgb,   val_hsi,   hsi_stats)

    return train_ds, val_ds, hsi_stats, rgb, hsi
