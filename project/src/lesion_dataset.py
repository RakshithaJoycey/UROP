"""
lesion_dataset.py
-----------------
Synthetic lesion HSI dataset for domain adaptation.

Why synthetic?
--------------
Real dermoscopy hyperspectral data (e.g., from specialised scanners) is not
publicly available in paired RGB+HSI format at scale.  We therefore simulate
lesion hyperspectral patches by modifying face-skin spectra to reflect the
known spectral differences of melanocytic lesions:

  Melanoma characteristics (vs healthy skin):
    • Higher melanin concentration  → stronger blue/UV absorption → lower
      reflectance in bands 1-8 (400-470 nm)
    • Increased vascularity (haemoglobin bands ~540 nm and ~577 nm depressed)
    • Reduced overall reflectance due to denser pigmentation
    • Spectral irregularities from heterogeneous chromophore distribution

  Benign naevus characteristics:
    • Elevated melanin (less than melanoma)
    • More uniform distribution
    • Moderate haemoglobin signal

This module:
  1. Loads face HSI patches as the healthy-skin base.
  2. Generates lesion-modified patches via spectral perturbation.
  3. Returns a LesionDataset compatible with PyTorch DataLoader.
  4. Also provides a real-image path for when ISIC RGB images are available.

ISIC Integration (optional)
----------------------------
If ISIC dermoscopy images (.jpg) are placed in `data/isic/`:
    data/isic/melanoma/   ← melanoma images
    data/isic/benign/     ← benign naevus images

The dataset will synthesise HSI from those RGB images instead of face patches.
Otherwise it falls back to synthetic generation from face HSI.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ── Spectral perturbation functions ───────────────────────────────────────────

def _melanin_absorption_profile() -> np.ndarray:
    """Returns normalised melanin absorption across 31 bands (400-700 nm)."""
    return np.array([
        1.000, 0.916, 0.841, 0.772, 0.708, 0.650, 0.596, 0.547, 0.502, 0.460,
        0.422, 0.387, 0.355, 0.326, 0.299, 0.274, 0.252, 0.231, 0.212, 0.195,
        0.179, 0.164, 0.151, 0.138, 0.127, 0.117, 0.107, 0.098, 0.090, 0.083, 0.076
    ], dtype=np.float32)


def _hbo2_absorption_profile() -> np.ndarray:
    """Returns normalised HbO2 absorption across 31 bands."""
    return np.array([
        0.781, 0.945, 0.872, 0.639, 0.422, 0.290, 0.206, 0.173, 0.169, 0.184,
        0.208, 0.238, 0.265, 0.281, 0.279, 0.261, 0.232, 0.197, 0.159, 0.122,
        0.091, 0.068, 0.052, 0.042, 0.037, 0.035, 0.036, 0.038, 0.040, 0.041, 0.040
    ], dtype=np.float32)


def apply_melanoma_perturbation(
    hsi_patch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates melanoma spectral signature on a (H, W, 31) HSI patch.

    Modifications:
      - Multiply by (1 - mel_factor * mel_absorption) to simulate dense melanin
      - Add haemoglobin depression at characteristic bands
      - Add spatial heterogeneity (different melanin in lesion core vs border)
      - Random overall darkening (0.4-0.7 of original intensity)
    """
    H, W, C = hsi_patch.shape
    mel_abs  = _melanin_absorption_profile()
    hbo2_abs = _hbo2_absorption_profile()

    # spatial mask: elliptical lesion core
    cy, cx   = H // 2, W // 2
    ry, rx   = H * rng.uniform(0.25, 0.45), W * rng.uniform(0.25, 0.45)
    yy, xx   = np.ogrid[:H, :W]
    dist     = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    core_mask = (dist <= 1.0).astype(np.float32)   # (H, W)
    border_mask = np.clip((1.5 - dist), 0, 1).astype(np.float32) * (1 - core_mask)

    result = hsi_patch.copy().astype(np.float32)

    for h in range(H):
        for w in range(W):
            spectrum = result[h, w, :]

            if core_mask[h, w] > 0:
                # dense melanin in core
                mel_factor  = rng.uniform(0.5, 0.8)
                hb_factor   = rng.uniform(0.2, 0.4)
                dark_factor = rng.uniform(0.35, 0.55)
            elif border_mask[h, w] > 0.1:
                # moderate melanin in border transition
                mel_factor  = rng.uniform(0.2, 0.4)
                hb_factor   = rng.uniform(0.1, 0.2)
                dark_factor = rng.uniform(0.55, 0.75)
            else:
                continue  # no change outside lesion

            absorption   = mel_factor * mel_abs + hb_factor * hbo2_abs
            spectrum     = spectrum * (1.0 - absorption) * dark_factor
            # small spectral noise
            spectrum    += rng.normal(0, spectrum.std() * 0.02, C).astype(np.float32)
            result[h, w, :] = np.clip(spectrum, 0, None)

    return result


def apply_benign_perturbation(
    hsi_patch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulates benign naevus spectral signature.
    Less extreme than melanoma: uniform moderate melanin, less vascular change.
    """
    H, W, C = hsi_patch.shape
    mel_abs  = _melanin_absorption_profile()

    # smoother, more uniform lesion
    cy, cx  = H // 2, W // 2
    r       = min(H, W) * rng.uniform(0.2, 0.4)
    yy, xx  = np.ogrid[:H, :W]
    dist    = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask    = np.clip((r - dist) / r, 0, 1).astype(np.float32)  # smooth falloff

    result = hsi_patch.copy().astype(np.float32)

    mel_factor  = rng.uniform(0.15, 0.35)
    dark_factor = rng.uniform(0.65, 0.85)

    for h in range(H):
        for w in range(W):
            if mask[h, w] < 0.05:
                continue
            spectrum = result[h, w, :]
            absorption = mask[h, w] * mel_factor * mel_abs
            spectrum   = spectrum * (1.0 - absorption) * (dark_factor + (1 - mask[h, w]) * 0.1)
            spectrum  += rng.normal(0, spectrum.std() * 0.01, C).astype(np.float32)
            result[h, w, :] = np.clip(spectrum, 0, None)

    return result


# ── Synthetic lesion dataset builder ─────────────────────────────────────────

def build_lesion_patches(
    hsi_patches: list,
    n_melanoma: int = 50,
    n_benign:   int = 50,
    seed:       int = 42,
) -> tuple[list, list]:
    """
    Given a list of healthy HSI patches, generates synthetic lesion patches.

    Returns
    -------
    patches : list of np.ndarray (H, W, 31)
    labels  : list of int  {0 = benign, 1 = melanoma}
    """
    rng = np.random.default_rng(seed)
    patches, labels = [], []

    # sample base patches randomly
    base_indices = rng.choice(len(hsi_patches), size=n_melanoma + n_benign, replace=True)

    for i, idx in enumerate(base_indices):
        base = hsi_patches[idx]
        if i < n_melanoma:
            p = apply_melanoma_perturbation(base, rng)
            patches.append(p)
            labels.append(1)   # melanoma
        else:
            p = apply_benign_perturbation(base, rng)
            patches.append(p)
            labels.append(0)   # benign

    print(f"  Generated {n_melanoma} melanoma + {n_benign} benign lesion patches")
    return patches, labels


# ── ISIC real-image path (optional) ──────────────────────────────────────────

def load_isic_images(
    isic_dir: str,
    patch_size: int = 64,
    n_per_class: int = 50,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    If ISIC images exist, loads them and returns (rgb_patches, None, labels).
    HSI is None because we only have RGB — the reconstruction model will
    produce the HSI.

    Returns (rgb_patches, hsi_patches, labels) where hsi_patches may be None.
    """
    mel_dir    = os.path.join(isic_dir, "melanoma")
    benign_dir = os.path.join(isic_dir, "benign")

    if not (os.path.isdir(mel_dir) and os.path.isdir(benign_dir)):
        return None, None, None

    rng = np.random.default_rng(seed)

    def load_class(folder, label, n):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = rng.choice(files, size=min(n, len(files)), replace=False).tolist()
        patches, lbls = [], []
        for fname in files:
            img = np.array(Image.open(os.path.join(folder, fname)).convert("RGB").resize(
                (patch_size, patch_size), Image.BILINEAR
            ))
            patches.append(img)
            lbls.append(label)
        return patches, lbls

    mel_patches,    mel_labels    = load_class(mel_dir,    1, n_per_class)
    benign_patches, benign_labels = load_class(benign_dir, 0, n_per_class)

    all_patches = mel_patches + benign_patches
    all_labels  = mel_labels  + benign_labels
    print(f"  Loaded {len(mel_patches)} melanoma + {len(benign_patches)} benign ISIC patches")
    return all_patches, None, all_labels


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class LesionDataset(Dataset):
    """
    Dataset for domain adaptation.  Returns (rgb_tensor, hsi_tensor, label).

    rgb_tensor : (3,  H, W) float32  in [0, 1]
    hsi_tensor : (31, H, W) float32  z-score normalised
    label      : int  {0 = benign, 1 = melanoma}

    If rgb_patches is provided without hsi_patches (ISIC mode), hsi_tensor
    is a zero tensor (the model will generate HSI on-the-fly during adapt.py).
    """

    def __init__(
        self,
        hsi_patches: list,          # list of (H,W,31) arrays
        labels:      list,
        hsi_stats:   dict,
        rgb_patches: list = None,   # optional, if available
    ):
        self.hsi_patches = hsi_patches
        self.labels      = labels
        self.hsi_stats   = hsi_stats
        self.rgb_patches = rgb_patches

    def __len__(self):
        return len(self.labels)

    def _normalise_hsi(self, hsi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(hsi, dtype=np.float32)
        for b in range(hsi.shape[-1]):
            m, s = self.hsi_stats[str(b)]
            out[..., b] = (hsi[..., b] - m) / (s + 1e-8)
        return out

    def __getitem__(self, idx):
        label = self.labels[idx]

        if self.rgb_patches is not None:
            # RGB mode (ISIC)
            rgb = self.rgb_patches[idx].astype(np.float32) / 255.0
            rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))
        else:
            # synthesise a rough RGB from HSI bands 7, 15, 25 (400+70=470nm ≈ B, 550nm ≈ G, 650nm ≈ R)
            hsi_raw = self.hsi_patches[idx].astype(np.float32)
            r = np.clip(hsi_raw[..., 24] / 65535.0, 0, 1)
            g = np.clip(hsi_raw[..., 14] / 65535.0, 0, 1)
            b_ch = np.clip(hsi_raw[..., 6]  / 65535.0, 0, 1)
            rgb_t = torch.from_numpy(np.stack([r, g, b_ch], axis=0).astype(np.float32))

        hsi     = self.hsi_patches[idx].astype(np.float32) if self.hsi_patches else \
                  np.zeros((64, 64, 31), dtype=np.float32)
        hsi_n   = self._normalise_hsi(hsi)
        hsi_t   = torch.from_numpy(hsi_n.transpose(2, 0, 1))

        return rgb_t, hsi_t, torch.tensor(label, dtype=torch.long)
