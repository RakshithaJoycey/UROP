"""
bio_loss.py
-----------
Physics-informed bio-optical loss terms for RGB → HSI reconstruction.

Background
----------
Skin reflectance is governed by the two-layer Kubelka-Munk / Beer-Lambert
model.  The dominant chromophores in the visible range are:

  • Melanin      – broadband absorber, strongest in UV/blue, falls off
                   monotonically toward NIR.
  • Oxyhaemoglobin (HbO2) – characteristic Soret band ~415 nm, Q-bands
                   ~542 nm and ~577 nm.
  • Deoxyhaemoglobin (Hb) – Q-bands at ~556 nm and ~760 nm.

Reference extinction spectra are taken from the well-known
Prahl / Oregon Medical Laser Center tabulation, sampled at the 31
wavelengths used by the CAVE multispectral camera:
    400, 410, 420, … 700 nm  (31 bands at 10 nm spacing)

Physics constraints implemented
---------------------------------
1. Chromophore constraint (ChromophoreConsistencyLoss)
   --------------------------------------------------
   For each pixel we estimate the melanin and haemoglobin contributions by
   least-squares fitting the predicted spectrum against the known extinction
   basis.  The residual (unexplained variance) acts as a regularisation term:
   a physically plausible spectrum should be *well explained* by the basis.

2. Monotonicity constraint (MelaninMonotonicityLoss)
   -------------------------------------------------
   Melanin absorption decreases monotonically from 400 nm to 700 nm.
   We penalise predicted spectra that show spectral peaks inconsistent
   with this monotonic absorption profile (i.e. when reflectance dips
   in a band where melanin dominates and the dip is not predicted).
   Implemented as a soft penalty on positive first-differences of the
   estimated melanin component.

3. Smoothness constraint (SpectralSmoothnessLoss)
   -----------------------------------------------
   Physical spectra are smooth — rapid oscillations between adjacent bands
   are artefacts.  We penalise the L1 norm of the second derivative of
   the predicted spectrum along the spectral axis.

Usage
-----
    from bio_loss import BioOpticalLoss
    criterion = BioOpticalLoss(lambda_bio=0.1)
    loss = criterion(pred_hsi, gt_hsi)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Chromophore extinction spectra (400–700 nm, 31 bands @ 10 nm steps) ──────
# Values are normalised to [0,1] range for numerical stability.
# Source: Prahl tabulation (https://omlc.org/spectra/hemoglobin/)
# Melanin source: Jacques SL, "Optical properties of biological tissues" PMB 2013

_WAVELENGTHS_NM = np.arange(400, 710, 10, dtype=np.float32)  # 31 values

# Melanin: monotonically decreasing power-law  μ_mel ∝ λ^{-3.48}
_MEL_RAW = np.array([
    1.000, 0.916, 0.841, 0.772, 0.708, 0.650, 0.596, 0.547, 0.502, 0.460,
    0.422, 0.387, 0.355, 0.326, 0.299, 0.274, 0.252, 0.231, 0.212, 0.195,
    0.179, 0.164, 0.151, 0.138, 0.127, 0.117, 0.107, 0.098, 0.090, 0.083, 0.076
], dtype=np.float32)

# Oxyhaemoglobin (HbO2) — normalised peak ≈ 415 nm
_HBO2_RAW = np.array([
    0.781, 0.945, 0.872, 0.639, 0.422, 0.290, 0.206, 0.173, 0.169, 0.184,
    0.208, 0.238, 0.265, 0.281, 0.279, 0.261, 0.232, 0.197, 0.159, 0.122,
    0.091, 0.068, 0.052, 0.042, 0.037, 0.035, 0.036, 0.038, 0.040, 0.041, 0.040
], dtype=np.float32)

# Deoxyhaemoglobin (Hb) — normalised peak ≈ 430 nm
_HB_RAW = np.array([
    0.612, 0.748, 0.851, 0.917, 0.964, 1.000, 0.942, 0.848, 0.733, 0.614,
    0.502, 0.407, 0.333, 0.283, 0.257, 0.256, 0.271, 0.295, 0.312, 0.309,
    0.281, 0.237, 0.184, 0.134, 0.093, 0.066, 0.050, 0.041, 0.037, 0.036, 0.036
], dtype=np.float32)


def _build_basis() -> torch.Tensor:
    """
    Returns basis matrix E of shape (31, 3):
        E[:, 0] = melanin extinction
        E[:, 1] = HbO2  extinction
        E[:, 2] = Hb    extinction
    Each column is L2-normalised for numerical stability.
    """
    mel  = _MEL_RAW  / (np.linalg.norm(_MEL_RAW)  + 1e-8)
    hbo2 = _HBO2_RAW / (np.linalg.norm(_HBO2_RAW) + 1e-8)
    hb   = _HB_RAW   / (np.linalg.norm(_HB_RAW)   + 1e-8)
    basis = np.stack([mel, hbo2, hb], axis=1)   # (31, 3)
    return torch.from_numpy(basis)               # float32


# ── Individual loss modules ───────────────────────────────────────────────────

class ChromophoreConsistencyLoss(nn.Module):
    """
    Penalises predicted spectra that cannot be explained by the
    melanin + HbO2 + Hb basis.

    For each pixel p_i (31-dim vector):
      1. Compute least-squares fit: c = pinv(E) @ p_i  (3 chromophore weights)
      2. Reconstruct: p_hat = E @ c
      3. Loss = mean( ||p_i - p_hat||² )   over all pixels

    A small residual means the spectrum is physically plausible.
    """

    def __init__(self):
        super().__init__()
        basis = _build_basis()            # (31, 3)
        # pseudo-inverse: (3, 31)
        pinv  = torch.linalg.pinv(basis)  # (3, 31)
        self.register_buffer("basis", basis)
        self.register_buffer("pinv",  pinv)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        pred : (B, 31, H, W)  predicted HSI in z-score or raw space
        """
        B, C, H, W = pred.shape
        # flatten to (B*H*W, 31)
        p = pred.permute(0, 2, 3, 1).reshape(-1, C)        # (N, 31)

        # chromophore weights c = pinv @ p.T → (3, N)
        c = (self.pinv @ p.T)                               # (3, N)

        # enforce non-negative chromophore concentrations (ReLU)
        c = F.relu(c)

        # reconstruct spectrum from chromophore weights
        p_hat = (self.basis @ c).T                          # (N, 31)

        loss = F.mse_loss(p_hat, p)
        return loss


class SpectralSmoothnessLoss(nn.Module):
    """
    Penalises rapid spectral oscillations via L1 of the 2nd derivative
    along the band axis.

    d²p/dλ² ≈ p[b+1] - 2*p[b] + p[b-1]
    """

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """pred : (B, 31, H, W)"""
        # second derivative along channel dim
        d2 = pred[:, 2:, :, :] - 2 * pred[:, 1:-1, :, :] + pred[:, :-2, :, :]
        return d2.abs().mean()


class MelaninMonotonicityLoss(nn.Module):
    """
    Encourages the melanin-weighted portion of the spectrum to be
    monotonically decreasing from blue to red.

    Strategy:
      1. Compute estimated melanin contribution per pixel:
         c_mel = (mel_basis · pred) / ||mel_basis||²
      2. Melanin component: m_i = c_mel * mel_basis[i]
      3. Penalise *positive* first-differences: max(0, m[b+1] - m[b])
         i.e. penalise when the melanin component increases with wavelength
    """

    def __init__(self):
        super().__init__()
        mel_norm = torch.from_numpy(_MEL_RAW / (np.linalg.norm(_MEL_RAW) ** 2 + 1e-8))
        self.register_buffer("mel_basis", mel_norm)    # (31,)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """pred : (B, 31, H, W)"""
        B, C, H, W = pred.shape

        # melanin projection coefficient per pixel  (B, H, W)
        # c_mel = sum_b(pred[b] * mel[b])
        mel = self.mel_basis.view(1, C, 1, 1)          # (1, 31, 1, 1)
        c_mel = (pred * mel).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # melanin spectral component  (B, 31, H, W)
        mel_component = c_mel * mel

        # first differences along spectral axis: should be ≤ 0 (monotone decreasing)
        diff = mel_component[:, 1:, :, :] - mel_component[:, :-1, :, :]   # (B, 30, H, W)
        penalty = F.relu(diff)   # only penalise positive increases
        return penalty.mean()


# ── Combined bio-optical loss ─────────────────────────────────────────────────

class BioOpticalLoss(nn.Module):
    """
    Combines standard MSE reconstruction loss with bio-optical regularisers.

    Total loss = MSE(pred, gt)
               + λ_chrom * ChromophoreConsistencyLoss(pred)
               + λ_smooth * SpectralSmoothnessLoss(pred)
               + λ_mono   * MelaninMonotonicityLoss(pred)

    Parameters
    ----------
    lambda_chrom  : weight for chromophore consistency  (default 0.05)
    lambda_smooth : weight for spectral smoothness      (default 0.01)
    lambda_mono   : weight for melanin monotonicity     (default 0.01)

    The defaults keep bio losses as gentle regularisers so they guide
    the model without overwhelming the data-fitting term.
    """

    def __init__(
        self,
        lambda_chrom:  float = 0.05,
        lambda_smooth: float = 0.01,
        lambda_mono:   float = 0.01,
    ):
        super().__init__()
        self.lambda_chrom  = lambda_chrom
        self.lambda_smooth = lambda_smooth
        self.lambda_mono   = lambda_mono

        self.chrom_loss  = ChromophoreConsistencyLoss()
        self.smooth_loss = SpectralSmoothnessLoss()
        self.mono_loss   = MelaninMonotonicityLoss()

    def forward(
        self,
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor
        breakdown  : dict with individual loss values (for logging)
        """
        mse    = F.mse_loss(pred, gt)
        chrom  = self.chrom_loss(pred)
        smooth = self.smooth_loss(pred)
        mono   = self.mono_loss(pred)

        total = (
            mse
            + self.lambda_chrom  * chrom
            + self.lambda_smooth * smooth
            + self.lambda_mono   * mono
        )

        breakdown = {
            "mse":    mse.item(),
            "chrom":  chrom.item(),
            "smooth": smooth.item(),
            "mono":   mono.item(),
            "total":  total.item(),
        }
        return total, breakdown


if __name__ == "__main__":
    criterion = BioOpticalLoss()
    pred = torch.randn(2, 31, 64, 64)
    gt   = torch.randn(2, 31, 64, 64)
    loss, breakdown = criterion(pred, gt)
    print("Loss breakdown:", {k: f"{v:.4f}" for k, v in breakdown.items()})
