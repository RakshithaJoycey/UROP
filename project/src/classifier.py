"""
classifier.py
-------------
Melanoma vs benign classifier operating on reconstructed HSI features.

Architecture
------------
The classifier takes the 31-band reconstructed HSI as input and predicts
binary label: 0 = benign, 1 = melanoma.

Two components:

1. SpectralFeatureExtractor
   - 1×1 convolutions to learn spectral-band combinations
   - Global average pooling → spectral feature vector
   - Captures which wavelength combinations are discriminative

2. SpatialSpectralClassifier  (full model)
   - Shared spectral CNN encoder (reuses some MST++ encoder layers)
   - Spatial pooling
   - MLP classification head
   - Outputs logits for {benign=0, melanoma=1}

The classifier can be:
  (a) trained from scratch on lesion HSI features
  (b) fine-tuned with a frozen/partial MST++ encoder as backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Spectral Feature Extractor ────────────────────────────────────────────────

class SpectralFeatureExtractor(nn.Module):
    """
    Lightweight CNN that maps (B, 31, H, W) → (B, feature_dim).

    Uses 1×1 convolutions to mix spectral bands, then spatial pooling.
    Inspired by spectral unmixing — learn which band combinations matter.
    """

    def __init__(self, in_bands: int = 31, feature_dim: int = 128):
        super().__init__()

        self.spectral_mix = nn.Sequential(
            # spectral mixing layers (1×1 conv = per-pixel band weighting)
            nn.Conv2d(in_bands, 64,  kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,       64,  kernel_size=3, padding=1, groups=64),  # spatial context
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64,       feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling → (B, feature_dim, 1, 1)

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        """hsi: (B, 31, H, W)  →  (B, feature_dim)"""
        x = self.spectral_mix(hsi)
        x = self.pool(x).flatten(1)
        return x


# ── Full classifier ───────────────────────────────────────────────────────────

class MelanomaClassifier(nn.Module):
    """
    End-to-end melanoma vs benign classifier on HSI input.

    Input  : (B, 31, H, W)  reconstructed HSI (z-score space)
    Output : (B, 2)          logits [benign, melanoma]

    Parameters
    ----------
    in_bands    : number of spectral bands (31)
    feature_dim : intermediate feature dimension
    dropout     : dropout rate in classifier head
    """

    def __init__(
        self,
        in_bands:    int   = 31,
        feature_dim: int   = 128,
        dropout:     float = 0.3,
        num_classes: int   = 2,
    ):
        super().__init__()

        self.extractor = SpectralFeatureExtractor(in_bands, feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, hsi: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        features = self.extractor(hsi)
        return self.classifier(features)

    def predict_proba(self, hsi: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities (B, num_classes)."""
        return F.softmax(self.forward(hsi), dim=-1)


# ── Combined reconstruction + classification model ────────────────────────────

class ReconstructAndClassify(nn.Module):
    """
    Wraps MST++ + MelanomaClassifier into a single forward pass.

    During domain adaptation, this allows joint training of:
      - HSI reconstruction (MSE + bio-optical loss)
      - Melanoma classification (cross-entropy)

    Parameters
    ----------
    reconstructor : MSTpp instance
    classifier    : MelanomaClassifier instance
    """

    def __init__(self, reconstructor: nn.Module, classifier: nn.Module):
        super().__init__()
        self.reconstructor = reconstructor
        self.classifier    = classifier

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        rgb : (B, 3, H, W)

        Returns
        -------
        pred_hsi : (B, 31, H, W)  reconstructed HSI
        logits   : (B, 2)         classification logits
        """
        pred_hsi = self.reconstructor(rgb)
        logits   = self.classifier(pred_hsi)
        return pred_hsi, logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    clf = MelanomaClassifier()
    print(f"Classifier parameters: {count_parameters(clf):,}")
    hsi = torch.randn(4, 31, 64, 64)
    out = clf(hsi)
    print(f"Input  : {hsi.shape}")
    print(f"Output : {out.shape}")   # (4, 2)
