"""
model.py
--------
Simple CNN encoder-decoder for RGB → 31-band HSI reconstruction.

Architecture (proof-of-concept):
  Encoder: 3 conv blocks that progressively double channels
  Bottleneck: one conv block at the deepest level
  Decoder: 3 upsampling conv blocks back to original spatial size
  Head: 1×1 conv to produce 31 output channels

No skip connections are used here to keep the model simple.
We can add U-Net style skips in a later stage.
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv-BN-ReLU layers. Standard building block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SimpleHSINet(nn.Module):
    """
    Lightweight encoder-decoder CNN.
      Input  : (B, 3,  H, W)  – normalised RGB
      Output : (B, 31, H, W)  – predicted HSI (z-score space)

    Parameters
    ----------
    base_channels : int
        Channel width of the first encoder layer.
        Doubles at each encoder stage.
        Default 32 keeps the model small (~0.5 M params).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 31, base_ch: int = 32):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = conv_block(in_channels, base_ch)          # (B, 32,  H,   W)
        self.pool1 = nn.MaxPool2d(2)                           # (B, 32,  H/2, W/2)

        self.enc2 = conv_block(base_ch,    base_ch * 2)       # (B, 64,  H/2, W/2)
        self.pool2 = nn.MaxPool2d(2)                           # (B, 64,  H/4, W/4)

        self.enc3 = conv_block(base_ch*2,  base_ch * 4)       # (B, 128, H/4, W/4)
        self.pool3 = nn.MaxPool2d(2)                           # (B, 128, H/8, W/8)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = conv_block(base_ch*4, base_ch * 8)  # (B, 256, H/8, W/8)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3  = conv_block(base_ch*8, base_ch * 4)       # (B, 128, H/4, W/4)

        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2  = conv_block(base_ch*4, base_ch * 2)       # (B, 64,  H/2, W/2)

        self.up1   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1  = conv_block(base_ch*2, base_ch)            # (B, 32,  H,   W)

        # ── Output head ──────────────────────────────────────────────────────
        self.head  = nn.Conv2d(base_ch, out_channels, kernel_size=1)  # (B, 31, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool3(e3))

        # Decoder (no skip connections – intentionally simple)
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))

        return self.head(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = SimpleHSINet()
    print(net)
    print(f"Trainable parameters: {count_parameters(net):,}")
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {y.shape}")   # expected (2, 31, 64, 64)
