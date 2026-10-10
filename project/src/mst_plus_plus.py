"""
mst_plus_plus.py
----------------
MST++ (Multi-Scale Spectral Transformer) for RGB → HSI reconstruction.

Based on:
  "MST++: Multi-stage Spectral-wise Transformer for Efficient Spectral
   Super-Resolution" (Cai et al., CVPRW 2022)

Architecture summary
--------------------
- Input RGB (B,3,H,W) is projected to a spectral embedding via a 1×1 conv.
- Three encoder stages each apply Spectral-wise Multi-head Self-Attention
  (S-MSA) which computes attention ACROSS the spectral dimension rather
  than the spatial one.  This is efficient (O(C²HW) instead of O(H²W²))
  and naturally captures inter-band correlations.
- A symmetric decoder with skip connections reconstructs at full resolution.
- Final 1×1 conv head maps to 31 output bands.

Key difference from plain CNN baseline
---------------------------------------
Attention is computed along the channel (spectral) axis so the model
explicitly learns how bands relate to each other — important for
physically consistent spectral reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Spectral-wise Multi-head Self-Attention ───────────────────────────────────

class SpectralAttention(nn.Module):
    """
    Computes self-attention across the spectral (channel) dimension.

    For an input of shape (B, C, H, W):
      - Flatten spatial dims → (B, C, H*W)
      - Treat C as the 'sequence' and H*W as the 'feature' dimension
      - Compute Q, K, V projections along C
      - Multi-head attention → output (B, C, H, W)

    This is O(C² · HW) — much cheaper than spatial attention for large images.
    """

    def __init__(self, channels: int, num_heads: int = 1, reduction: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        # project to Q, K, V along channel axis
        self.qkv  = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels,     bias=False)

        # channel-wise gating to control information flow
        self.gate = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # (B, C, H, W) → (B, H*W, C)  treat pixels as batch elements
        x_flat = x.flatten(2).transpose(1, 2)   # (B, HW, C)

        # QKV along C
        qkv = self.qkv(x_flat)                  # (B, HW, 3C)
        q, k, v = qkv.chunk(3, dim=-1)          # each (B, HW, C)

        # reshape for multi-head: (B, HW, heads, head_dim) → (B, heads, HW, head_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # scaled dot-product attention over spatial positions for each head
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                                 # (B, heads, HW, head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')    # (B, HW, C)

        # gating
        gate = self.gate(x_flat)
        out  = out * gate

        out = self.proj(out)                             # (B, HW, C)
        out = out.transpose(1, 2).reshape(B, C, H, W)   # (B, C, H, W)
        return out


# ── Feed-Forward Network ──────────────────────────────────────────────────────

class SpectralFFN(nn.Module):
    """
    Depth-wise separable conv FFN operating on spectral features.
    Uses GELU activation (same as original MST++ paper).
    """

    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden,   kernel_size=1),
            nn.GELU(),
            # depth-wise conv for local spatial mixing
            nn.Conv2d(hidden,   hidden,   kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden,   channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Spectral Transformer Block ────────────────────────────────────────────────

class SpectralTransformerBlock(nn.Module):
    """
    One transformer block:  LayerNorm → S-MSA → residual
                          + LayerNorm → FFN    → residual
    LayerNorm is applied over the channel dimension (spectral).
    """

    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)   # equivalent to LayerNorm over C
        self.attn  = SpectralAttention(channels, num_heads)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn   = SpectralFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ── Encoder / Decoder stages ──────────────────────────────────────────────────

class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 2, num_heads: int = 1):
        super().__init__()
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(
            *[SpectralTransformerBlock(out_ch, num_heads) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(self.downsample(x))


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2, num_heads: int = 1):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(
            *[SpectralTransformerBlock(out_ch + skip_ch, num_heads) for _ in range(num_blocks)]
        )
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1)

    def forward(self, x, skip):
        x   = self.upsample(x)
        x   = torch.cat([x, skip], dim=1)
        x   = self.blocks(x)
        return self.fuse(x)


# ── MST++ ─────────────────────────────────────────────────────────────────────

class MSTpp(nn.Module):
    """
    MST++ encoder-decoder with spectral transformer blocks and skip connections.

    Input  : (B, 3,  H, W)  normalised RGB
    Output : (B, 31, H, W)  predicted HSI (z-score space)

    Parameters
    ----------
    in_channels  : 3  (RGB)
    out_channels : 31 (spectral bands)
    base_ch      : channel width of first stage (doubles each level)
    num_blocks   : transformer blocks per stage
    num_heads    : attention heads (must divide base_ch)
    """

    def __init__(
        self,
        in_channels:  int = 3,
        out_channels: int = 31,
        base_ch:      int = 32,
        num_blocks:   int = 2,
        num_heads:    int = 1,
    ):
        super().__init__()

        # ── stem: project RGB to spectral embedding ──
        self.stem = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # ── encoder ──
        self.enc1 = EncoderStage(base_ch,     base_ch * 2, num_blocks, num_heads)
        self.enc2 = EncoderStage(base_ch * 2, base_ch * 4, num_blocks, num_heads)
        self.enc3 = EncoderStage(base_ch * 4, base_ch * 8, num_blocks, num_heads)

        # ── bottleneck ──
        self.bottleneck = nn.Sequential(
            *[SpectralTransformerBlock(base_ch * 8, num_heads) for _ in range(num_blocks)]
        )

        # ── decoder (with skip connections) ──
        self.dec3 = DecoderStage(base_ch * 8, base_ch * 4, base_ch * 4, num_blocks, num_heads)
        self.dec2 = DecoderStage(base_ch * 4, base_ch * 2, base_ch * 2, num_blocks, num_heads)
        self.dec1 = DecoderStage(base_ch * 2, base_ch,     base_ch,     num_blocks, num_heads)

        # ── output head ──
        self.head = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # stem
        s = self.stem(x)          # (B, C,   H,   W)

        # encoder
        e1 = self.enc1(s)         # (B, 2C,  H/2, W/2)
        e2 = self.enc2(e1)        # (B, 4C,  H/4, W/4)
        e3 = self.enc3(e2)        # (B, 8C,  H/8, W/8)

        # bottleneck
        b  = self.bottleneck(e3)  # (B, 8C,  H/8, W/8)

        # decoder
        d3 = self.dec3(b,  e2)    # (B, 4C,  H/4, W/4)
        d2 = self.dec2(d3, e1)    # (B, 2C,  H/2, W/2)
        d1 = self.dec1(d2, s)     # (B, C,   H,   W)

        return self.head(d1)      # (B, 31,  H,   W)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = MSTpp(base_ch=32, num_blocks=2, num_heads=1)
    print(f"Trainable parameters: {count_parameters(net):,}")
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {y.shape}")   # expected (2, 31, 64, 64)
