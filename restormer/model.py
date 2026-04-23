"""Restormer: Efficient Transformer for High-Resolution Image Restoration.

Faithful PyTorch re-implementation of Zamir et al., CVPR 2022
(https://arxiv.org/abs/2111.09881).

Main building blocks:
    - MDTA  (Multi-Dconv Head Transposed Attention): channel-wise self-attention
            with depthwise-conv Q/K/V projections, keeps cost linear in tokens.
    - GDFN  (Gated-Dconv Feed-Forward Network): depthwise-conv + gating replaces
            the standard MLP.
    - Hierarchical 4-level encoder/decoder with pixel-unshuffle/shuffle sampling,
      skip connections, and a final refinement stage.

The module is written so the `dim`, `num_blocks`, `heads`, `ffn_expansion`
defaults match the paper's "Restormer" config, but every shape is configurable
so we can train a smaller variant when no GPU is available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer primitives
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for (B, C, H, W) tensors (bias-free by default)."""

    def __init__(self, channels: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x


# ---------------------------------------------------------------------------
# Attention (MDTA) and feed-forward (GDFN)
# ---------------------------------------------------------------------------


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.

    Computes attention across the channel dimension instead of spatial tokens,
    giving O(C^2 * H * W) cost rather than O((HW)^2 * C).
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim: int, ffn_expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_expansion: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim, ffn_expansion=ffn_expansion, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Down/Up sampling via pixel (un)shuffle (halves/doubles resolution, 1 conv).
# ---------------------------------------------------------------------------


class Downsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ---------------------------------------------------------------------------
# Full Restormer
# ---------------------------------------------------------------------------


class Restormer(nn.Module):
    """4-level encoder-decoder Restormer.

    Args:
        in_channels:   number of input channels (1 for grayscale, 3 for RGB)
        out_channels:  number of output channels (usually same as in)
        dim:           base feature dim; doubles each level down
        num_blocks:    blocks per level `(enc1, enc2, enc3, latent)`
        num_refinement_blocks: blocks in the final refinement stage
        heads:         attention heads at each of the 4 levels
        ffn_expansion: GDFN inner ratio
        bias:          convolution bias flag

    Default hyper-parameters match the paper's main "Restormer" config.
    Pass `dim=24, num_blocks=(2, 2, 2, 2), num_refinement_blocks=2` for a
    lighter "Restormer-S" that can train on MPS/CPU.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: tuple = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        heads: tuple = (1, 2, 4, 8),
        ffn_expansion: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()
        if len(num_blocks) != 4 or len(heads) != 4:
            raise ValueError("num_blocks and heads must each have length 4")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # Encoder
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion, bias) for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias) for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias) for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)

        # Latent
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion, bias) for _ in range(num_blocks[3])
        ])

        # Decoder
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion, bias) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion, bias) for _ in range(num_blocks[1])
        ])

        # The first decoder level intentionally keeps 2*dim channels (paper sec 3.2)
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion, bias) for _ in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion, bias) for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim * 2, out_channels, kernel_size=3, padding=1, bias=bias)

    # ------------------------------------------------------------------
    # Padding helpers: the 4-level U-Net requires H, W divisible by 8.
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_multiple(x: torch.Tensor, mult: int = 8) -> tuple:
        _, _, h, w = x.shape
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0, 0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (0, pad_w, 0, pad_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h0, w0 = x.shape
        x_in, pad = self._pad_multiple(x, mult=8)

        inp = self.patch_embed(x_in)

        e1 = self.encoder_level1(inp)
        x2 = self.down1_2(e1)
        e2 = self.encoder_level2(x2)
        x3 = self.down2_3(e2)
        e3 = self.encoder_level3(x3)
        x4 = self.down3_4(e3)

        z = self.latent(x4)

        d3 = self.up4_3(z)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.reduce_chan_level3(d3)
        d3 = self.decoder_level3(d3)

        d2 = self.up3_2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.reduce_chan_level2(d2)
        d2 = self.decoder_level2(d2)

        d1 = self.up2_1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder_level1(d1)

        d1 = self.refinement(d1)
        out = self.output(d1) + x_in  # global residual: predict blur minus ground truth

        if pad != (0, 0, 0, 0):
            out = out[..., :h0, :w0]
        return out


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def restormer_small(in_channels: int = 1, out_channels: int = 1) -> Restormer:
    """Lightweight variant for CPU/MPS training (~2-3 M params)."""
    return Restormer(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=24,
        num_blocks=(2, 2, 2, 2),
        num_refinement_blocks=2,
        heads=(1, 2, 4, 8),
        ffn_expansion=2.0,
    )


def restormer_base(in_channels: int = 1, out_channels: int = 1) -> Restormer:
    """Paper default (~26 M params)."""
    return Restormer(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion=2.66,
    )


def build_model(variant: str, in_channels: int = 1, out_channels: int = 1) -> Restormer:
    variant = variant.lower()
    if variant in ("small", "s"):
        return restormer_small(in_channels, out_channels)
    if variant in ("base", "b", "paper"):
        return restormer_base(in_channels, out_channels)
    raise ValueError(f"Unknown Restormer variant '{variant}', expected 'small' or 'base'")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ("small", "base"):
        m = build_model(name)
        n = count_params(m)
        x = torch.randn(1, 1, 200, 200)
        with torch.no_grad():
            y = m(x)
        print(f"{name:>5}: {n/1e6:.2f} M params, in {tuple(x.shape)} -> out {tuple(y.shape)}")
