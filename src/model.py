import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual Block with Adaptive Group Normalization (AdaGN) conditioning.
    Scale and shift coefficients are predicted from time/temperature embeddings.
    """
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        
        self.ada_emb_proj = nn.Linear(emb_dim, out_channels * 2)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)

        h = self.conv2(h)
        h = self.norm2(h)

        scale, shift = self.ada_emb_proj(emb)[:, :, None, None].chunk(2, dim=1)
        h = h * (1 + scale) + shift

        return F.gelu(h + self.shortcut(x))


class SinusoidalEmbedding(nn.Module):
    """
    Transforms continuous values into high-dimensional sinusoidal features.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        args = x[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TEmbedding(nn.Module):
    """
    Multi-layer Perceptron to project sinusoidal embeddings.
    """
    def __init__(self, sin_dim, out_dim):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(self.sinusoidal(x))


class SelfAttention(nn.Module):
    """
    Spatial Self-Attention block.
    Captures long-range spatial correlations near the critical point.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.view(B, C, H * W).permute(0, 2, 1)
        
        normed = self.ln(flat)
        attn, _ = self.mha(normed, normed, normed)
        flat = flat + attn
        flat = flat + self.ff(flat)
        
        return flat.permute(0, 2, 1).view(B, C, H, W)


class UNet_S(nn.Module):
    """
    Conditional UNet architecture with self-attention and bilinear upsampling.
    Predicts both noise mean (channel 0) and variance projection (channel 1).
    """
    def __init__(self, emb_dim=256):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.time_embed = TEmbedding(sin_dim=128, out_dim=self.emb_dim)
        self.temp_embed = TEmbedding(sin_dim=128, out_dim=self.emb_dim)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Downsampling)
        self.down_conv1 = ResBlock(1, 64, self.emb_dim)
        self.down_conv2 = ResBlock(64, 128, self.emb_dim)
        self.sa1 = SelfAttention(128)
        self.down_conv3 = ResBlock(128, 256, self.emb_dim)
        self.sa2 = SelfAttention(256)

        # Bottleneck
        self.bottleneck1 = ResBlock(256, 512, self.emb_dim)
        self.sa3 = SelfAttention(512)
        self.bottleneck2 = ResBlock(512, 512, self.emb_dim)

        # Decoder (Upsampling)
        self.up_trans1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        )
        self.up_conv1 = ResBlock(512, 256, self.emb_dim)
        self.sa4 = SelfAttention(256)
        
        self.up_trans2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        )
        self.up_conv2 = ResBlock(256, 128, self.emb_dim)
        self.sa5 = SelfAttention(128)
        
        self.up_trans3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        )
        self.up_conv3 = ResBlock(128, 64, self.emb_dim)

        self.out = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)

    def forward(self, x, t, T_phys):
        t_emb = self.time_embed(t)
        T_emb = self.temp_embed(T_phys)
        emb = t_emb + T_emb

        # Encoder
        x1 = self.down_conv1(x, emb)
        x2 = self.maxpool(x1)
        
        x3 = self.down_conv2(x2, emb)
        x3 = self.sa1(x3)
        x4 = self.maxpool(x3)
        
        x5 = self.down_conv3(x4, emb)
        x5 = self.sa2(x5)
        x6 = self.maxpool(x5)

        # Bottleneck
        x7 = self.bottleneck1(x6, emb)
        x7 = self.sa3(x7)
        x7 = self.bottleneck2(x7, emb)

        # Decoder
        x = self.up_trans1(x7)
        x = torch.cat([x5, x], dim=1)
        x = self.up_conv1(x, emb)
        x = self.sa4(x)

        x = self.up_trans2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.up_conv2(x, emb)
        x = self.sa5(x)

        x = self.up_trans3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up_conv3(x, emb)

        return self.out(x)
