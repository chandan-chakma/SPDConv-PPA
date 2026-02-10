"""
MSP-YOLO: Multi-Scale Position-aware YOLO
FINAL PRODUCTION VERSION.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

# ==================== Component 1: Position Encoding ====================


class PositionEncoding2D(nn.Module):
    """2D position encoding for spatial information preservation."""

    def __init__(self, channels, height=80, width=80):
        super().__init__()
        self.channels = int(channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, int(channels), int(height), int(width)) * 0.02)

    def forward(self, x):
        B, _C, H, W = x.shape
        if H != self.pos_embedding.shape[2] or W != self.pos_embedding.shape[3]:
            pos_emb = F.interpolate(self.pos_embedding, size=(H, W), mode="bilinear", align_corners=False)
        else:
            pos_emb = self.pos_embedding
        return x + pos_emb.expand(B, -1, -1, -1)


class PAConv(nn.Module):
    """Position-Aware Convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.position_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=False), nn.BatchNorm2d(in_channels // 2), nn.SiLU()
        )
        self.pos_encoding = PositionEncoding2D(in_channels // 2)

        self.semantic_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(),
        )

        self.output_conv = Conv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        pos_feat = self.position_conv(x)
        pos_feat = self.pos_encoding(pos_feat)
        sem_feat = self.semantic_conv(x)
        fused = torch.cat([pos_feat, sem_feat], dim=1)
        return self.output_conv(fused)


# ==================== Component 2: Adaptive Multi-Kernel Pooling ====================


class AMKP(nn.Module):
    """Adaptive Multi-Kernel Pooling."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)

        self.pool_small = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), Conv(channels, channels, 1, 1))
        self.pool_medium = nn.Sequential(nn.MaxPool2d(5, stride=1, padding=2), Conv(channels, channels, 1, 1))
        self.pool_large = nn.Sequential(nn.MaxPool2d(7, stride=1, padding=3), Conv(channels, channels, 1, 1))

        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 3, 1),
            nn.Softmax(dim=1),
        )

        self.fusion = nn.Sequential(Conv(channels * 3, channels, 1, 1), Conv(channels, channels, 3, 1))

    def forward(self, x):
        feat_small = self.pool_small(x)
        feat_medium = self.pool_medium(x)
        feat_large = self.pool_large(x)

        scale_weights = self.scale_predictor(x)
        w_small = scale_weights[:, 0:1, :, :]
        w_medium = scale_weights[:, 1:2, :, :]
        w_large = scale_weights[:, 2:3, :, :]

        weighted_small = feat_small * w_small
        weighted_medium = feat_medium * w_medium
        weighted_large = feat_large * w_large

        multi_scale = torch.cat([weighted_small, weighted_medium, weighted_large], dim=1)
        output = self.fusion(multi_scale)
        return output + x


# ==================== Component 3: Scale-Selective Feature Pyramid ====================


class ScaleSelector(nn.Module):
    """Per-pixel scale selection."""

    def __init__(self, channels, num_scales=3):
        super().__init__()
        channels = int(channels)
        self.predictor = nn.Sequential(
            Conv(channels, channels // 2, 3, 1), Conv(channels // 2, num_scales, 1, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.predictor(x)


class SSFP(nn.Module):
    """Scale-Selective Feature Pyramid."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)

        self.scale_small = nn.Sequential(Conv(channels, channels, 3, 1), Conv(channels, channels, 1, 1))
        self.scale_medium = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            Conv(channels, channels, 1, 1),
        )
        self.scale_large = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            Conv(channels, channels, 1, 1),
        )

        self.selector = ScaleSelector(channels, num_scales=3)
        self.fusion = Conv(channels, channels, 1, 1)

    def forward(self, x):
        feat_small = self.scale_small(x)
        feat_medium = self.scale_medium(x)
        feat_large = self.scale_large(x)

        scale_map = self.selector(x)

        selected = (
            feat_small * scale_map[:, 0:1, :, :]
            + feat_medium * scale_map[:, 1:2, :, :]
            + feat_large * scale_map[:, 2:3, :, :]
        )

        output = self.fusion(selected)
        return output + x


# ==================== Main MSP Module ====================


class MSP(nn.Module):
    """Multi-Scale Position-aware Module."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.paconv = PAConv(in_channels, in_channels)
        self.amkp = AMKP(in_channels)
        self.ssfp = SSFP(in_channels)

        self.output = Conv(in_channels, out_channels, 1, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.paconv(x)
        x = self.amkp(x)
        x = self.ssfp(x)
        return self.output(x)


# ==================== YOLOv8 Integration ====================


class Bottleneck_MSP(nn.Module):
    """Bottleneck with MSP."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)
        g = int(g)
        e = float(e)

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.msp = MSP(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = self.msp(out)
        return x + out if self.add else out


class C2f_MSP(nn.Module):
    """C2f with MSP bottlenecks - PRODUCTION VERSION."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()

        # Robust type conversion
        c1 = int(c1)
        c2 = int(c2)
        n = int(n)
        g = int(g)
        e = float(e)

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_MSP(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = ["AMKP", "MSP", "SSFP", "Bottleneck_MSP", "C2f_MSP", "PAConv"]
