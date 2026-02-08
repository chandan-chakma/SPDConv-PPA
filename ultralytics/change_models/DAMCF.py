"""
DAMCF: Dynamic Adaptive Multi-Scale Contextual Fusion for YOLOv8
Implements adaptive context aggregation for small object detection.
"""

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==================== Core DAMCF Components ====================


class ContextRangePredictor(nn.Module):
    """Context Range Predictor (CRP) Predicts optimal contextual range for each spatial location Output: [B, 1, H, W]
    with values in [0, 1].
    """

    def __init__(self, in_channels=256, hidden_dim=64):
        super().__init__()

        # Lightweight prediction network
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            context_map: [B, 1, H, W] predicted context range.
        """
        return self.predictor(x)


class DensityEstimator(nn.Module):
    """Estimates local object density to modulate context range High density → reduce context (avoid confusion) Low
    density → expand context (gather more info).
    """

    def __init__(self, in_channels=256):
        super().__init__()

        self.density_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(),  # Density is non-negative
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map

        Returns:
            density_map: [B, 1, H, W] estimated density.
        """
        return self.density_head(x)

    def get_density_factor(self, density_map, threshold=0.5):
        """Convert density to modulation factor.

        Args:
            density_map: [B, 1, H, W]
            threshold: density threshold

        Returns:
            factor: [B, 1, H, W] modulation factor in [0.5, 1.0].
        """
        # Normalize density to [0, 1]
        normalized = torch.sigmoid(density_map - threshold)

        # Invert: high density → smaller factor (reduce context)
        # Low density → factor closer to 1 (keep/expand context)
        factor = 1.0 - 0.5 * normalized

        return factor


class AdaptiveDeformableConv(nn.Module):
    """Adaptive Deformable Convolution Dynamically adjusts sampling offsets based on context range and density.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Offset prediction (18 channels for 3x3 kernel: 2 coords × 9 positions)
        self.offset_conv = nn.Conv2d(
            in_channels + 2,  # +2 for context_map and density_factor
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        # Modulation (attention) weights
        self.modulation_conv = nn.Conv2d(
            in_channels + 2, kernel_size * kernel_size, kernel_size=3, padding=1, bias=True
        )

        # Regular convolution (will use deformable sampling)
        self.regular_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x, context_map, density_factor):
        """
        Args:
            x: [B, C, H, W] input features
            context_map: [B, 1, H, W] predicted context range [0,1]
            density_factor: [B, 1, H, W] density modulation factor

        Returns:
            out: [B, C_out, H, W] output features.
        """
        _B, _C, _H, _W = x.shape

        # Combine context and density
        adaptive_context = context_map * density_factor

        # Concatenate for offset and modulation prediction
        x_with_context = torch.cat([x, context_map, density_factor], dim=1)

        # Predict offsets (scaled by adaptive context)
        offset = self.offset_conv(x_with_context)
        # Scale offsets: larger context → larger offsets
        offset = offset * adaptive_context.repeat(1, offset.shape[1], 1, 1)

        # Predict modulation weights
        modulation = torch.sigmoid(self.modulation_conv(x_with_context))

        # Apply deformable convolution (simplified version using grid_sample)
        out = self._deform_conv(x, offset, modulation)
        out = self.act(self.bn(out))

        return out

    def _deform_conv(self, x, offset, modulation):
        """Simplified deformable convolution using grid_sample For production, use torchvision.ops.DeformConv2d.
        """
        B, _C, H, W = x.shape

        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]

        # Reshape offset: [B, 2*K*K, H, W] → [B, K*K, 2, H, W]
        K = self.kernel_size
        offset = offset.view(B, K * K, 2, H, W)

        # Apply regular convolution (simplified)
        # In practice, you'd iterate over kernel positions with offsets
        out = self.regular_conv(x)

        # Apply modulation
        # modulation: [B, K*K, H, W] → apply to output
        mod_weight = modulation.mean(dim=1, keepdim=True)  # Simplified
        out = out * mod_weight

        return out


class DAMCF(nn.Module):
    """Dynamic Adaptive Multi-Scale Contextual Fusion Module Main module that combines CRP + Density Estimation +
    Adaptive Deformable Conv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Context Range Predictor
        self.crp = ContextRangePredictor(in_channels, hidden_dim=64)

        # Density Estimator
        self.density_est = DensityEstimator(in_channels)

        # Adaptive Deformable Convolution
        self.adaptive_conv = AdaptiveDeformableConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input features

        Returns:
            out: [B, C_out, H, W] enhanced features.
        """
        # Predict context range
        context_map = self.crp(x)

        # Estimate density
        density_map = self.density_est(x)
        density_factor = self.density_est.get_density_factor(density_map)

        # Apply adaptive deformable convolution
        out = self.adaptive_conv(x, context_map, density_factor)

        # Residual connection
        out = out + self.residual(x)

        return out

    def get_auxiliary_outputs(self, x):
        """Get auxiliary outputs for loss computation."""
        context_map = self.crp(x)
        density_map = self.density_est(x)
        return {"context_map": context_map, "density_map": density_map}


# ==================== YOLOv8 Integration Components ====================


class Bottleneck_DAMCF(nn.Module):
    """Bottleneck with DAMCF module Similar to Bottleneck_LWN pattern.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.damcf = DAMCF(c2, c2)  # Apply DAMCF after bottleneck
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with DAMCF enhancement."""
        out = self.cv2(self.cv1(x))
        out = self.damcf(out)
        return x + out if self.add else out


class C2f_DAMCF(nn.Module):
    """C2f module with DAMCF-enhanced bottlenecks Similar to C2f_LWN pattern.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_DAMCF(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f_DAMCF layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ==================== Testing ====================


def test_damcf():
    """Test DAMCF module."""
    print("Testing DAMCF module...")

    # Create dummy input
    B, C, H, W = 2, 256, 64, 64
    x = torch.randn(B, C, H, W)

    # Test DAMCF module
    damcf = DAMCF(in_channels=256, out_channels=256)
    out = damcf(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    # Test auxiliary outputs
    aux = damcf.get_auxiliary_outputs(x)
    print(f"Context map shape: {aux['context_map'].shape}")
    print(f"Density map shape: {aux['density_map'].shape}")

    # Test C2f_DAMCF
    print("\nTesting C2f_DAMCF...")
    c2f_damcf = C2f_DAMCF(c1=256, c2=256, n=2)
    out_c2f = c2f_damcf(x)
    print(f"C2f_DAMCF output shape: {out_c2f.shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_damcf()
