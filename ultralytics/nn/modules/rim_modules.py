import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# For Mamba - install: pip install mamba-ssm
try:
    from mamba_ssm import Mamba
except:
    print("Installing mamba-ssm...")
    import subprocess

    subprocess.check_call(["pip", "install", "mamba-ssm"])
    from mamba_ssm import Mamba

__all__ = ["AdaptiveFrequencyGate", "C2f_RIM", "LightweightINR", "RecursiveMambaBlock"]


class LightweightINR(nn.Module):
    """Lightweight Implicit Neural Representation for continuous object detection."""

    def __init__(self, in_channels=256, hidden_dim=128, out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        # Positional encoding parameters
        self.pos_freq = nn.Parameter(torch.randn(1, 32, 1, 1) * 0.1)

        # Lightweight MLP for implicit representation
        self.input_proj = nn.Conv2d(in_channels + 64, hidden_dim, 1)
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GroupNorm(8, hidden_dim), nn.SiLU(inplace=True)
        )
        self.output_proj = nn.Conv2d(hidden_dim, out_channels, 1)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def generate_positional_encoding(self, H, W, device):
        """Generate continuous positional encoding."""
        y_coords = torch.linspace(-1, 1, H, device=device).view(H, 1).repeat(1, W)
        x_coords = torch.linspace(-1, 1, W, device=device).view(1, W).repeat(H, 1)

        coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0)  # [1, 2, H, W]

        # Apply learnable frequencies
        pos_features = []
        for i in range(16):
            freq = self.pos_freq[:, i : i + 1, :, :]
            pos_features.append(torch.sin(coords * freq * math.pi))
            pos_features.append(torch.cos(coords * freq * math.pi))

        return torch.cat(pos_features, dim=1)  # [1, 64, H, W]

    def forward(self, x):
        B, _C, H, W = x.shape

        # Generate positional encoding
        pos_enc = self.generate_positional_encoding(H, W, x.device)
        pos_enc = pos_enc.expand(B, -1, -1, -1)

        # Concatenate features with positional encoding
        x_with_pos = torch.cat([x, pos_enc], dim=1)

        # Apply implicit transformation
        out = self.input_proj(x_with_pos)
        out = self.hidden_layer(out)
        out = self.output_proj(out)

        # Add residual connection
        return out + self.residual(x)


class RecursiveMambaBlock(nn.Module):
    """Recursive Mamba block for iterative refinement."""

    def __init__(self, dim=256, d_state=16, d_conv=4, expand=2, num_iterations=3):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations

        # Shared Mamba block for all iterations
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Iteration-specific gates
        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GroupNorm(8, dim), nn.Sigmoid())
                for _ in range(num_iterations)
            ]
        )

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        _B, _C, H, W = x.shape

        # Flatten spatial dimensions for Mamba
        x_flat = rearrange(x, "b c h w -> b (h w) c")

        hidden = x_flat
        outputs = []

        for i in range(self.num_iterations):
            # Apply Mamba transformation
            new_hidden = self.mamba(hidden)

            # Reshape for gating
            hidden_2d = rearrange(hidden, "b (h w) c -> b c h w", h=H, w=W)
            new_hidden_2d = rearrange(new_hidden, "b (h w) c -> b c h w", h=H, w=W)

            # Apply gated skip connection
            gate = self.gates[i](torch.cat([hidden_2d, new_hidden_2d], dim=1))
            gated_2d = gate * new_hidden_2d + (1 - gate) * hidden_2d

            # Flatten back
            hidden = rearrange(gated_2d, "b c h w -> b (h w) c")
            outputs.append(gated_2d)

        # Multi-scale fusion
        fused = sum(outputs) / len(outputs)
        return self.out_proj(fused)


class WaveletTransform2d(nn.Module):
    """2D Wavelet Transform using PyTorch."""

    def __init__(self, wave="db3"):
        super().__init__()
        self.wave = wave

    def forward(self, x):
        B, C, _H, _W = x.shape

        # Process each batch and channel separately
        coeffs_batch = []
        for b in range(B):
            coeffs_channels = []
            for c in range(C):
                # Convert to numpy for pywt
                x_np = x[b, c].detach().cpu().numpy()

                # Apply 2D wavelet transform
                coeffs = pywt.dwt2(x_np, self.wave)
                LL, (LH, HL, HH) = coeffs

                # Convert back to tensors
                LL = torch.from_numpy(LL).to(x.device).float()
                LH = torch.from_numpy(LH).to(x.device).float()
                HL = torch.from_numpy(HL).to(x.device).float()
                HH = torch.from_numpy(HH).to(x.device).float()

                coeffs_channels.append([LL, LH, HL, HH])
            coeffs_batch.append(coeffs_channels)

        # Stack into tensors
        LL_batch = torch.stack([torch.stack([coeffs_batch[b][c][0] for c in range(C)]) for b in range(B)])
        LH_batch = torch.stack([torch.stack([coeffs_batch[b][c][1] for c in range(C)]) for b in range(B)])
        HL_batch = torch.stack([torch.stack([coeffs_batch[b][c][2] for c in range(C)]) for b in range(B)])
        HH_batch = torch.stack([torch.stack([coeffs_batch[b][c][3] for c in range(C)]) for b in range(B)])

        return LL_batch, LH_batch, HL_batch, HH_batch

    def inverse(self, LL, LH, HL, HH):
        B, C = LL.shape[:2]

        # Reconstruct each batch and channel
        x_batch = []
        for b in range(B):
            x_channels = []
            for c in range(C):
                # Convert to numpy
                LL_np = LL[b, c].detach().cpu().numpy()
                LH_np = LH[b, c].detach().cpu().numpy()
                HL_np = HL[b, c].detach().cpu().numpy()
                HH_np = HH[b, c].detach().cpu().numpy()

                # Apply inverse wavelet transform
                coeffs = (LL_np, (LH_np, HL_np, HH_np))
                x_np = pywt.idwt2(coeffs, self.wave)

                # Convert back to tensor
                x_channels.append(torch.from_numpy(x_np).to(LL.device).float())
            x_batch.append(torch.stack(x_channels))

        return torch.stack(x_batch)


class AdaptiveFrequencyGate(nn.Module):
    """Adaptive frequency selection using wavelet decomposition."""

    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels

        # Wavelet transform
        self.dwt = WaveletTransform2d(wave="db3")

        # Frequency attention mechanism
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 4, 1),
            nn.Sigmoid(),
        )

        # Channel-wise modulation
        self.channel_mod = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1), nn.GroupNorm(8, channels), nn.SiLU(inplace=True)
        )

    def forward(self, x):
        _B, C, H, W = x.shape

        # Wavelet decomposition
        LL, LH, HL, HH = self.dwt(x)

        # Resize high-frequency components to match LL
        h_ll, w_ll = LL.shape[-2:]
        LH = F.interpolate(LH, size=(h_ll, w_ll), mode="bilinear", align_corners=False)
        HL = F.interpolate(HL, size=(h_ll, w_ll), mode="bilinear", align_corners=False)
        HH = F.interpolate(HH, size=(h_ll, w_ll), mode="bilinear", align_corners=False)

        # Stack frequency components
        freq_stack = torch.cat([LL, LH, HL, HH], dim=1)

        # Generate attention weights
        attention = self.freq_attention(x)
        attention = F.interpolate(attention, size=(h_ll, w_ll), mode="nearest")

        # Apply attention to frequency components
        freq_modulated = freq_stack * attention

        # Split back into components
        LL, LH, HL, HH = torch.split(freq_modulated, C, dim=1)

        # Resize back to original dimensions if needed
        LH = F.interpolate(LH, size=(H, W), mode="bilinear", align_corners=False)
        HL = F.interpolate(HL, size=(H, W), mode="bilinear", align_corners=False)
        HH = F.interpolate(HH, size=(H, W), mode="bilinear", align_corners=False)
        LL = F.interpolate(LL, size=(H, W), mode="bilinear", align_corners=False)

        # Reconstruct with modulated frequencies
        output = self.dwt.inverse(LL, LH, HL, HH)

        # Final channel modulation
        output = self.channel_mod(torch.cat([output, x], dim=1))

        return output + x  # Residual connection


class C2f_RIM(nn.Module):
    """C2f module with RIM enhancements for YOLOv8."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        # Standard bottlenecks
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

        # RIM modules
        self.inr = LightweightINR(c2, c2 // 2, c2)
        self.mamba = RecursiveMambaBlock(c2, num_iterations=2)
        self.freq_gate = AdaptiveFrequencyGate(c2)

    def forward(self, x):
        # Standard C2f forward
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))

        # Apply RIM enhancements
        out = self.inr(out)
        out = self.mamba(out)
        out = self.freq_gate(out)

        return out


# Import necessary YOLOv8 modules
from ultralytics.nn.modules import Bottleneck, Conv
