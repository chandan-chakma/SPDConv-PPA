"""
MSCA-YOLO Custom Modules - CORRECTED VERSION
Place this file in: ultralytics/nn/modules/msca.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCABlock(nn.Module):
    """
    PCFA: Pixel-Context Feature Aggregator
    Preserves pixel-level information with multi-scale context
    """
    def __init__(self, c1, c2):
        """
        Args:
            c1: Input channels
            c2: Output channels (usually same as c1)
        """
        super().__init__()
        c = c2  # Use output channels
        reduction = 4
        
        # Pixel pathway
        self.pixel_conv = nn.Sequential(
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        
        # Local context (dilation=1)
        self.local_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        
        # Medium context (dilation=2)
        self.medium_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c),
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        
        # Wide context (dilation=3)
        self.wide_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=3, dilation=3, groups=c),
            nn.Conv2d(c, c, 1),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 4, c // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(c // reduction, c * 4, 1),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c * 4, c, 1),
            nn.BatchNorm2d(c)
        )
        
        # Channel projection if needed
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
    def forward(self, x):
        # Project input if needed
        identity = self.proj(x)
        
        # Four pathways
        p1 = self.pixel_conv(identity)
        p2 = self.local_conv(identity)
        p3 = self.medium_conv(identity)
        p4 = self.wide_conv(identity)
        
        # Concatenate
        concat = torch.cat([p1, p2, p3, p4], dim=1)
        
        # Attention
        att = self.attention(concat)
        weighted = concat * att
        
        # Fuse and residual
        out = self.fusion(weighted)
        return out + identity


class ADAFFusion(nn.Module):
    """
    ADAF: Adaptive Detail-Aware Fusion
    Dynamically fuses features based on scale
    """
    def __init__(self, c1, c2):
        """
        Args:
            c1: Input channels (auto-detected from input)
            c2: Output channels
        """
        super().__init__()
        self.c2 = c2
        
        # Channel projection layers (will be created dynamically if needed)
        self.high_proj = None
        self.low_proj = None
        
        # Detail enhancement (for high-res)
        self.detail = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1),
            nn.BatchNorm2d(c2)
        )
        
        # Context enhancement (for low-res)
        self.context = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2),
            nn.Conv2d(c2, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # Scale-aware gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2 * 2, c2, 1),
            nn.SiLU(),
            nn.Conv2d(c2, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: Can be:
               - List of 2 tensors [high_res, low_res]
               - Single tensor (pass through)
        """
        # Handle input
        if isinstance(x, list) and len(x) >= 2:
            high, low = x[0], x[1]
        elif isinstance(x, list) and len(x) == 1:
            return x[0]
        else:
            return x
        
        # Create projection layers if needed (first forward pass)
        if high.shape[1] != self.c2:
            if self.high_proj is None:
                self.high_proj = nn.Conv2d(high.shape[1], self.c2, 1).to(high.device)
            high = self.high_proj(high)
        
        if low.shape[1] != self.c2:
            if self.low_proj is None:
                self.low_proj = nn.Conv2d(low.shape[1], self.c2, 1).to(low.device)
            low = self.low_proj(low)
        
        # Upsample low_res to match high_res spatial dimensions
        if high.shape[2:] != low.shape[2:]:
            low = F.interpolate(low, size=high.shape[2:], mode='bilinear', align_corners=False)
        
        # Enhance
        high_enh = self.detail(high)
        low_enh = self.context(low)
        
        # Gate
        concat = torch.cat([high_enh, low_enh], dim=1)
        weights = self.gate(concat)
        w_h = weights[:, 0:1, :, :]
        w_l = weights[:, 1:2, :, :]
        
        # Fuse
        weighted = torch.cat([high_enh * w_h, low_enh * w_l], dim=1)
        out = self.fusion(weighted)
        
        return out


class HSFPN(nn.Module):
    """
    HSFPN: Hierarchical Scale-Focused Pyramid Network
    Enhanced multi-scale feature pyramid with small object focus
    """
    def __init__(self, c1, c2):
        """
        Args:
            c1: Input channels (can be list or int)
            c2: Output channels
        """
        super().__init__()
        
        # Handle input channels
        if isinstance(c1, list):
            in_channels_list = c1
        else:
            in_channels_list = [c1]
        
        out_channels = c2
        
        # Lateral connections
        self.laterals = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) 
            for in_ch in in_channels_list
        ])
        
        # Top-down fusion with ADAF
        self.fusions = nn.ModuleList([
            ADAFFusion(out_channels, out_channels) 
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Small object enhancement (PCFA on top 2 levels)
        self.enhance = nn.ModuleList([
            MSCABlock(out_channels, out_channels) 
            for _ in range(min(2, len(in_channels_list)))
        ])
        
        # Bottom-up pathway
        self.downsample = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Output convs
        self.outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU()
            )
            for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features):
        """
        Args:
            features: List of features [P2, P3, P4, P5]
        Returns:
            List of enhanced multi-scale features
        """
        # Lateral connections
        laterals = [lat(feat) for lat, feat in zip(self.laterals, features)]
        
        # Top-down pathway with ADAF
        top_down = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            # Fuse high-res with upsampled low-res
            fused = self.fusions[i]([laterals[i], top_down[0]])
            top_down.insert(0, fused)
        
        # Enhance top levels for small objects
        for i in range(min(len(self.enhance), len(top_down))):
            top_down[i] = self.enhance[i](top_down[i])
        
        # Bottom-up pathway
        bottom_up = [top_down[0]]
        for i in range(len(top_down) - 1):
            down = self.downsample[i](bottom_up[-1])
            # Add with top-down feature
            fused = down + top_down[i + 1]
            bottom_up.append(fused)
        
        # Output refinement
        outputs = [out(feat) for out, feat in zip(self.outputs, bottom_up)]
        
        return outputs