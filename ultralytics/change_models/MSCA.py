"""
MSCA-YOLO: Multi-Scale Context Aggregation for Small Object Detection
Advanced modules for detecting extremely small objects (6-8 pixels) in remote sensing imagery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class PixelContextFeatureAggregator(nn.Module):
    """
    PCFA: Pixel-Context Feature Aggregator
    Preserves pixel-level information while capturing surrounding context
    Critical for 6-8 pixel objects where every pixel matters
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        
        # Pixel-level pathway - no spatial reduction
        self.pixel_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Local context pathway - 3x3 with dilation=1
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Medium context pathway - 3x3 with dilation=2
        self.medium_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Wide context pathway - 3x3 with dilation=3
        self.wide_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Context fusion with learned attention
        self.context_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 4, channels // reduction, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 4, 1),
            nn.Sigmoid()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        # Four parallel pathways
        pixel = self.pixel_conv(x)
        local = self.local_conv(x)
        medium = self.medium_conv(x)
        wide = self.wide_conv(x)
        
        # Concatenate all pathways
        concat = torch.cat([pixel, local, medium, wide], dim=1)
        
        # Apply attention weights
        attention = self.context_attention(concat)
        weighted = concat * attention
        
        # Fuse and add residual
        out = self.fusion(weighted)
        return out + x


class AdaptiveDetailAwareFusion(nn.Module):
    """
    ADAF: Adaptive Detail-Aware Fusion
    Dynamically adjusts feature fusion based on object scale
    Gives more weight to high-resolution features for small objects
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # Scale-aware gating mechanism
        self.scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Detail enhancement for high-res features
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        
        # Context enhancement for low-res features
        self.context_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, high_res, low_res):
        """
        Args:
            high_res: High resolution feature map (more detail)
            low_res: Low resolution feature map (more context)
        """
        # Upsample low_res to match high_res
        if high_res.shape[2:] != low_res.shape[2:]:
            low_res_up = F.interpolate(low_res, size=high_res.shape[2:], 
                                       mode='bilinear', align_corners=False)
        else:
            low_res_up = low_res
        
        # Enhance each branch
        enhanced_high = self.detail_enhance(high_res)
        enhanced_low = self.context_enhance(low_res_up)
        
        # Compute adaptive weights
        concat = torch.cat([enhanced_high, enhanced_low], dim=1)
        weights = self.scale_gate(concat)
        w_high = weights[:, 0:1, :, :]
        w_low = weights[:, 1:2, :, :]
        
        # Weighted fusion
        weighted_high = enhanced_high * w_high
        weighted_low = enhanced_low * w_low
        
        # Final fusion
        out = self.fusion(torch.cat([weighted_high, weighted_low], dim=1))
        return out


class HierarchicalScaleFocusedPyramid(nn.Module):
    """
    HSFPN: Hierarchical Scale-Focused Pyramid Network
    Enhanced FPN that focuses on small object scales
    Creates dense feature pyramid with emphasis on high-resolution levels
    """
    def __init__(self, channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.num_levels = len(channels_list)
        
        # Lateral connections for each level
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            ) for ch in channels_list
        ])
        
        # Top-down pathway with ADAF fusion
        self.adaf_modules = nn.ModuleList([
            AdaptiveDetailAwareFusion(out_channels) 
            for _ in range(self.num_levels - 1)
        ])
        
        # Bottom-up pathway for refinement
        self.bottom_up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(self.num_levels - 1)
        ])
        
        # Small object enhancement layers
        self.small_object_enhance = nn.ModuleList([
            PixelContextFeatureAggregator(out_channels) 
            for _ in range(2)  # Apply to top 2 levels (highest resolution)
        ])
        
        # Output refinement
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            ) for _ in range(self.num_levels)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from backbone [P3, P4, P5, ...]
        Returns:
            List of enhanced feature maps
        """
        # Apply lateral connections
        lateral_features = [
            lateral_conv(feat) 
            for lateral_conv, feat in zip(self.lateral_convs, features)
        ]
        
        # Top-down pathway with ADAF fusion
        top_down_features = [lateral_features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            high_res = lateral_features[i]
            low_res = top_down_features[0]
            fused = self.adaf_modules[i](high_res, low_res)
            top_down_features.insert(0, fused)
        
        # Enhance small object detection in top levels
        for i in range(min(2, len(top_down_features))):
            top_down_features[i] = self.small_object_enhance[i](top_down_features[i])
        
        # Bottom-up pathway for refinement
        bottom_up_features = [top_down_features[0]]
        for i in range(self.num_levels - 1):
            down = self.bottom_up_convs[i](bottom_up_features[-1])
            # Element-wise addition with top-down feature
            if down.shape[2:] != top_down_features[i + 1].shape[2:]:
                down = F.interpolate(down, size=top_down_features[i + 1].shape[2:],
                                    mode='bilinear', align_corners=False)
            fused = down + top_down_features[i + 1]
            bottom_up_features.append(fused)
        
        # Output refinement
        outputs = [
            output_conv(feat)
            for output_conv, feat in zip(self.output_convs, bottom_up_features)
        ]
        
        return outputs


class MSCABlock(nn.Module):
    """
    Complete MSCA Block combining PCFA and ADAF
    Can be inserted into YOLOv8 backbone or neck
    """
    def __init__(self, channels, use_adaf=False):
        super().__init__()
        self.use_adaf = use_adaf
        
        self.pcfa = PixelContextFeatureAggregator(channels)
        
        if use_adaf:
            self.adaf = AdaptiveDetailAwareFusion(channels)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        # Apply PCFA
        x = self.pcfa(x)
        
        # If skip connection provided, use ADAF
        if self.use_adaf and skip is not None:
            x = self.adaf(x, skip)
        
        # Output refinement
        x = self.output_conv(x)
        return x


class SmallObjectDetectionHead(nn.Module):
    """
    Specialized detection head for small objects
    Reduces stride and increases resolution for tiny object detection
    """
    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super().__init__()
        
        # Feature refinement with PCFA
        self.refine = PixelContextFeatureAggregator(in_channels)
        
        # Detection branches
        hidden_channels = max(in_channels, 64)
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_anchors * num_classes, 1)
        )
        
        # Regression branch with higher precision
        self.reg_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, num_anchors * 4, 1)
        )
        
        # Objectness branch
        self.obj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, num_anchors, 1)
        )
    
    def forward(self, x):
        # Refine features
        x = self.refine(x)
        
        # Get predictions
        cls_pred = self.cls_conv(x)
        reg_pred = self.reg_conv(x)
        obj_pred = self.obj_conv(x)
        
        return cls_pred, reg_pred, obj_pred


def test_modules():
    """Test all modules"""
    print("Testing MSCA-YOLO Modules...")
    
    # Test PCFA
    print("\n1. Testing PCFA...")
    pcfa = PixelContextFeatureAggregator(channels=256)
    x = torch.randn(2, 256, 64, 64)
    out = pcfa(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "PCFA output shape mismatch"
    print("   ✓ PCFA working correctly")
    
    # Test ADAF
    print("\n2. Testing ADAF...")
    adaf = AdaptiveDetailAwareFusion(channels=256)
    high_res = torch.randn(2, 256, 64, 64)
    low_res = torch.randn(2, 256, 32, 32)
    out = adaf(high_res, low_res)
    print(f"   High-res: {high_res.shape}, Low-res: {low_res.shape}, Output: {out.shape}")
    assert out.shape == high_res.shape, "ADAF output shape mismatch"
    print("   ✓ ADAF working correctly")
    
    # Test HSFPN
    print("\n3. Testing HSFPN...")
    hsfpn = HierarchicalScaleFocusedPyramid(channels_list=[128, 256, 512], out_channels=256)
    features = [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 512, 20, 20)
    ]
    outputs = hsfpn(features)
    print(f"   Input features: {[f.shape for f in features]}")
    print(f"   Output features: {[o.shape for o in outputs]}")
    assert len(outputs) == len(features), "HSFPN output count mismatch"
    print("   ✓ HSFPN working correctly")
    
    # Test MSCABlock
    print("\n4. Testing MSCABlock...")
    msca = MSCABlock(channels=256, use_adaf=True)
    x = torch.randn(2, 256, 64, 64)
    skip = torch.randn(2, 256, 32, 32)
    out = msca(x, skip)
    print(f"   Input: {x.shape}, Skip: {skip.shape}, Output: {out.shape}")
    assert out.shape == x.shape, "MSCABlock output shape mismatch"
    print("   ✓ MSCABlock working correctly")
    
    # Test SmallObjectDetectionHead
    print("\n5. Testing SmallObjectDetectionHead...")
    head = SmallObjectDetectionHead(in_channels=256, num_classes=10, num_anchors=3)
    x = torch.randn(2, 256, 40, 40)
    cls_pred, reg_pred, obj_pred = head(x)
    print(f"   Input: {x.shape}")
    print(f"   Cls: {cls_pred.shape}, Reg: {reg_pred.shape}, Obj: {obj_pred.shape}")
    print("   ✓ SmallObjectDetectionHead working correctly")
    
    print("\n" + "="*50)
    print("✓ All modules tested successfully!")
    print("="*50)


if __name__ == "__main__":
    test_modules()