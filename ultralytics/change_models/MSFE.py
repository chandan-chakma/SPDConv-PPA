"""
MSFE-YOLO: Multi-Scale Feature Enhancement YOLO for Small Object Detection
Designed specifically for 6-8 pixel objects in UAV/satellite imagery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleUpsample(nn.Module):
    """Advanced upsampling that preserves fine details using pixel shuffle."""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor**2), 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        return x


class DetailPreservingDownsample(nn.Module):
    """Downsampling that preserves small object information."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Main path with strided convolution
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(),
        )

        # Detail path with max pooling to preserve edges
        self.detail_path = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.SiLU(),
        )

    def forward(self, x):
        main = self.main_path(x)
        detail = self.detail_path(x)
        return torch.cat([main, detail], dim=1)


class AdaptiveFeaturePyramid(nn.Module):
    """Scale-adaptive feature pyramid specifically for tiny objects."""

    def __init__(self, channels_list: list[int]):
        super().__init__()
        self.channels_list = channels_list

        # Create lateral connections with 1x1 convs
        self.lateral_convs = nn.ModuleList()
        for channels in channels_list:
            self.lateral_convs.append(nn.Conv2d(channels, 256, 1))

        # Top-down pathway with pixel shuffle upsampling
        self.td_convs = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.td_convs.append(PixelShuffleUpsample(256, 256, scale_factor=2))

        # Bottom-up pathway for context
        self.bu_convs = nn.ModuleList()
        for i in range(len(channels_list) - 1):
            self.bu_convs.append(DetailPreservingDownsample(256, 256))

        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(channels_list), 3) / 3)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Lateral connections
        laterals = []
        for i, (feat, conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(conv(feat))

        # Top-down pathway
        td_features = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            upsampled = self.td_convs[len(laterals) - 2 - i](td_features[-1])
            # Ensure size matching
            if upsampled.shape[2:] != laterals[i].shape[2:]:
                upsampled = F.interpolate(upsampled, size=laterals[i].shape[2:], mode="bilinear", align_corners=False)
            td_features.append(laterals[i] + upsampled)
        td_features = td_features[::-1]

        # Bottom-up pathway
        bu_features = [td_features[0]]
        for i in range(1, len(td_features)):
            downsampled = self.bu_convs[i - 1](bu_features[-1])
            # Ensure size matching
            if downsampled.shape[2:] != td_features[i].shape[2:]:
                downsampled = F.interpolate(
                    downsampled, size=td_features[i].shape[2:], mode="bilinear", align_corners=False
                )
            bu_features.append(td_features[i] + downsampled)

        # Adaptive fusion
        outputs = []
        weights = F.softmax(self.fusion_weights, dim=1)
        for i in range(len(features)):
            if i == 0:
                fused = weights[i, 0] * laterals[i] + weights[i, 1] * td_features[i] + weights[i, 2] * bu_features[i]
            else:
                fused = weights[i, 0] * laterals[i] + weights[i, 1] * td_features[i] + weights[i, 2] * bu_features[i]
            outputs.append(fused)

        return outputs


class DilatedContextModule(nn.Module):
    """Multi-scale context aggregation using dilated convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4)
        self.branch4 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=8, dilation=8)

        self.fusion = nn.Sequential(nn.BatchNorm2d(out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        combined = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(combined)


class SmallObjectHead(nn.Module):
    """Specialized detection head for small objects."""

    def __init__(self, in_channels, num_classes, anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.SiLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )

        # Separate branches for classification and regression
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, anchors * num_classes, 1),
        )

        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, anchors * 4, 1),
        )

        self.obj_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, anchors, 1),
        )

    def forward(self, x):
        x = self.refine(x)
        cls = self.cls_branch(x)
        reg = self.reg_branch(x)
        obj = self.obj_branch(x)

        return torch.cat([reg, obj, cls], dim=1)


class MSFE_Module(nn.Module):
    """Main Multi-Scale Feature Enhancement Module."""

    def __init__(self, in_channels_list: list[int], num_classes: int = 80):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes

        # Adaptive Feature Pyramid
        self.afp = AdaptiveFeaturePyramid(in_channels_list)

        # Dilated context modules for each scale
        self.context_modules = nn.ModuleList()
        for _ in in_channels_list:
            self.context_modules.append(DilatedContextModule(256, 256))

        # Small object heads for each scale
        self.heads = nn.ModuleList()
        for _ in in_channels_list:
            self.heads.append(SmallObjectHead(256, num_classes))

        # Feature enhancement for smallest scale
        self.small_scale_enhance = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        # Apply adaptive feature pyramid
        pyramid_features = self.afp(features)

        # Apply context aggregation
        context_features = []
        for i, (feat, context_mod) in enumerate(zip(pyramid_features, self.context_modules)):
            context_feat = context_mod(feat)
            # Extra enhancement for P3 (smallest objects)
            if i == 0:
                context_feat = self.small_scale_enhance(context_feat)
            context_features.append(context_feat)

        # Apply detection heads
        outputs = []
        for feat, head in zip(context_features, self.heads):
            outputs.append(head(feat))

        return outputs


class MSFE_YOLO(nn.Module):
    """Complete MSFE-YOLO model for YOLOv8 integration."""

    def __init__(self, base_channels: list[int] = [128, 256, 512], num_classes: int = 80):
        super().__init__()
        self.msfe = MSFE_Module(base_channels, num_classes)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        return self.msfe(x)


# Configuration function for YOLOv8
def get_msfe_config():
    """Returns configuration for integrating MSFE with YOLOv8."""
    return {
        "module": MSFE_YOLO,
        "num_classes": 80,  # Update based on dataset
        "base_channels": [128, 256, 512],  # YOLOv8n channels
        "anchor_sizes": [
            [[10, 13], [16, 30], [33, 23]],  # P3
            [[30, 61], [62, 45], [59, 119]],  # P4
            [[116, 90], [156, 198], [373, 326]],  # P5
        ],
    }


if __name__ == "__main__":
    # Test the module
    model = MSFE_YOLO(base_channels=[128, 256, 512], num_classes=10)

    # Simulate YOLOv8n feature maps
    p3 = torch.randn(1, 128, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 512, 20, 20)

    outputs = model([p3, p4, p5])

    print("MSFE-YOLO Output shapes:")
    for i, out in enumerate(outputs):
        print(f"Scale {i + 1}: {out.shape}")
