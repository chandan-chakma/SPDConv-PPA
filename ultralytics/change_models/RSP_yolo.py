"""RSP-YOLO FINAL FIXED"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, DFL

class SPBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = max(6, int(c2 * e))
        c_ = c_ - (c_ % 3)
        branch_c = c_ // 3
        self.branch_1x1 = Conv(c1, branch_c, 1, 1)
        self.branch_3x3 = Conv(c1, branch_c, 3, 1)
        self.branch_5x5 = Conv(c1, branch_c, 5, 1)
        self.fusion = Conv(c_, c2, 1, 1)
        self.ca = ChannelAttention(c2)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        b1, b2, b3 = self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x)
        out = self.fusion(torch.cat([b1, b2, b3], dim=1))
        return x + self.ca(out) if self.add else self.ca(out)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_c = max(1, channels // min(reduction, channels))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_c, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_c, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * (1 + self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))))

class C2f_SP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = max(16, int(c2 * e))  # CRITICAL: minimum 16 channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(SPBottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(self.conv(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))

__all__ = ['SPBottleneck', 'C2f_SP', 'ChannelAttention', 'SpatialAttention']