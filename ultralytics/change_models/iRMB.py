import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models._efficientnet_blocks import SqueezeExcite as SE
from timm.models.layers import DropPath

# 参考文献: https://arxiv.org/pdf/2301.01146


class LayerNorm2d(nn.Module):
    # 自定义二维层归一化类
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        # 初始化LayerNorm
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        # 将输入张量从 (B, C, H, W) 重新排列为 (B, H, W, C)
        x = rearrange(x, "b c h w -> b h w c").contiguous()
        # 进行归一化
        x = self.norm(x)
        # 再次排列回 (B, C, H, W)
        x = rearrange(x, "b h w c -> b c h w").contiguous()
        return x


def get_norm(norm_layer="in_1d"):
    # 根据输入获取归一化层
    eps = 1e-6
    norm_dict = {
        "none": nn.Identity,
        "in_1d": partial(nn.InstanceNorm1d, eps=eps),
        "in_2d": partial(nn.InstanceNorm2d, eps=eps),
        "in_3d": partial(nn.InstanceNorm3d, eps=eps),
        "bn_1d": partial(nn.BatchNorm1d, eps=eps),
        "bn_2d": partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        "bn_3d": partial(nn.BatchNorm3d, eps=eps),
        "gn": partial(nn.GroupNorm, eps=eps),
        "ln_1d": partial(nn.LayerNorm, eps=eps),
        "ln_2d": partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


class ConvNormAct(nn.Module):
    # 卷积 + 归一化 + 激活类
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        skip=False,
        norm_layer="bn_2d",
        act_layer="relu",
        inplace=True,
        drop_path_rate=0.0,
    ):
        super().__init__()
        # 是否使用跳跃连接
        self.has_skip = skip and dim_in == dim_out
        # 计算填充大小
        padding = math.ceil((kernel_size - stride) / 2)
        # 定义卷积层
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        # 获取归一化层
        self.norm = get_norm(norm_layer)(dim_out)
        # 获取激活层
        self.act = get_act(act_layer)(inplace=inplace)
        # 定义DropPath层
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        # 保存跳跃连接的输入
        shortcut = x
        # 执行卷积
        x = self.conv(x)
        # 执行归一化
        x = self.norm(x)
        # 执行激活函数
        x = self.act(x)
        # 如果有跳跃连接，则将其添加到输出
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


def get_act(act_layer="relu"):
    # 根据输入获取激活函数
    act_dict = {
        "none": nn.Identity,
        "sigmoid": nn.Sigmoid,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "prelu": nn.PReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    return act_dict[act_layer]


class iRMB(nn.Module):
    # 自定义iRMB类
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=True,
        has_skip=True,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        act_layer="relu",
        v_proj=True,
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        dim_head=64,
        window_size=7,
        attn_s=True,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        v_group=False,
        attn_pre=False,
        inplace=True,
    ):
        super().__init__()
        # 根据输入决定是否使用归一化
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        # 计算中间维度
        dim_mid = int(dim_in * exp_ratio)
        # 是否使用跳跃连接
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            # 确保输入维度可被头数整除
            assert dim_in % dim_head == 0, "dim should be divisible by num_heads"
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head**-0.5
            self.attn_pre = attn_pre
            # 定义查询和键的卷积层
            self.qk = ConvNormAct(
                dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer="none", act_layer="none"
            )
            # 定义值的卷积层
            self.v = ConvNormAct(
                dim_in,
                dim_mid,
                kernel_size=1,
                groups=self.num_head if v_group else 1,
                bias=qkv_bias,
                norm_layer="none",
                act_layer=act_layer,
                inplace=inplace,
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                # 如果需要，定义值的卷积层
                self.v = ConvNormAct(
                    dim_in,
                    dim_mid,
                    kernel_size=1,
                    bias=qkv_bias,
                    norm_layer="none",
                    act_layer=act_layer,
                    inplace=inplace,
                )
            else:
                # 否则，使用身份映射
                self.v = nn.Identity()  # 返回输入的原样输出，即不对输入进行任何操作或修改。
        # 定义局部卷积层
        self.conv_local = ConvNormAct(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            dilation=dilation,
            groups=dim_mid,
            norm_layer="bn_2d",
            act_layer="silu",
            inplace=inplace,
        )
        # 定义Squeeze-Excite层
        self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        # 定义投影层
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer="none", act_layer="none", inplace=inplace)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        # 保存跳跃连接的输入
        shortcut = x
        # 执行归一化
        x = self.norm(x)
        _B, _C, H, W = x.shape
        if self.attn_s:
            # 进行填充以满足窗口大小
            # 判断窗口大小，如果小于等于0，则使用输入张量的宽度和高度作为窗口大小
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            # 初始化左和上填充量为0
            pad_l, pad_t = 0, 0
            # 计算右侧和底部的填充量
            pad_r = (window_size_W - W % window_size_W) % window_size_W  # 右侧填充量
            pad_b = (window_size_H - H % window_size_H) % window_size_H  # 底部填充量
            # 填充输入张量，按照顺序填充左、右、上、下
            x = F.pad(
                x,
                (
                    pad_l,
                    pad_r,
                    pad_t,
                    pad_b,
                    0,
                    0,
                ),
            )
            # 计算填充后，每个窗口将包含的块数
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            # 重新排列输入张量，将其形状调整为适合后续操作的形式
            x = rearrange(x, "b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1", n1=n1, n2=n2).contiguous()
            # 注意力计算
            # 获取输入张量的形状，b 为批量大小，c 为通道数，h 为高度，w 为宽度
            _b, _c, h, w = x.shape
            # 使用 qk 线性层处理输入张量 x，得到 qk
            qk = self.qk(x)
            # 重新排列 qk 张量的维度
            # 将 qk 的形状调整为 (qk, b, heads, h*w, dim_head)，其中 qk 为 2，表示查询和键
            qk = rearrange(
                qk,
                "b (qk heads dim_head) h w -> qk b heads (h w) dim_head",
                qk=2,
                heads=self.num_head,
                dim_head=self.dim_head,
            ).contiguous()
            # 将查询和键分开
            q, k = qk[0], qk[1]
            # 计算空间注意力
            # 使用矩阵乘法计算 q 和 k 的注意力得分，并乘以缩放因子
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            # 对注意力得分进行 softmax 归一化
            attn_spa = attn_spa.softmax(dim=-1)
            # 应用 dropout，防止过拟合
            attn_spa = self.attn_drop(attn_spa)
            if (
                self.attn_pre
            ):  # 前处理注意力是在计算注意力之前进行特征优化，而后处理注意力则是在计算注意力之后对特征进行加权处理
                # 前处理注意力
                x = rearrange(x, "b (heads dim_head) h w -> b heads (h w) dim_head", heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(
                    x_spa, "b heads (h w) dim_head -> b (heads dim_head) h w", heads=self.num_head, h=h, w=w
                ).contiguous()
                x_spa = self.v(x_spa)
            else:
                # 后处理注意力
                v = self.v(x)
                v = rearrange(v, "b (heads dim_head) h w -> b heads (h w) dim_head", heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(
                    x_spa, "b heads (h w) dim_head -> b (heads dim_head) h w", heads=self.num_head, h=h, w=w
                ).contiguous()
            # 去填充
            x = rearrange(x_spa, "(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)", n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                # 截断到原始尺寸
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        # 将局部卷积输出与输入相加
        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        # 如果有跳跃连接，则将其与输入相加
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x


class Bottleneck_iRMB(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = iRMB(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class C2f_iRMB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck_iRMB(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == "__main__":
    SAFM = iRMB(256, 128)
    # 创建一个输入张量
    batch_size = 1
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    # 运行模型并打印输入和输出的形状
    output_tensor = SAFM(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("0utput shape:", output_tensor.shape)


# # Ultralytics YOLO 🚀, AGPL-3.0 license
# # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
#
# # Parameters
# nc: 80 # number of classes
# scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
#   s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
#   m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
#   l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
#   x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
#
# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f_iRMB, [1024, True]]
#   - [-1, 1, SPPF, [1024, 5]] # 9
#
# # YOLOv8.0n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 3, C2f, [512]] # 12
#
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 3, C2f, [256]] # 15 (P3/8-small)
#
#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 12], 1, Concat, [1]] # cat head P4
#   - [-1, 3, C2f, [512]] # 18 (P4/16-medium)
#
#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 9], 1, Concat, [1]] # cat head P5
#   - [-1, 3, C2f, [1024]] # 21 (P5/32-large)
#
#   - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
