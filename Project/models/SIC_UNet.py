import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


# ==================== TCN 模块 ====================
class CausalConv1d(nn.Module):
    """因果扩张卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock_TCN(nn.Module):
    """TCN残差块"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.skip is not None:
            residual = self.skip(residual)
        return F.relu(out + residual)


class TCN(nn.Module):
    """完整TCN模型"""

    def __init__(self, input_size, num_channels, kernel_size):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(ResidualBlock_TCN(in_ch, out_ch, kernel_size, dilation=dilation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==================== TSAM 注意力模块 ====================
class TemporalAttention(nn.Module):
    """时间注意力模块"""

    def __init__(self, in_channels, base_c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 动态计算TCN的kernel_size
        kernel_size = max(3, in_channels // base_c)

        self.tcn = nn.Sequential(
            TCN(input_size=1, num_channels=[8, 8, 8], kernel_size=kernel_size),
            TCN(input_size=8, num_channels=[8, 8, 8], kernel_size=kernel_size),
            TCN(input_size=8, num_channels=[1, 1], kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.avg_pool(x)  # (B, C, 1, 1)
        avg_out = avg_out.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B, 1, C)
        avg_out = self.tcn(avg_out)  # (B, 1, C)
        avg_out = avg_out.permute(0, 2, 1).unsqueeze(-1)  # (B, C, 1, 1)

        # 最大池化分支
        max_out = self.max_pool(x)
        max_out = max_out.squeeze(-1).squeeze(-1).unsqueeze(1)
        max_out = self.tcn(max_out)
        max_out = max_out.permute(0, 2, 1).unsqueeze(-1)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class TSAM(nn.Module):
    """时空注意力模块 (Temporal-Spatial Attention Module)"""

    def __init__(self, in_channels, base_c, kernel_size=3):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, base_c)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.temporal_attention(x)
        x = self.spatial_attention(x)
        return x


# ==================== UNet 基础模块（集成TSAM） ====================
class ResidualBlock(nn.Module):
    """残差块（集成TSAM注意力）"""

    def __init__(self, in_channels, out_channels, base_c, mid_channels=None,
                 dropout=0.0, use_attention=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # ✅ 集成TSAM注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.tsam = TSAM(out_channels, base_c, kernel_size=3)

        # 处理通道数变化
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # ✅ 在残差连接前应用TSAM
        if self.use_attention:
            out = self.tsam(out)

        out += identity
        out = self.relu(out)
        return out


class DownSample(nn.Module):
    """下采样模块"""

    def __init__(self, in_channels, out_channels, base_c, dropout=0.0, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_channels, out_channels, base_c,
                          dropout=dropout, use_attention=use_attention),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpSample(nn.Module):
    """上采样模块"""

    def __init__(self, in_channels, out_channels, base_c, mid_channels=None,
                 dropout=0.0, use_attention=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResidualBlock(in_channels, out_channels, base_c, mid_channels,
                                  dropout=dropout, use_attention=use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# ==================== 主模型 ====================
class UNet_TSAM(nn.Module):
    """UNet + TSAM 时空注意力机制"""

    def __init__(self, args):
        super().__init__()
        in_channels = args.seq_len
        self.add_anomaly = args.add_anomaly
        if args.add_anomaly:
            in_channels = args.seq_len * 2
        out_channels = args.correction_len
        base_channels = args.base_c

        dropout = getattr(args, 'dropout', 0.0)
        # ✅ 新增：是否使用TSAM注意力的开关
        self.use_attention = getattr(args, 'use_tsam', True)

        # 下采样路径
        self.down1 = DownSample(in_channels, base_channels, base_channels,
                                dropout=0.0, use_attention=self.use_attention)
        self.down2 = DownSample(base_channels, base_channels * 2, base_channels,
                                dropout=0.0, use_attention=self.use_attention)
        self.down3 = DownSample(base_channels * 2, base_channels * 4, base_channels,
                                dropout=dropout * 0.5, use_attention=self.use_attention)
        self.down4 = DownSample(base_channels * 4, base_channels * 8, base_channels,
                                dropout=dropout, use_attention=self.use_attention)

        # 瓶颈层
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 8, base_channels,
                                        dropout=dropout, use_attention=self.use_attention)

        # 上采样路径
        self.up4 = UpSample(base_channels * 16, base_channels * 4, base_channels,
                            base_channels * 8, dropout=dropout, use_attention=self.use_attention)
        self.up3 = UpSample(base_channels * 8, base_channels * 2, base_channels,
                            base_channels * 4, dropout=dropout * 0.5, use_attention=self.use_attention)
        self.up2 = UpSample(base_channels * 4, base_channels, base_channels,
                            base_channels * 2, dropout=0.0, use_attention=self.use_attention)
        self.up1 = UpSample(base_channels * 2, base_channels, base_channels,
                            dropout=0.0, use_attention=self.use_attention)

        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, x_anomaly):
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)

        # 下采样
        x, skip1 = checkpoint.checkpoint(self.down1, x, use_reentrant=False)
        x, skip2 = checkpoint.checkpoint(self.down2, x, use_reentrant=False)
        x, skip3 = checkpoint.checkpoint(self.down3, x, use_reentrant=False)
        x, skip4 = checkpoint.checkpoint(self.down4, x, use_reentrant=False)

        # 瓶颈层
        x = checkpoint.checkpoint(self.bottleneck, x, use_reentrant=False)

        # 上采样
        x = checkpoint.checkpoint(self.up4, x, skip4, use_reentrant=False)
        x = checkpoint.checkpoint(self.up3, x, skip3, use_reentrant=False)
        x = checkpoint.checkpoint(self.up2, x, skip2, use_reentrant=False)
        x = checkpoint.checkpoint(self.up1, x, skip1, use_reentrant=False)

        # 输出
        x = checkpoint.checkpoint(self.out_conv, x, use_reentrant=False)
        return x


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import argparse

    # 创建测试参数
    args = argparse.Namespace(
        seq_len=3,
        correction_len=1,
        add_anomaly=False,
        base_c=32,
        dropout=0.2,
        use_tsam=True  # 是否使用TSAM注意力
    )

    # 创建模型
    model = UNet_TSAM(args).cuda()

    # 统计参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 模型参数量: {params:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 96, 1440).cuda()
    x_anomaly = torch.zeros(1).cuda()

    try:
        y = model(x, x_anomaly)
        print(f"✅ 测试通过!")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {y.shape}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()