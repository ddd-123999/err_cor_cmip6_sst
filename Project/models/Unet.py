import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


# 残差块（处理通道变化）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0): # ✅ 新增dropout参数
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ✅ 新增Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        # 处理通道数变化
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 先调整通道

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)  # ✅ 在第一个卷积后添加dropout
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 确保维度匹配
        out = self.relu(out)
        return out


# 下采样模块
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):  # ✅ 新增dropout参数
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlock(in_channels, out_channels, dropout=dropout),  # ✅ 传递dropout
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        skip = x  # 保存跳跃连接
        x = self.pool(x)
        return x, skip


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0): # ✅ 新增dropout参数
        super().__init__()
        # 双线性插值上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 残差块处理拼接后的通道
        self.conv = ResidualBlock(in_channels, out_channels, mid_channels, dropout=dropout)  # ✅ 传递dropout

    def forward(self, x, skip):
        x = self.up(x)
        # 确保尺寸对齐（处理奇偶尺寸问题）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)  # 通道维度拼接
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels = args.seq_len
        self.add_anomaly = args.add_anomaly
        if args.add_anomaly:
            in_channels = args.seq_len * 2
        out_channels = args.correction_len
        base_channels = args.base_c

        # ✅ 获取dropout参数
        dropout = getattr(args, 'dropout', 0.0)  # 如果args没有dropout属性，默认为0

        # 下采样路径
        self.down1 = DownSample(in_channels, base_channels, dropout=0.0)  # 32 ✅ 浅层不dropout
        self.down2 = DownSample(base_channels, base_channels * 2, dropout=0.0)  # 64
        self.down3 = DownSample(base_channels * 2, base_channels * 4, dropout=dropout * 0.5)  # 128 ✅ 中层dropout减半
        self.down4 = DownSample(base_channels * 4, base_channels * 8, dropout=dropout)  # 256 ✅ 深层全dropout

        # 瓶颈层
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 8, dropout=dropout)  # 256 ✅ 瓶颈层dropout

        # 上采样路径
        self.up4 = UpSample(base_channels * 16, base_channels * 4, base_channels * 8, dropout=dropout)  # 512 -> 128 ✅ 深层全dropout
        self.up3 = UpSample(base_channels * 8, base_channels * 2, base_channels * 4,dropout=dropout * 0.5)  # 256 -> 64 ✅ 中层dropout减半
        self.up2 = UpSample(base_channels * 4, base_channels, base_channels * 2, dropout=0.0)  # 128 -> 32 ✅ 浅层不dropout
        self.up1 = UpSample(base_channels * 2, base_channels, dropout=0.0)  # 64 -> 32

        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, x_anomaly):
        # 下采样
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)
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


if __name__ == "__main__":
    pass
