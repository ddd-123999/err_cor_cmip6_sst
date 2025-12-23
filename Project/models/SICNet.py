import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """因果扩张卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # 保证因果性的左侧填充量
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # 手动处理padding以保证因果性
            dilation=dilation
        )

    def forward(self, x):
        # 左侧填充，右侧不填充
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    """TCN残差块（支持跳跃连接和通道调整）"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        # 第一层因果卷积
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 第二层因果卷积
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 跳跃连接的1x1卷积（当通道数变化时）
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x

        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # 调整残差连接的通道
        if self.skip is not None:
            residual = self.skip(residual)

        return F.relu(out + residual)  # 残差连接后激活


class TCN(nn.Module):
    """完整TCN模型"""

    def __init__(self, input_size, num_channels, kernel_size):
        """
        Args:
            input_size (int): 输入特征维度
            num_channels (list): 各层通道数，如[64, 64, 64]表示3个隐藏层
            kernel_size (int): 卷积核大小
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        # 构建多个残差块（自动计算扩张率）
        for i in range(num_levels):
            dilation = 2 ** i  # 指数级增长的扩张率
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                ResidualBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    dilation=dilation,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):  # -->(b, 1, c)
        out = self.network(x)  # -->(b, 8, c)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, base_c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.tcn = nn.Sequential(
            TCN(input_size=1, num_channels=[8, 8, 8], kernel_size=in_channels//base_c),
            TCN(input_size=8, num_channels=[8, 8, 8], kernel_size=in_channels//base_c),
            TCN(input_size=8, num_channels=[1, 1], kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = avg_out.squeeze(-1).squeeze(-1)  # 形状变为 (b, c)
        avg_out = avg_out.unsqueeze(1)  # 形状变为 (b, 1, c)
        avg_out = self.tcn(avg_out)
        avg_out = avg_out.unsqueeze(-1)  # 形状变为 (b, 1, c, 1)
        avg_out = avg_out.permute(0, 2, 1, 3)  # 形状变为 (b, c, 1, 1)

        max_out = self.max_pool(x)
        max_out = max_out.squeeze(-1).squeeze(-1)
        max_out = max_out.unsqueeze(1)
        max_out = self.tcn(max_out)
        max_out = max_out.unsqueeze(-1)
        max_out = max_out.permute(0, 2, 1, 3)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
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
    def __init__(self, in_channels, base_c, kernel_size=3):
        super().__init__()
        self.temporal_attention = TemporalAttention(in_channels, base_c)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.temporal_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNetTsamBlock(nn.Module):
    def __init__(self, in_channels, out_channels, base_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.tsam = TSAM(out_channels, base_c)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.tsam(self.bn2(self.conv2(out)))
        return self.relu2(out + identity)


class CnnTsamBlock(nn.Module):
    def __init__(self, in_channels, out_channels, base_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.tsam = TSAM(out_channels, base_c)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.tsam(self.conv(x))
        x = self.pool(skip)
        return x, skip  # 返回池化结果和跳跃连接


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, base_c):
        super().__init__()
        self.conv = nn.Sequential(
            ResNetTsamBlock(in_channels, out_channels, base_c),
            ResNetTsamBlock(out_channels, out_channels, base_c)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip  # 返回池化结果和跳跃连接


class UpSample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, base_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 拼接后总通道数: out_channels (up输出) + skip_channels
        self.conv = nn.Sequential(
            ResNetTsamBlock(out_channels + skip_channels, out_channels, base_c),
            ResNetTsamBlock(out_channels, out_channels, base_c)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, base_c):
        super().__init__()
        self.conv = nn.Sequential(
            ResNetTsamBlock(in_channels, out_channels, base_c),
            ResNetTsamBlock(out_channels, out_channels, base_c)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SICNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels = args.seq_len + args.seq_len * args.atm_len + args.seq_len * args.temporal_feature
        out_channels = args.pred_len
        base_c = in_channels
        # CnnTsamBlock
        self.cnn_tsam_block = CnnTsamBlock(base_c, base_c * 2, base_c)

        # 下采样路径
        self.down1 = DownSample(base_c * 2, base_c * 4, base_c)
        self.down2 = DownSample(base_c * 4, base_c * 6, base_c)
        self.down3 = DownSample(base_c * 6, base_c * 8, base_c)

        # 瓶颈层
        self.bottleneck = BottleNeck(base_c * 8, base_c * 10, base_c)

        # 上采样路径
        self.up4 = UpSample(base_c * 10, base_c * 8, base_c * 8, base_c)
        self.up3 = UpSample(base_c * 8, base_c * 6, base_c * 6, base_c)
        self.up2 = UpSample(base_c * 6, base_c * 4, base_c * 4, base_c)
        self.up1 = UpSample(base_c * 4, base_c * 2, base_c * 2, base_c)

        self.out_conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sm = nn.Sigmoid()

    def forward(self, x, x_atm, x_time):

        x = torch.cat([x, x_atm, x_time], dim=1)

        # 下采样
        x, skip0 = self.cnn_tsam_block(x)
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        # 瓶颈
        x = self.bottleneck(x)
        # 上采样
        x = self.up4(x, skip3)
        x = self.up3(x, skip2)
        x = self.up2(x, skip1)
        x = self.up1(x, skip0)

        x = self.out_conv(x)
        x = self.bn(x)
        x = self.sm(x)
        return x
