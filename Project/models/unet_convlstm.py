from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from Project.models.conv_lstm import ConvLSTM


class convlstm(nn.Module):
    def __init__(self, correction_len, out_channels, kernel_size, num_layers):
        super().__init__()
        in_channels = 1
        correction_len = correction_len
        out_channels = out_channels
        kernel_size = kernel_size
        num_layers = num_layers

        self.conv_lstm = ConvLSTM(in_channels, out_channels,
                                  kernel_size, num_layers)
        self.conv_out = nn.Conv2d(out_channels[-1], correction_len, kernel_size=1)  # 如果要进行多步输出，此处换成Conv3d比较合适
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        _, layer_output = self.conv_lstm(x)
        h = layer_output[-1][0]
        preds = self.sigmoid(self.conv_out(h))
        return preds


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_convlstm=False):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # 基础双卷积块
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 如果启用 ConvLSTM
        self.use_convlstm = use_convlstm
        if self.use_convlstm:
            self.convlstm = convlstm(
                correction_len=out_channels,
                out_channels=[8],
                kernel_size=[(3, 3)],
                num_layers=1,
            )

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_convlstm:
            x = self.convlstm(x)
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_convlstm=False):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels, use_convlstm=use_convlstm)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class UNet_LSTM(nn.Module):
    def __init__(self, args):
        super(UNet_LSTM, self).__init__()
        in_channels = args.seq_len
        self.add_anomaly = args.add_anomaly
        if args.add_anomaly:
            in_channels = args.seq_len * 2
        out_channels = args.correction_len
        base_c = args.base_c

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2, use_convlstm=True)
        self.down2 = Down(base_c * 2, base_c * 4, use_convlstm=True)
        self.down3 = Down(base_c * 4, base_c * 8, use_convlstm=True)

        self.up1 = Up(base_c * 8, base_c * 4)
        self.up2 = Up(base_c * 4, base_c * 2)
        self.up3 = Up(base_c * 2, base_c)

        self.out_conv = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor, x_anomaly) -> Dict[str, torch.Tensor]:
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # 上采样路径
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.out_conv(x)

        return x


if __name__ == '__main__':
    pass


