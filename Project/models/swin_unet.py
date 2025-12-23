import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from Project.models.swin_transformer import SwinTransformer
from Project.models.Unet import ResidualBlock


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, in_resblock=None):
        super().__init__()
        if in_resblock is None:
            in_resblock = in_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_resblock, out_channels, mid_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # 确保尺寸对齐（处理奇偶尺寸问题）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)  # 通道维度拼接
        x = self.conv(x)
        return x


class SwinUnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channels = args.seq_len
        self.add_anomaly = args.add_anomaly
        if args.add_anomaly:
            in_channels = args.seq_len * 2

        depths = args.depths
        num_heads = args.num_heads
        embed_dim = args.embed_dim

        self.encoder = SwinTransformer(in_chans=in_channels, depths=depths, num_heads=num_heads, embed_dim=embed_dim)
        self.bottleneck = ResidualBlock(16 * embed_dim, 16 * embed_dim)
        self.up1 = UpSample(16 * embed_dim, 8 * embed_dim)
        self.up2 = UpSample(8 * embed_dim, 4 * embed_dim)
        self.up3 = UpSample(4 * embed_dim, 2 * embed_dim)
        self.up4 = UpSample(2 * embed_dim, embed_dim)
        self.up5 = UpSample(embed_dim, embed_dim, in_resblock=embed_dim // 2 + in_channels)
        self.output = nn.Sequential(
            nn.Conv2d(embed_dim, args.correction_len, kernel_size=1),
        )

    def forward(self, x, x_anomaly):
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)
        identity = x
        x, x_skip = self.encoder(x)
        x = self.bottleneck(x)
        x = checkpoint.checkpoint(self.up1, x, x_skip[3], use_reentrant=False)
        x = checkpoint.checkpoint(self.up2, x, x_skip[2], use_reentrant=False)
        x = checkpoint.checkpoint(self.up3, x, x_skip[1], use_reentrant=False)
        x = checkpoint.checkpoint(self.up4, x, x_skip[0], use_reentrant=False)
        x = checkpoint.checkpoint(self.up5, x, identity, use_reentrant=False)
        x = self.output(x)

        return x


if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.add_anomaly = False
    # args.seq_len = 4
    # args.embed_dim = 48
    # args.num_heads = (3, 6, 12, 24)
    # args.depths = (2, 2, 6, 2)
    # args.correction_len = 1
    #
    # su = SwinUnet(args)
    # x = torch.randn(1, 4, 448, 304)
    # x_anomaly = torch.randn(1, 4, 448, 304)
    # y = su(x, x_anomaly=torch.zeros_like(x))
    # print(y.shape)
