import torch
import torch.nn as nn
from torch.utils import checkpoint
from Project.models.nd_mamba2 import NdMamba2_2d


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch * 4)
        self.reduction = nn.Linear(in_ch * 4, out_ch, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nn.functional.pad(x, (0, W % 2, 0, H % 2))
            _, _, H, W = x.shape

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)

        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchExpanding(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.expand = nn.Linear(in_ch, in_ch * 4, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.expand(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = nn.functional.pixel_shuffle(x, 2)
        return x


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)  # 直接输出out_ch
        self.merge = PatchMerging(out_ch, out_ch)  # 保持out_ch

    def forward(self, x):
        x = self.res(x)
        skip = x  # 现在是out_ch通道
        x = self.merge(x)
        return x, skip


class UpSample(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.expand = PatchExpanding(in_ch)
        self.res = ResidualBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.expand(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


class MambaUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.add_anomaly = args.add_anomaly

        in_channels = args.seq_len
        if args.add_anomaly:
            in_channels = args.seq_len * 2

        base = args.base_c

        self.down1 = DownSample(in_channels, base)
        self.down2 = DownSample(base, base * 2)
        self.down3 = DownSample(base * 2, base * 4)
        self.down4 = DownSample(base * 4, base * 8)

        self.bottleneck_res = ResidualBlock(base * 8, base * 8)

        mamba_dim = getattr(args, "mamba_dim", 256)

        self.bottleneck_mamba = NdMamba2_2d(
            cin=base * 8,
            mamba_dim=mamba_dim,
            cout=base * 8,
            n_layer=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 64),
            headdim=getattr(args, "mamba_headdim", 64),
            chunk_size=getattr(args, "mamba_chunk_size", 32)
        )

        self.dropout = nn.Dropout2d(p=args.dropout) if args.dropout > 0 else nn.Identity()

        self.up4 = UpSample(base * 8, base * 8, base * 4)
        self.up3 = UpSample(base * 4, base * 4, base * 2)
        self.up2 = UpSample(base * 2, base * 2, base)
        self.up1 = UpSample(base, base, base)

        self.out_conv = nn.Conv2d(base, args.correction_len, kernel_size=1)

    def forward(self, x, x_anomaly):
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)

        x, s1 = checkpoint.checkpoint(self.down1, x, use_reentrant=False)
        x, s2 = checkpoint.checkpoint(self.down2, x, use_reentrant=False)
        x, s3 = checkpoint.checkpoint(self.down3, x, use_reentrant=False)
        x, s4 = checkpoint.checkpoint(self.down4, x, use_reentrant=False)

        x = self.bottleneck_res(x)
        x = self.dropout(x)
        x = self.bottleneck_mamba(x)
        x = self.dropout(x)

        x = checkpoint.checkpoint(self.up4, x, s4, use_reentrant=False)
        x = checkpoint.checkpoint(self.up3, x, s3, use_reentrant=False)
        x = checkpoint.checkpoint(self.up2, x, s2, use_reentrant=False)
        x = checkpoint.checkpoint(self.up1, x, s1, use_reentrant=False)

        return checkpoint.checkpoint(self.out_conv, x, use_reentrant=False)


if __name__ == "__main__":
    import argparse

    print("\n" + "=" * 60)
    print("🧪 测试 MambaUNet (Patch 版本)")
    print("=" * 60)

    args = argparse.Namespace(
        seq_len=3,
        correction_len=1,
        add_anomaly=False,
        base_c=32,
        dropout=0.0,
        mamba_dim=256,
        mamba_layers=2,
        mamba_d_state=64,
        mamba_headdim=64,
        mamba_chunk_size=32,
    )

    model = MambaUNet(args).cuda()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 模型参数量: {params:.2f}M")

    x = torch.randn(2, 3, 96, 1440).cuda()
    x_anomaly = torch.zeros(1).cuda()

    try:
        print(f"\n🚀 开始前向传播...")
        y = model(x, x_anomaly)
        print(f"✅ 测试通过!")
        print(f"   输入: {x.shape}")
        print(f"   输出: {y.shape}")
        print(f"\n📊 显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB")
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()