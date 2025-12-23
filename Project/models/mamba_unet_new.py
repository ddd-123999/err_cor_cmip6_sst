import torch
import torch.nn as nn
from torch.utils import checkpoint
from Project.models.nd_mamba2 import NdMamba2_2d


# =============================
# Residual Block（全残差）
# =============================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# =============================
# DownSample
# =============================
class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.res(x)
        skip = x
        x = self.pool(x)
        return x, skip


# =============================
# UpSample（最终修复版本）
# =============================
class UpSample(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # 转置卷积：直接从 in_ch 降到 out_ch 并上采样
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # 拼接后处理：out_ch + skip_ch → out_ch
        self.res = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        return x


# =============================
# 主结构 MambaUNet_new
# =============================
class MambaUNet_new(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.add_anomaly = args.add_anomaly

        in_channels = args.seq_len
        if args.add_anomaly:
            in_channels = args.seq_len * 2

        base = args.base_c

        # 下采样路径
        self.down1 = DownSample(in_channels, base)  # → base
        self.down2 = DownSample(base, base * 2)  # → base*2
        self.down3 = DownSample(base * 2, base * 4)  # → base*4
        self.down4 = DownSample(base * 4, base * 8)  # → base*8

        # bottleneck (全残差 + Mamba2)
        self.bottleneck_res = ResidualBlock(base * 8, base * 8)

        # 获取 mamba_dim，这是 Mamba 实际的输出通道数
        mamba_dim = getattr(args, "mamba_dim", 256)

        self.bottleneck_mamba = NdMamba2_2d(
            cin=base * 8,
            cout=base * 8,  # 期望输出 base*8，但实际可能输出 mamba_dim
            mamba_dim=mamba_dim,
            n_layer=getattr(args, "mamba_layers", 2),
            d_state=getattr(args, "mamba_d_state", 64),
            headdim=getattr(args, "mamba_headdim", 64),
            chunk_size=getattr(args, "mamba_chunk_size", 32)
        )

        # 通道对齐层：Mamba 可能输出 mamba_dim 通道，需要恢复到 base*8
        # 这一步很关键！确保后续上采样路径的通道数匹配
        self.channel_align = nn.Conv2d(mamba_dim, base * 8, kernel_size=1)

        # dropout
        self.dropout = nn.Dropout2d(p=args.dropout) if args.dropout > 0 else nn.Identity()

        # 上采样路径
        # 每一层：当前输入 → 上采样+通道降维 → 和 skip 拼接 → 残差处理
        self.up4 = UpSample(base * 8, base * 8, base * 4)  # base*8 → base*4
        self.up3 = UpSample(base * 4, base * 4, base * 2)  # base*4 → base*2
        self.up2 = UpSample(base * 2, base * 2, base)  # base*2 → base
        self.up1 = UpSample(base, base, base)  # base → base

        # 输出层
        self.out_conv = nn.Conv2d(base, args.correction_len, kernel_size=1)

    def forward(self, x, x_anomaly):
        # 处理 anomaly
        if self.add_anomaly:
            x = torch.cat([x, x_anomaly], dim=1)

        # 下采样路径
        x, s1 = checkpoint.checkpoint(self.down1, x, use_reentrant=False)
        x, s2 = checkpoint.checkpoint(self.down2, x, use_reentrant=False)
        x, s3 = checkpoint.checkpoint(self.down3, x, use_reentrant=False)
        x, s4 = checkpoint.checkpoint(self.down4, x, use_reentrant=False)

        # bottleneck（不使用 checkpoint 避免梯度问题）
        x = self.bottleneck_res(x)
        x = self.dropout(x)

        # 🔍 调试：打印 Mamba 前后的形状
        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"  Before Mamba: {x.shape}")

        x = self.bottleneck_mamba(x)

        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"  After Mamba: {x.shape}")

        x = self.channel_align(x)  # 确保通道数正确

        if hasattr(self, '_debug_mode') and self._debug_mode:
            print(f"  After channel_align: {x.shape}")

        x = self.dropout(x)

        # 上采样路径
        x = checkpoint.checkpoint(self.up4, x, s4, use_reentrant=False)
        x = checkpoint.checkpoint(self.up3, x, s3, use_reentrant=False)
        x = checkpoint.checkpoint(self.up2, x, s2, use_reentrant=False)
        x = checkpoint.checkpoint(self.up1, x, s1, use_reentrant=False)

        # 输出
        return checkpoint.checkpoint(self.out_conv, x, use_reentrant=False)


# =============================
# 测试代码
# =============================
if __name__ == "__main__":
    pass
    # import argparse
    #
    # # 测试不同的 base_c 配置
    # for base_c in [32, 64]:
    #     print(f"\n{'=' * 60}")
    #     print(f"测试 base_c={base_c}")
    #     print('=' * 60)
    #
    #     args = argparse.Namespace(
    #         seq_len=3,
    #         correction_len=1,
    #         add_anomaly=False,
    #         base_c=base_c,
    #         dropout=0.0,
    #         mamba_dim=256,
    #         mamba_layers=2,
    #         mamba_d_state=64,
    #         mamba_headdim=64,
    #         mamba_chunk_size=32,
    #     )
    #
    #     model = MambaUNet_new(args)
    #     model._debug_mode = True  # 启用调试模式
    #     params = sum(p.numel() for p in model.parameters()) / 1e6
    #     print(f"模型参数量: {params:.2f}M")
    #
    #     # 测试前向传播
    #     x = torch.randn(2, 3, 96, 1440)
    #     x_anomaly = torch.zeros(1)
    #
    #     try:
    #         print(f"开始前向传播...")
    #         y = model(x, x_anomaly)
    #         print(f"✅ 测试通过！")
    #         print(f"   输入形状: {x.shape}")
    #         print(f"   输出形状: {y.shape}")
    #     except Exception as e:
    #         print(f"❌ 测试失败: {e}")
    #         import traceback
    #
    #         traceback.print_exc()