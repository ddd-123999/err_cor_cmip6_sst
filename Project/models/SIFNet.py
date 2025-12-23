import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=3):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图


# ResnetGen
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Depthwise Conv: groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            padding=1, groups=in_channels, bias=False
        )
        # Pointwise Conv: 1x1 卷积调整通道数
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResnetGen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 初始卷积调整通道数
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 使用改进后的深度可分离卷积
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

        self.cbam = CBAM(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv0(x)  # 调整通道并提取初始特征

        out = self.conv1(identity)
        out = self.conv2(out)
        out = self.cbam(out)

        out += identity  # 残差连接
        out = self.relu(out)
        return out


# TFCM模块
class TFCM(nn.Module):
    def __init__(self, base_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.pred_len = out_channels // 8

        self.conv_modules = nn.ModuleList()
        in_channels = 0
        for i in range(self.pred_len):
            # 当前时间步的输入通道数 = 原始C + 当前使用到的C'通道数
            in_channels = self.base_channels + (i + 1) * 8
            # 示例卷积模块（可自定义结构）
            module = ResnetGen(in_channels, 8)
            self.conv_modules.append(module)
        if in_channels - self.base_channels != out_channels:
            print(in_channels, out_channels)
            raise ValueError('TFCM模块验证失败!')

    def forward(self, x):
        outputs = []

        for i in range(self.pred_len):
            # 计算当前时间不需要切分的通道数
            slice_dim = self.base_channels + (i + 1) * 8
            # 通道切分
            x_slice = x[:, :slice_dim, :, :]
            # 通过对应卷积模块
            out = self.conv_modules[i](x_slice)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


# 输入模块
class InputBlock(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, x):
        result = self.input(x)
        skip = result
        return result, skip


# 下采样模块
class RestNetDownBlock(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        self.input = nn.Conv2d(base_channels, base_channels, kernel_size=1, padding=0, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(base_channels)

        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.cbam = CBAM(base_channels)

    def forward(self, x):
        identity = self.input(x)
        identity = self.bn0(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        out += identity  # 确保维度匹配
        out = self.relu2(out)
        skip = out
        return out, skip


# 上采样模块
class UpSample(nn.Module):
    def __init__(self, base_channels, in_channels, out_channels):
        super().__init__()
        # 转置卷积上采样
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False
        )
        self.tfcm = TFCM(base_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # 确保尺寸对齐（处理奇偶尺寸问题）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([skip, x], dim=1)  # 通道维度拼接
        x = self.tfcm(x)

        return x


# 输出模块

class OutputBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.max_in_channels = out_channels * 8  # 最大输入通道数

        # 动态生成权重：所有时间步 共享同一权重矩阵，但通过掩码控制有效区域
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.max_in_channels, 1, 1))
        # 初始化掩码：每个时间步 t 只能访问前 (t+1)*8 个通道
        self.register_buffer('mask', torch.zeros_like(self.weight))
        for t in range(self.out_channels):
            self.mask[t, : (t + 1) * 8] = 1

        # 参数初始化
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # 共享的 BatchNorm 和激活层（可选）
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x 形状: (B, max_in_channels, H, W)
        """
        # 应用动态权重
        masked_weight = self.weight * self.mask  # (out_channels, max_in_channels, 1, 1)

        # 一次性计算所有时间步的输出（利用分组卷积实现并行）
        outputs = F.conv2d(
            x,
            masked_weight,
            bias=None,
            groups=1  # 分组卷积，每组独立处理
        )  # 输出形状: (B, out_channels, H, W)

        # BatchNorm 和激活
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        return outputs


# ResUNet 模型
class SIFNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.seq_len + args.seq_len * args.atm_len + args.seq_len * args.temporal_feature
        self.base_channels = 256
        self.features_channels = 8 * args.pred_len
        self.out_channels = args.pred_len

        #  输入层
        self.input = InputBlock(self.in_channels, self.base_channels)
        # 下采样路径
        self.down1 = RestNetDownBlock(self.base_channels)
        self.down2 = RestNetDownBlock(self.base_channels)
        self.down3 = RestNetDownBlock(self.base_channels)
        self.down4 = RestNetDownBlock(self.base_channels)
        # 上采样路径
        self.up1 = UpSample(self.base_channels, self.base_channels, self.features_channels)
        self.up2 = UpSample(self.base_channels, self.features_channels, self.features_channels)
        self.up3 = UpSample(self.base_channels, self.features_channels, self.features_channels)
        self.up4 = UpSample(self.base_channels, self.features_channels, self.features_channels)
        # 输出层
        self.output = OutputBlock(self.out_channels)

    def forward(self, x, x_atm, x_time):
        x = torch.cat([x, x_atm, x_time], dim=1)
        x, skip1 = self.input(x)
        x, skip2 = self.down1(x)
        x, skip3 = self.down2(x)
        x, skip4 = self.down3(x)
        x, _ = self.down4(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        x = self.output(x)

        return x
