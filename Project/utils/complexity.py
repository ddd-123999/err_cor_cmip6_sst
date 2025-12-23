import argparse
import torch

from thop import profile
from Project.models.swin_unet import SwinUnet
from Project.models.conv_lstm import ConvLSTMSIC
from Project.models.Unet import UNet
from Project.models.unet_nores import UNet_nores

# 参数设置
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.add_anomaly = False
args.seq_len = 3
args.embed_dim = 48
args.num_heads = (3, 6, 12, 24)
args.depths = (2, 2, 6, 2)
args.correction_len = 1
args.base_c = 32

# 模型实例化
# model = SwinUnet(args).to('cuda')
model = UNet(args).to('cuda')
model.eval()

# 加载模拟数据
x = torch.randn(4, 3, 448, 304).to('cuda')

# 计算复杂度
flops, params = profile(model, inputs=(x, torch.zeros_like(x)))

print("FLOPs: %.2fG" % (flops / 1e9))
print("Params: %.2fM" % (params / 1e6))
