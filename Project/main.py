import argparse
import os
import random

import numpy as np
import torch
from exp.exp import Exp

file = '../Preprocessing/anomaly_data/ACCESS-CM2/ssp245_pre_anomaly.npz'
# 通用
parser = argparse.ArgumentParser(description='correction CMIP')
parser.add_argument('--model', type=str, default='MambaUNet_new', help='SwinUNet, SwinUNet_new, '
                                                                     'MambaUNet, MambaUNet_new'
                                                                     'UNet, UNet_new'
                                                                     'ConvLSTM, ConvLSTM_new, UNet_LSTM')
parser.add_argument('--seq_len', type=int, default=3, help='input sequence length')
parser.add_argument('--correction_len', type=int, default=1, help='correction sequence length')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--loss_name', type=str, default='mse', help='mse')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')    #原来：15
parser.add_argument('--delta', type=float, default=0.00001, help='early stopping delta threshold')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial optimizer learning rate')
parser.add_argument('--step_size', type=int, default=15, help='learning rate step size')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability (0.0-0.5, default: 0.2)')
parser.add_argument('--add_anomaly', type=bool, default=False, help='should anomaly be added')
parser.add_argument('--use_normalized', type=bool, default=True, help='Whether to use normalised data')
parser.add_argument('--use_standardized', type=bool, default=False, help='Whether to use Z-SCORE standardized data')

parser.add_argument('--root_path_target', type=str, default='../Preprocessing/observation/obs_normalized/', help='root path of the target data file')
parser.add_argument('--data_path_target', type=str, default='sst_daily_ACCESS-CM2.npz', help='target data file')
parser.add_argument('--root_path_norm_params', type=str, default='../Preprocessing/dataset/ACCESS-CM2_normalized/', help='root path of normalization parameters file')
parser.add_argument('--data_path_norm_params', type=str, default='normalization_params.npz', help='normalization parameters file name')
parser.add_argument('--root_path_std_params', type=str, default='../Preprocessing/dataset/ACCESS-CM2_standardized/', help='root path of Z-SCORE standardization parameters file')
parser.add_argument('--data_path_std_params', type=str, default='standardization_params.npz', help='Z-SCORE standardization parameters file name')
parser.add_argument('--root_path_mask', type=str, default='../Preprocessing/observation/obs/', help='root path of the mask data file')
parser.add_argument('--root_path_correction', type=str, default='../Preprocessing/dataset/ACCESS-CM2/', help='root path correction data file')
parser.add_argument('--data_path_train', type=str, default='ssp245_train.npz', help='training correction data file')
parser.add_argument('--data_path_val', type=str, default='ssp245_val.npz', help='validation correction data file')
parser.add_argument('--data_path_test', type=str, default='ssp245_test.npz', help='testing correction data file')
parser.add_argument('--checkpoints', type=str, default='../Experiment/MambaUNet', help='location of model checkpoints')

parser.add_argument('--root_path_pre', type=str, default='../Preprocessing/dataset/ACCESS-CM2/', help='root path of the pred data file')
parser.add_argument('--data_path_pre', type=str, default='ssp245_pre.npz', help='pred data file')
parser.add_argument('--root_path_anomaly', type=str, default='../Preprocessing/anomaly_data/ACCESS-CM2/', help='root path of the anomaly data file')
parser.add_argument('--data_path_anomaly_train', type=str, default='ssp245_train_anomaly.npz', help='anomaly data file for training')
parser.add_argument('--data_path_anomaly_val', type=str, default='ssp245_val_anomaly.npz', help='anomaly data file for validation')
parser.add_argument('--data_path_anomaly_test', type=str, default='ssp245_test_anomaly.npz', help='anomaly data file for testing')
parser.add_argument('--root_path_anomaly_in_pre', type=str, default='../Preprocessing/anomaly_data/ACCESS-CM2/', help='root path anomaly in pred')
parser.add_argument('--data_path_anomaly_in_pre', type=str, default='ssp245_pre_anomaly.npz', help='anomaly in pred data file')

# UNet
parser.add_argument('--base_c', type=int, default=32, help='base channel')

# ✅ 新增：Mamba相关参数（放在Swin-transformer参数后面）
parser.add_argument('--use_mamba', type=bool, default=True, help='whether to use Mamba in UNet')
parser.add_argument('--mamba_dim', type=int, default=256, help='Mamba internal dimension')
parser.add_argument('--mamba_layers', type=int, default=3, help='Number of Mamba layers')
parser.add_argument('--mamba_d_state', type=int, default=128, help='Mamba state dimension')
parser.add_argument('--mamba_headdim', type=int, default=128, help='Mamba head dimension')
parser.add_argument('--mamba_chunk_size', type=int, default=32, help='Mamba chunk size')

# # ConvLstm
parser.add_argument('--out_channels', type=int, nargs='+', default=[16, 16, 16, 16], help='ConvLstm out channels')
parser.add_argument('--kernel_size', type=int, nargs='+', default=[(3, 3), (3, 3), (3, 3), (3, 3)], help='ConvLstm kernel size')
parser.add_argument('--num_layers', type=int, default=4, help='ConvLstm num layers')

# # Swin-transformer
# parser.add_argument('--depths', type=int, nargs='+', default=(2, 2, 6, 2, 2, 2, 6, 2, 2), help='Swin transformer depths')
# parser.add_argument('--num_heads', type=int, nargs='+', default=(2, 4, 8, 16, 32, 16, 8, 4, 2), help='Swin transformer num heads')
# parser.add_argument('--embed_dim', type=int, default=48, help='Swin transformer embed dim')

args = parser.parse_args()

# 配置cuda 加载mask、area
args.device = torch.device('cuda')

folder_path_mask = os.path.join(args.root_path_mask, 'mask.npy')
args.mask_data = np.load(folder_path_mask)

def setup_seed(seed=42):
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
setup_seed(42)

# 提取模型名称，移除_normalized后缀
cmip_path = args.root_path_correction.rstrip('/')  # 移除末尾的斜杠
cmip_name = os.path.basename(cmip_path)
# 1. 移除 _normalized 后缀（如果存在）
if cmip_name.endswith('_normalized'):
    cmip_name = cmip_name.replace('_normalized', '')
    print(f"📝 提取模型名称 (移除_normalized): {cmip_name}")
# 2. ✅ 新增：移除 _standardized 后缀（如果存在）
if cmip_name.endswith('_standardized'):
    cmip_name = cmip_name.replace('_standardized', '')
    print(f"📝 提取模型名称 (移除_standardized): {cmip_name}")

# 打印dropout设置
if args.dropout > 0:
    print(f"✅ 使用Dropout正则化: {args.dropout}")
else:
    print("ℹ️ 不使用Dropout")

# 检查互斥性
if args.use_normalized and args.use_standardized:
    raise ValueError("不能同时启用 Min-Max (--use_normalized) 和 Z-score (--use_standardized)!")

# 动态设置模型数据路径
if args.use_normalized:
    print("✅ 使用 Min-Max 归一化数据")
    args.root_path_correction = "../Preprocessing/dataset/ACCESS-CM2_normalized/"
    args.root_path_pre = "../Preprocessing/dataset/ACCESS-CM2_normalized/"
elif args.use_standardized:
    print("✅ 使用 Z-score 标准化数据")
    # 假设 Z-score 数据放在 _standardized 文件夹中
    args.root_path_correction = "../Preprocessing/dataset/ACCESS-CM2_standardized/"
    args.root_path_pre = "../Preprocessing/dataset/ACCESS-CM2_standardized/"
else:
    print("ℹ️ 使用原始数据")
    args.root_path_correction = "../Preprocessing/dataset/ACCESS-CM2/"
    args.root_path_pre = "../Preprocessing/dataset/ACCESS-CM2/"


# 加载归一化/标准化参数
if args.use_normalized:
    # --- Min-Max 归一化 ---
    norm_params_path = os.path.join(args.root_path_norm_params, args.data_path_norm_params)
    if os.path.exists(norm_params_path):
        norm_params = np.load(norm_params_path)
        args.data_min = norm_params['min']
        args.data_max = norm_params['max']
        args.data_mean = None  # 确保Z-score参数为空
        args.data_std = None
        print(f"✅ 加载 Min-Max 参数 - Min: {args.data_min:.4f}, Max: {args.data_max:.4f}")
    else:
        print(f"⚠️ 未找到 Min-Max 参数文件: {norm_params_path}")
        args.data_min = None; args.data_max = None; args.data_mean = None; args.data_std = None
elif args.use_standardized:
    # --- Z-score 标准化 ---
    std_params_path = os.path.join(args.root_path_std_params, args.data_path_std_params)
    if os.path.exists(std_params_path):
        std_params = np.load(std_params_path)
        # ⚠️ 加载 Z-score 参数 (mean 和 std)
        args.data_mean = std_params['mean']
        args.data_std = std_params['std']
        args.data_min = None  # 确保Min-Max参数为空
        args.data_max = None
        print(f"✅ 加载 Z-score 参数 - Mean (Min/Max): {np.nanmin(args.data_mean):.4f}/{np.nanmax(args.data_mean):.4f}")
        print(f"                         Std (Min/Max): {np.nanmin(args.data_std):.4f}/{np.nanmax(args.data_std):.4f}")
    else:
        print(f"⚠️ 未找到 Z-score 参数文件: {std_params_path}")
        args.data_min = None; args.data_max = None; args.data_mean = None; args.data_std = None
else:
    # --- 原始数据 ---
    args.data_min = None; args.data_max = None; args.data_mean = None; args.data_std = None

# 加载归一化/标准化观测数据
if args.use_normalized:
    # --- Min-Max 归一化模式 ---
    # Min-Max 归一化后的观测数据通常存放在 obs_normalized 文件夹中
    args.root_path_target = '../Preprocessing/observation/obs_normalized/'
    args.data_path_target = f'sst_daily_{cmip_name}.npz'
    print(f"✅ 使用 Min-Max 归一化后的观测数据: {args.data_path_target}")
elif args.use_standardized:
    # --- Z-score 标准化模式 ---
    # ⚠️ 修正：Z-score 标准化后的观测数据应从 obs_standardized 文件夹加载
    args.root_path_target = '../Preprocessing/observation/obs_standardized/'
    args.data_path_target = f'sst_daily_{cmip_name}.npz'
    print(f"✅ 使用 Z-score 标准化后的观测数据: {args.data_path_target}")
else:
    # --- 原始数据模式 ---
    args.root_path_target = '../Preprocessing/observation/obs/'
    args.data_path_target = 'sst_daily_not_to_be_normalized.npz'
    print(f"ℹ️ 使用原始观测数据: {args.data_path_target}")

# 保存训练信息
setting = 'md-{}_cn-{}_bs-{}_pt-{}_sl-{}_cl-{}_dp-{}_ln-{}_norm-{}'.format(
    args.model,
    cmip_name,
    args.batch_size,
    args.patience,
    args.seq_len,
    args.correction_len,
    args.dropout,
    args.loss_name,
    args.use_normalized # 新增
)

# ============================================================
# 在 main.py 的最后，注释掉训练，添加：
# 如果已经训练过，直接加载测试
# ============================================================
# if os.path.exists(os.path.join(args.checkpoints, setting, 'checkpoint.pth')):
#     print("\n" + "=" * 60)
#     print("🔍 检测到已保存的模型，直接加载测试")
#     print("=" * 60)
#
#     exp = Exp(args)
#     checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
#     exp.model.load_state_dict(
#         torch.load(checkpoint_path, map_location=args.device, weights_only=True)
#     )
#
#     _, test_loader = exp._get_data(flag='test')
#     exp.test(setting, test_loader)
# else:
#     # 原有的训练代码
#     exp = Exp(args)
#     print('>' * 20 + 'start training : {}'.format(setting) + '<' * 20)
#     train_time, test_loader = exp.train(setting)
#     exp.test(setting, test_loader)
# ============================================================
# 保存当前实验的参数设置到txt文件
# ============================================================
save_dir = os.path.join(args.checkpoints, setting)
os.makedirs(save_dir, exist_ok=True)

param_file = os.path.join(save_dir, 'args_config.txt')
with open(param_file, 'w', encoding='utf-8') as f:
    f.write("=== 实验参数配置 ===\n")

    # ✅ 按类别分组保存参数，排除anomaly相关
    categories = {
        '模型参数': ['model', 'batch_size', 'seq_len', 'correction_len', 'loss_name'],
        '训练参数': ['train_epochs', 'patience', 'delta', 'learning_rate', 'warmup_epochs', 'step_size'],
        '正则化参数': ['dropout', 'use_normalized', 'use_standardized', 'add_anomaly'],
        '模型架构': ['base_c', 'use_mamba', 'mamba_dim', 'mamba_layers',
                     'mamba_d_state', 'mamba_headdim', 'mamba_chunk_size',
                     'out_channels', 'kernel_size', 'num_layers'],
        '数据路径': ['root_path_target', 'data_path_target', 'root_path_correction',
                     'data_path_train', 'data_path_val', 'data_path_test',
                     'root_path_pre', 'data_path_pre', 'root_path_mask'],
        '环境配置': ['device']
    }

    for category, keys in categories.items():
        f.write(f"\n--- {category} ---\n")
        for k in keys:
            if hasattr(args, k):
                v = getattr(args, k)
                f.write(f"{k}: {v}\n")

    f.write("\n=== 运行环境 ===\n")
    f.write(f"PyTorch CUDA 版本: {torch.version.cuda}\n")
    f.write(f"设备: {args.device}\n")
    f.write(f"随机种子: 42\n")

print(f"📁 参数配置已保存到: {param_file}")

# 开始训练测试
exp = Exp(args)

print('>'*20 + 'start training : {}'.format(setting) + '<'*20)
train_time, test_loader = exp.train(setting)

print('>'*20 + 'testing' + '<'*20)
exp.test(setting, test_loader)

# print('>'*20 + 'predict' + '<'*20)
# exp.predict(setting)

torch.cuda.empty_cache()
