import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path


# ==================== 配置部分 ====================
class Config:
    # --- 1. 模型列表 ---
    MODELS = [
        'ACCESS-CM2',
        'ACCESS-ESM1-5',
        'BCC-CSM2-MR',
        'CanESM5',
        'CESM2-WACCM',
        'CMCC-CM2-SR5',
        'CMCC-ESM2',
        'EC-Earth3-CC',
        'EC-Earth3-Veg-LR',
        'EC-Earth3-veg',
        'EC-Earth3',
        'GFDL-CM4',
        'GFDL-ESM4',
        'IPSL-CM6A-LR',
        'MIROC6',
        'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'NESM3',
        'NorESM2-LM',
        'NorESM2-MM'
    ]

    # --- 2. 方法列表 ---
    METHODS = [
        ('base', 'Control', '../../Baseline/Base/base_metrics_all.npy'),
        ('EDCDF', 'EDCDF', '../../Baseline/QM/qm_edcdf_q100_metrics_all.npy'),
        # ('linear_reg', 'Linear Reg', '../../Baseline/Linear_regression/lr_results_data_s3_p1/lr_metrics_s3_p1_all.npy'),
        ('ConvLSTM', 'ConvLSTM', '../../Baseline/ConvLSTM/first/convlstm_metrics_s3_p1_all.npy'),
        ('UNet', 'UNet', '../../Baseline/UNet/first/unet_metrics_s3_p1_all.npy'),
        ('MambaUNet', 'Mamba-TempNet', '../../Baseline/MambaUNet/first/mambaunet_metrics_s3_p1_all.npy'),
    ]

    # --- 3. 指标配置（两个子图）---
    METRICS_CONFIG = [
        {
            'metric': 'rmse',
            'title': '(a)',
            'cmap': 'Spectral_r',
            'vmin': 0.8,
            'vmax': 1.6
        },
        {
            'metric': 'mae',
            'title': '(b)',
            'cmap': 'Spectral_r',
            'vmin': 0.4,
            'vmax': 0.9
        }
    ]

    # --- 输出配置 ---
    OUTPUT_DIR = './heatmap_figures'
    OUTPUT_FILENAME = 'heatmap_rmse_mae_combined.png'
    DPI = 600


# ==================== 数据加载函数 ====================
def load_metric_data(target_metric):
    """加载指定指标的数据并返回DataFrame"""
    matrix_data = []
    method_names = []

    for method_key, method_label, file_path in Config.METHODS:
        try:
            if not Path(file_path).exists():
                matrix_data.append([np.nan] * len(Config.MODELS))
                method_names.append(method_label)
                continue

            data = np.load(file_path, allow_pickle=True).item()
            row_values = []

            for model in Config.MODELS:
                model_metrics = data.get(model, {})
                val = np.nan

                # 查找指标值
                possible_keys = [target_metric, target_metric.upper(), f'mean_{target_metric}']
                for k in possible_keys:
                    if k in model_metrics:
                        val = model_metrics[k]
                        break

                # 特殊处理MSE
                if np.isnan(val) and target_metric == 'mse' and 'rmse' in model_metrics:
                    val = model_metrics['rmse'] ** 2

                # 格式转换
                if isinstance(val, (np.ndarray, list)):
                    val = float(val) if val else np.nan

                row_values.append(val)

            matrix_data.append(row_values)
            method_names.append(method_label)

        except Exception as e:
            print(f"   ⚠️ 加载 {method_label} 数据时出错: {e}")
            matrix_data.append([np.nan] * len(Config.MODELS))
            method_names.append(method_label)

    return pd.DataFrame(matrix_data, index=method_names, columns=Config.MODELS)


# ==================== 绘图主程序 ====================
def plot_combined_heatmap():
    """绘制RMSE和MAE的组合热力图"""
    print("🚀 开始绘制组合热力图...")

    # 创建图形：2行1列的子图布局
    fig, axes = plt.subplots(2, 1, figsize=(20, 10),
                             dpi=Config.DPI,
                             sharex=True)

    # 循环绘制每个子图
    for idx, metric_config in enumerate(Config.METRICS_CONFIG):
        ax = axes[idx]

        metric = metric_config['metric']
        title = metric_config['title']
        cmap = metric_config['cmap']
        vmin = metric_config['vmin']
        vmax = metric_config['vmax']

        print(f"   📊 绘制子图 {idx + 1}: {metric.upper()}")
        print(f"      🎨 颜色: {cmap}")
        print(f"      📏 范围: [{vmin}, {vmax}]")

        # 加载数据
        df = load_metric_data(metric)

        # 绘制热力图
        sns.heatmap(df,
                    annot=True,
                    fmt=".3f",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'shrink': 0.80, 'pad': 0.02},
                    annot_kws={"fontsize": 14, "color": 'black'},
                    linewidths=0,
                    linecolor='none',
                    square=True,
                    ax=ax)

        # 添加黑色边框
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

        # 设置标题和标签
        ax.set_title(title, fontsize=22, pad=15, fontweight='normal', loc='left')

        # 设置刻度
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=35,
                           ha='right',
                           fontsize=18,
                           fontweight='normal')
        ax.set_yticklabels(ax.get_yticklabels(),
                           rotation=0,
                           fontsize=18,
                           fontweight='normal')

        # 获取色标对象并设置字体大小和标签
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        if metric == 'rmse':
            cbar.set_label('RMSE (°C)', fontsize=16, fontweight='normal', rotation=90, labelpad=12)
        elif metric == 'mae':
            cbar.set_label('MAE (°C)', fontsize=16, fontweight='normal', rotation=90, labelpad=12)

        # 只在底部子图显示X轴标签
        if idx < len(Config.METRICS_CONFIG) - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('', fontsize=16)

    # 调整布局
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.06)
    # 保存图片
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = Path(Config.OUTPUT_DIR) / Config.OUTPUT_FILENAME
    plt.savefig(save_path, bbox_inches='tight')

    print(f"\n✅ 图片已保存: {save_path}")
    # plt.show()


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_combined_heatmap()