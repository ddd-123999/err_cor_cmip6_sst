import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ==================== 1. 配置中心 (修改这里即可) ====================
class Config:
    # --- [关键] 纵坐标指标选择 ---
    # 可选值: 'rmse', 'mae', 'mse', 'pcc', 'nse'
    TARGET_METRIC = 'pcc'

    # --- [关键] 横坐标模型选择 (想画几个就填几个) ---
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

    # --- 方法列表 (不同颜色的柱子) ---
    METHODS = [
        # # (配置键名, 图例显示名, 颜色)  蓝绿黄橙红
        # ('base', 'CMIP6 Raw', '#7fb2d3'),
        # ('EDCDF', 'EDCDF', '#8dd3c9'),
        # ('linear_reg', 'Linear Reg', '#ffb55f'),
        # ('UNet', 'UNet', '#fc7f71'),
        # ('MambaUNet', 'MambaUNet', 'red')

        # (配置键名, 图例显示名, 颜色)  反过来
        ('base', 'Control', '#3d67a0'),
        ('EDCDF', 'EDCDF', '#97c3dd'),
        # ('linear_reg', 'Linear Reg', '#efe9c2'),
        ('ConvLSTM', 'ConvLSTM', '#efe9c2'),
        ('UNet', 'UNet', '#fc945d'),
        ('MambaUNet', 'MambaUNet', '#c32b23')

        # # (配置键名, 图例显示名, 颜色)  再反过来
        # ('base', 'CMIP6 Raw', '#354e97'),
        # ('EDCDF', 'EDCDF', '#70a3c4'),
        # ('linear_reg', 'Linear Reg', '#c7e5ec'),
        # ('UNet', 'UNet', '#f5b46f'),
        # ('MambaUNet', 'MambaUNet', '#df5b3f')

    ]

    # --- 数据文件路径 ---
    METRICS_FILES = {
        'base': '../../Baseline/Base/base_metrics_all.npy',
        'EDCDF': '../../Baseline/QM/qm_edcdf_q100_metrics_all.npy',
        # 'linear_reg': '../../Baseline/Linear_regression/lr_results_data_s3_p1/lr_metrics_s3_p1_all.npy',
        'ConvLSTM': '../../Baseline/ConvLSTM/first/convlstm_metrics_s3_p1_all.npy',
        'UNet': '../../Baseline/UNet/first/unet_metrics_s3_p1_all.npy',
        'MambaUNet': '../../Baseline/MambaUNet/first/mambaunet_metrics_s3_p1_all.npy',
    }

    # --- 图表样式设置 ---
    OUTPUT_DIR = './metric_figures'
    DPI = 600
    # 自动根据模型数量计算宽度，保证不拥挤
    # FIG_WIDTH = max(10, len(MODELS) * 1.2)
    FIG_HEIGHT = 7
    FIG_WIDTH =18


# ==================== 2. 数据加载器 ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.metrics_data = {}

    def load_data(self):
        print(f"📂 正在读取 {self.config.TARGET_METRIC.upper()} 数据...")
        for method_key, _, _ in self.config.METHODS:
            path = self.config.METRICS_FILES.get(method_key)
            if path and Path(path).exists():
                try:
                    self.metrics_data[method_key] = np.load(path, allow_pickle=True).item()
                except:
                    self.metrics_data[method_key] = {}
            else:
                print(f"   ⚠️ 文件缺失: {path}")
                self.metrics_data[method_key] = {}

    def get_plot_data(self):
        """获取绘图矩阵"""
        target = self.config.TARGET_METRIC.lower()
        n_methods = len(self.config.METHODS)
        n_models = len(self.config.MODELS)
        data = np.zeros((n_methods, n_models))

        # 键名映射：不同文件可能用不同的键名存储同一个指标
        key_map = {
            'rmse': ['rmse', 'RMSE'],
            'mae': ['mae', 'MAE'],
            'mse': ['mse', 'MSE'],
            'pcc': ['pcc', 'PCC', 'mean_pcc', 'pearson', 'correlation'],
            'nse': ['nse', 'NSE']
        }
        # 获取可能的键名列表
        search_keys = key_map.get(target, [target])

        for i, (method_key, _, _) in enumerate(self.config.METHODS):
            method_dict = self.metrics_data.get(method_key, {})

            for j, model_name in enumerate(self.config.MODELS):
                model_metrics = method_dict.get(model_name, {})

                val = np.nan
                # 1. 尝试直接查找
                for k in search_keys:
                    if k in model_metrics:
                        val = model_metrics[k]
                        break

                # 2. 特殊处理：如果没有MSE但有RMSE，自动计算平方
                if np.isnan(val) and target == 'mse':
                    for k in ['rmse', 'RMSE']:
                        if k in model_metrics:
                            val = model_metrics[k] ** 2
                            break

                # 3. 格式转换
                if isinstance(val, (np.ndarray, list)):
                    val = float(val) if val else np.nan

                data[i, j] = val

        return data


# ==================== 3. 绘图主程序 ====================
def main():
    cfg = Config()
    loader = DataLoader(cfg)
    loader.load_data()
    data_matrix = loader.get_plot_data()  # shape: (n_methods, n_models)

    print(f"🎨 开始绘制: {len(cfg.MODELS)}个模型 x {cfg.TARGET_METRIC.upper()}")

    # 设置画布
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH, cfg.FIG_HEIGHT), dpi=cfg.DPI)

    # 计算柱子位置
    n_models = len(cfg.MODELS)
    n_methods = len(cfg.METHODS)
    indices = np.arange(n_models)

    total_width = 0.7  # 一组柱子的总宽度
    bar_width = total_width / n_methods

    # 循环绘制每种方法的柱子
    for i, (method_key, label, color) in enumerate(cfg.METHODS):
        # 计算偏移量，让柱子居中排列
        offset = (i - n_methods / 2 + 0.5) * bar_width
        x_pos = indices + offset

        values = data_matrix[i, :]

        # 绘图
        ax.bar(x_pos, values, width=bar_width, label=label,
               color=color, edgecolor='none', linewidth=0.5, zorder=3)

    # 装饰图表
    metric_name = cfg.TARGET_METRIC.upper()
    ax.set_ylabel(metric_name, fontsize=18, fontweight='normal')
    # ax.set_title(f'{metric_name} Comparison across Models', fontsize=16, fontweight='bold', pad=15)

    # 设置 Y 轴刻度数字的大小
    # axis='y': 只改Y轴
    # labelsize=12: 设置字号大小 (根据需要修改数字)
    ax.tick_params(axis='y', labelsize=16)

    # X轴设置
    ax.set_xticks(indices)
    ax.set_xticklabels(cfg.MODELS, rotation=35, ha='right', fontsize=18, fontweight='normal')
    # ax.set_xlabel('Models', fontsize=14, fontweight='bold')

    # 网格与图例
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    # 去掉上面和右边的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例设置
    ax.legend(
        loc='lower center',  # 【修改点1】将图例的"下边缘中心"作为锚点
        bbox_to_anchor=(0.5, 1.01),  # 【修改点2】坐标位置 (x=0.5居中, y>1.0 表示在图表上方)
        ncol=len(cfg.METHODS),  # 设置列数等于方法数 -> 横向排列
        frameon=False,  # 无边框
        fontsize=18,
        edgecolor='black',
        facecolor='white',
        framealpha=0.9,
        columnspacing=0.8,  # 列间距
        handletextpad=0.4  # 图标与文字间距
    )

    # 调整X轴范围，减少左右留白
    # 比如从 -0.6 到 n_models - 0.4
    ax.set_xlim(-0.6, len(cfg.MODELS) - 0.4)

    # 如果是 PCC (相关系数)，通常范围是 0-1，可以固定一下
    if cfg.TARGET_METRIC.lower() == 'pcc':
        ax.set_ylim(0.4, 1.0)

    # 保存
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
    filename = f'{cfg.TARGET_METRIC}_comparison_{n_models}models.png'
    save_path = Path(cfg.OUTPUT_DIR) / filename

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 图片已保存: {save_path}")
    # plt.show()


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    main()