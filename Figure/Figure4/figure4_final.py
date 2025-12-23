import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from scipy.stats import gaussian_kde
import pickle


# ==================== 配置部分 ====================
class Config:
    # --- 模型列表 (共21个) ---
    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
        'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
        'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
        'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    # --- 方法列表 ---
    COMPARE_METHODS = [
        ('base', 'Control', 'black', '-'),
        ('EDCDF', 'EDCDF', '#D8211C', '-'),
        # ('linear_reg', 'Linear Reg', '#fc945d', '-.'),
        ('ConvLSTM', 'ConvLSTM', '#fc945d', '-.'),
        ('UNet', 'UNet', '#4baf73', '-'),
        ('MambaUNet', 'MambaUNet', '#299BCF', '-'),
    ]

    # --- 数据路径 ---
    METHODS_CONFIG = {
        'base': {
            'cmip': '../../Preprocessing/dataset/{model}/ssp245_test.npz',
            'obs': '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
        },
        'EDCDF': {
            'pred': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_corrections.npy',
            'true': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_trues.npy'
        },
        # 'linear_reg': {
        #     'pred': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_corrections.npy',
        #     'true': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_trues.npy'
        # },
        'ConvLSTM': {
             'pred': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
             'true': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        },
        'UNet': {
            'pred': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        },
        'MambaUNet': {
            'pred': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        }
    }

    # --- 【新增】缓存文件路径 ---
    CACHE_DIR = './cache_data'
    CACHE_FILE = 'figure3_bias_cache.pkl'

    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'
    OUTPUT_DIR = './histogram_figures'
    OUTPUT_FILENAME = 'bias_comparison_7x3_V2.png'
    DPI = 300
    N_COLS = 3
    N_ROWS = 7


# ==================== 数据加载器 (带缓存功能) ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mask = None
        self.obs_data = None  # 缓存观测数据

        # 创建缓存目录
        Path(config.CACHE_DIR).mkdir(exist_ok=True)
        self.cache_path = Path(config.CACHE_DIR) / config.CACHE_FILE

    def load_geo_info(self):
        """加载掩码"""
        if Path(self.config.MASK_PATH).exists():
            self.mask = np.load(self.config.MASK_PATH)


    def load_obs(self):
        """统一加载观测真值 obs（全局使用）"""
        if self.obs_data is None:
            obs_path = self.config.METHODS_CONFIG['base']['obs']
            self.obs_data = np.load(obs_path)['sst'][-1827:]
        return self.obs_data

    def calculate_bias_map(self, model_name, method_key):
        """
        计算 bias map (逐像素): model_output - obs
        返回: (H, W) 的空间平均 bias map
        """
        method_paths = self.config.METHODS_CONFIG.get(method_key)
        if not method_paths:
            return None

        paths = {k: v.format(model=model_name) for k, v in method_paths.items()}

        try:
            obs = self.load_obs()

            # === base（CMIP6 Raw）===
            if method_key == 'base':
                cmip = np.load(paths['cmip'])['sst']
                min_len = min(cmip.shape[0], obs.shape[0])
                model_output = cmip[:min_len]

            # === 其余方法（pred vs obs）===
            else:
                pred = np.load(paths['pred'])
                min_len = min(pred.shape[0], obs.shape[0])
                model_output = pred[:min_len]

            # 统一 bias 计算: (T, H, W)
            bias = model_output - obs[:min_len]

            # 时间平均 -> (H, W)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bias_map = np.nanmean(bias, axis=0)

            # 应用mask
            if self.mask.shape == bias_map.shape:
                bias_map[~self.mask.astype(bool)] = np.nan

            return bias_map

        except Exception as e:
            print(f"   ⚠️ {model_name} [{method_key}] Failed: {e}")
            return None

    def load_or_compute_all_bias(self, force_recompute=False):
        """
        加载或计算所有模型和方法的 bias 数据

        Returns:
            bias_data: {model_name: {method_key: bias_map_array}}
        """
        # 检查缓存
        if not force_recompute and self.cache_path.exists():
            print(f"📦 发现缓存文件，正在加载: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    bias_data = pickle.load(f)
                print(f"   ✅ 缓存加载成功！")
                return bias_data
            except Exception as e:
                print(f"   ⚠️ 缓存加载失败: {e}，将重新计算")

        # 重新计算
        print(f"🔄 开始计算所有 bias 数据...")
        bias_data = {}

        for i, model in enumerate(self.config.MODELS, 1):
            print(f"   [{i}/{len(self.config.MODELS)}] 处理: {model}")
            bias_data[model] = {}

            for method_key, _, _, _ in self.config.COMPARE_METHODS:
                bias_map = self.calculate_bias_map(model, method_key)
                if bias_map is not None:
                    bias_data[model][method_key] = bias_map

        # 保存缓存
        print(f"\n💾 保存缓存到: {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(bias_data, f)
        print(f"   ✅ 缓存保存成功！")

        return bias_data


# ==================== 绘图函数 ====================
def plot_comparison_histograms(force_recompute=False):
    """
    绘制 bias 对比直方图

    Args:
        force_recompute: 是否强制重新计算数据（忽略缓存）
    """
    config = Config()
    loader = DataLoader(config)
    loader.load_geo_info()

    # 【关键】加载或计算所有数据
    bias_data = loader.load_or_compute_all_bias(force_recompute=force_recompute)

    n_models = len(config.MODELS)
    print(f"\n🎨 开始绘制 {n_models} 个模型...")

    fig, axes = plt.subplots(config.N_ROWS, config.N_COLS,
                             figsize=(5.5 * config.N_COLS, 3.5 * config.N_ROWS),
                             dpi=config.DPI,
                             sharex=True,
                             sharey=True)

    axes_flat = axes.flatten()
    HIST_RANGE = (-2, 2)
    handles, labels = [], []

    for i in range(len(axes_flat)):
        ax = axes_flat[i]

        if i >= n_models:
            ax.axis('off')
            continue

        model_name = config.MODELS[i]
        print(f"   [{i + 1}/{n_models}] 绘制: {model_name}")

        stats_lines = []

        for method_key, label_name, color, linestyle in config.COMPARE_METHODS:
            # 从缓存中获取数据
            bias_map = bias_data.get(model_name, {}).get(method_key)
            if bias_map is None:
                continue

            # 展平并去除NaN
            data = bias_map.flatten()
            valid_data = data[~np.isnan(data)]
            if len(valid_data) < 10:
                continue

            # 计算统计量
            mean_bias = np.mean(valid_data)
            stats_lines.append(f"{label_name}: {mean_bias:+.2f}")

            # 绘制密度曲线
            try:
                density = gaussian_kde(valid_data)
                xs = np.linspace(HIST_RANGE[0], HIST_RANGE[1], 200)
                ys = density(xs)

                line, = ax.plot(xs, ys, color=color, linestyle=linestyle,
                                linewidth=1.5, label=label_name, alpha=0.9)

                if i == 0:
                    handles.append(line)
                    labels.append(label_name)
            except Exception:
                pass

        # 统计文字
        if stats_lines:
            stats_text = "\n".join(stats_lines)
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=17, linespacing=1.5)

        # 装饰
        ax.set_title(model_name, fontsize=24, fontweight='bold', pad=8)
        ax.set_xlim(-2, 2)
        ax.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        ax.set_ylim(-0.1, 5.0)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)

        if i // config.N_COLS == config.N_ROWS - 1:
            ax.set_xlabel('Bias (°C)', fontsize=19)

        if i % config.N_COLS == 0:
            ax.set_ylabel('Density', fontsize=19)

    # 图例
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.subplots_adjust(hspace=0.25, wspace=0.10)

    leg = fig.legend(handles, labels,
                     loc='lower center',
                     bbox_to_anchor=(0.5, 0.01),
                     ncol=len(handles),
                     fontsize=20,
                     frameon=False,
                     columnspacing=2.5,
                     handlelength=3.0)

    for line in leg.get_lines():
        line.set_linewidth(4.0)

    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = Path(config.OUTPUT_DIR) / config.OUTPUT_FILENAME
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    # 第一次运行设置为 True 强制计算，后续设置为 False 使用缓存
    plot_comparison_histograms(force_recompute=True)