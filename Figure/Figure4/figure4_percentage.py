import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
from scipy.stats import gaussian_kde
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial


# ==================== 配置部分 ====================
class Config:
    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
        'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC',
        'EC-Earth3-Veg-LR', 'EC-Earth3-veg', 'EC-Earth3', 'GFDL-CM4',
        'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM'
    ]

    COMPARE_METHODS = [
        ('base', 'Control', 'black', '-'),
        ('EDCDF', 'EDCDF', '#D8211C', '-'),
        ('ConvLSTM', 'ConvLSTM', '#fc945d', '-.'),
        ('UNet', 'UNet', '#4baf73', '-'),
        ('MambaUNet', 'MambaUNet', '#299BCF', '-'),
    ]

    METHODS_CONFIG = {
        'base': {
            'cmip': '../../Preprocessing/dataset/{model}/ssp245_test.npz',
            'obs': '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
        },
        'EDCDF': {
            'pred': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_corrections.npy',
            'true': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_trues.npy'
        },
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

    CACHE_DIR = './cache_data'
    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'
    OUTPUT_DIR = './histogram_figures'
    OUTPUT_FILENAME = 'bias_timeseries_7x3_day_mean.png'
    DPI = 300
    N_COLS = 3
    N_ROWS = 7


# ==================== 数据加载器（保持原样，不修改）====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mask = None
        self.obs_data = None
        Path(config.CACHE_DIR).mkdir(exist_ok=True)

    def load_geo_info(self):
        """加载掩码"""
        if Path(self.config.MASK_PATH).exists():
            self.mask = np.load(self.config.MASK_PATH)

    def load_obs(self):
        """统一加载观测真值"""
        if self.obs_data is None:
            obs_path = self.config.METHODS_CONFIG['base']['obs']
            self.obs_data = np.load(obs_path)['sst'][-1827:]
        return self.obs_data

    def calculate_bias_flattened(self, model_name, method_key):
        """直接返回展平且过滤后的有效数据（节省内存）"""
        method_paths = self.config.METHODS_CONFIG.get(method_key)
        if not method_paths:
            return None

        paths = {k: v.format(model=model_name) for k, v in method_paths.items()}

        try:
            obs = self.load_obs()

            # 加载数据
            if method_key == 'base':
                cmip = np.load(paths['cmip'])['sst']
                min_len = min(cmip.shape[0], obs.shape[0])
                model_output = cmip[:min_len]
            else:
                pred = np.load(paths['pred'])
                min_len = min(pred.shape[0], obs.shape[0])
                model_output = pred[:min_len]

            # 计算bias
            bias = model_output - obs[:min_len]

            # 应用mask
            if self.mask is not None:
                mask_3d = np.broadcast_to(self.mask, bias.shape)
                bias = np.where(mask_3d.astype(bool), bias, np.nan)

            # 展平并过滤NaN
            valid_data = bias.flatten()
            valid_data = valid_data[~np.isnan(valid_data)]

            return valid_data

        except Exception as e:
            print(f"      ⚠️ {model_name} [{method_key}] Failed: {e}")
            return None

    def load_or_compute_bias_for_model(self, model_name, force_recompute=False):
        """每个模型单独加载/计算缓存"""
        model_cache = Path(self.config.CACHE_DIR) / f"bias_daily_{model_name}.pkl"

        # 尝试加载缓存
        if not force_recompute and model_cache.exists():
            try:
                with open(model_cache, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"      ⚠️ 缓存加载失败: {e}，将重新计算")

        # 计算数据
        model_data = {}
        for method_key, _, _, _ in self.config.COMPARE_METHODS:
            valid_data = self.calculate_bias_flattened(model_name, method_key)
            if valid_data is not None:
                model_data[method_key] = valid_data

        # 保存缓存
        with open(model_cache, 'wb') as f:
            pickle.dump(model_data, f)

        return model_data

    def load_or_compute_kde_for_model(self, model_name, force_recompute=False):
        """KDE 缓存优先级（保持原样）"""
        kde_cache = Path(self.config.CACHE_DIR) / f"kde_{model_name}.pkl"

        if kde_cache.exists() and not force_recompute:
            try:
                with open(kde_cache, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ KDE缓存读取失败 {model_name}: {e}")

        bias_cache = Path(self.config.CACHE_DIR) / f"bias_daily_{model_name}.pkl"
        if bias_cache.exists():
            try:
                with open(bias_cache, "rb") as f:
                    model_bias = pickle.load(f)
            except Exception as e:
                print(f"⚠️ Bias缓存读取失败 {model_name}: {e}")
                model_bias = None
        else:
            model_bias = None

        if model_bias is None:
            print(f"   🔍 计算Bias数据: {model_name}")
            model_bias = self.load_or_compute_bias_for_model(
                model_name, force_recompute=True
            )

        kde_data = {}
        for method_key, _, _, _ in self.config.COMPARE_METHODS:
            valid_data = model_bias.get(method_key)
            if valid_data is None or len(valid_data) < 100:
                continue

            try:
                kde_data[method_key] = compute_kde_from_bias(valid_data)
            except Exception as e:
                print(f"⚠️ KDE计算失败 {model_name}-{method_key}: {e}")

        try:
            with open(kde_cache, "wb") as f:
                pickle.dump(kde_data, f)
        except Exception as e:
            print(f"⚠️ KDE缓存保存失败 {model_name}: {e}")

        return kde_data


# ==================== ✅ 新增：计算平均bias的函数 ====================
def compute_mean_bias_for_model(args):
    """
    从缓存的bias数据中计算平均bias（用于并行计算）

    Args:
        args: (model_name, config)

    Returns:
        (model_name, {method_key: mean_bias})
    """
    model_name, config = args

    try:
        bias_cache = Path(config.CACHE_DIR) / f"bias_daily_{model_name}.pkl"

        if not bias_cache.exists():
            print(f"   ⚠️ 未找到缓存: {model_name}")
            return model_name, None

        # 加载bias数据
        with open(bias_cache, 'rb') as f:
            model_bias = pickle.load(f)

        # 计算每个方法的平均bias
        mean_bias_dict = {}
        for method_key, _, _, _ in config.COMPARE_METHODS:
            valid_data = model_bias.get(method_key)
            if valid_data is not None and len(valid_data) > 0:
                mean_bias_dict[method_key] = float(np.mean(valid_data))

        print(f"   ✅ 平均bias计算完成: {model_name}")
        return model_name, mean_bias_dict

    except Exception as e:
        print(f"   ❌ 计算失败 {model_name}: {e}")
        return model_name, None


def load_or_compute_all_mean_bias(config, use_multiprocessing=True, n_workers=None):
    """
    ✅ 新增：并行加载/计算所有模型的平均bias

    Returns:
        {model_name: {method_key: mean_bias}}
    """
    mean_bias_cache = Path(config.CACHE_DIR) / "mean_bias_all_models.pkl"

    # 尝试加载总缓存
    if mean_bias_cache.exists():
        try:
            with open(mean_bias_cache, 'rb') as f:
                all_mean_bias = pickle.load(f)
            print(f"✅ 加载平均bias缓存: {len(all_mean_bias)} 个模型")
            return all_mean_bias
        except Exception as e:
            print(f"⚠️ 平均bias缓存加载失败: {e}，将重新计算")

    # 重新计算
    print("\n📊 计算所有模型的平均bias...")

    all_mean_bias = {}
    tasks = [(model_name, config) for model_name in config.MODELS]

    if use_multiprocessing:
        if n_workers is None:
            n_workers = min(cpu_count() - 1, len(config.MODELS))

        print(f"🚀 使用 {n_workers} 个进程并行计算...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_model = {
                executor.submit(compute_mean_bias_for_model, task): task[0]
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_name, mean_bias_dict = future.result()
                    if mean_bias_dict is not None:
                        all_mean_bias[model_name] = mean_bias_dict
                        completed += 1
                except Exception as e:
                    print(f"   ❌ 处理失败 {model_name}: {e}")
    else:
        # 串行计算
        for i, task in enumerate(tasks, 1):
            print(f"   [{i}/{len(tasks)}] 计算: {task[0]}")
            model_name, mean_bias_dict = compute_mean_bias_for_model(task)
            if mean_bias_dict is not None:
                all_mean_bias[model_name] = mean_bias_dict

    # 保存总缓存
    try:
        with open(mean_bias_cache, 'wb') as f:
            pickle.dump(all_mean_bias, f)
        print(f"💾 平均bias缓存已保存")
    except Exception as e:
        print(f"⚠️ 平均bias缓存保存失败: {e}")

    return all_mean_bias


# ==================== KDE计算函数（保持原样）====================
def compute_kde_from_bias(valid_data, hist_range=(-2, 2), n_points=200):
    """从 bias 一维数组计算 KDE"""
    xs = np.linspace(hist_range[0], hist_range[1], n_points)
    kde = gaussian_kde(valid_data)
    ys = kde(xs)

    peak_idx = np.argmax(ys)
    peak_bias = xs[peak_idx]

    return {
        "xs": xs.astype(np.float32),
        "ys": ys.astype(np.float32),
        "peak_bias": float(peak_bias)
    }


def compute_kde_for_model(args):
    """用于并行计算的函数（保持原样）"""
    model_name, config, force_recompute = args
    try:
        loader = DataLoader(config)
        loader.load_geo_info()
        kde_data = loader.load_or_compute_kde_for_model(model_name, force_recompute)
        print(f"   ✅ KDE完成: {model_name}")
        return model_name, kde_data
    except Exception as e:
        print(f"   ❌ KDE失败: {model_name} - {e}")
        return model_name, None


def process_single_model(model_name, config, force_recompute):
    """单个模型的处理函数（保持原样）"""
    try:
        loader = DataLoader(config)
        loader.load_geo_info()
        print(f"   🔄 正在处理: {model_name}")
        model_data = loader.load_or_compute_bias_for_model(model_name, force_recompute)
        print(f"   ✅ 完成: {model_name}")
        return model_name, model_data
    except Exception as e:
        print(f"   ❌ 失败: {model_name} - {e}")
        return model_name, None


# ==================== 新增：计算频率百分比并返回平滑曲线 ====================
def compute_smooth_frequency_curve(args):
    """计算频率百分比并生成平滑曲线（用于并行计算）"""
    model_name, config = args

    try:
        # 加载bias数据
        bias_cache = Path(config.CACHE_DIR) / f"bias_daily_{model_name}.pkl"
        if not bias_cache.exists():
            return model_name, None

        with open(bias_cache, 'rb') as f:
            model_bias = pickle.load(f)

        frequency_data = {}

        # 使用更多的bin来获得更平滑的曲线
        bins = np.linspace(-2, 2, 81)  # 80个区间，更细的分辨率

        for method_key, label_name, _, _ in config.COMPARE_METHODS:
            valid_data = model_bias.get(method_key)
            if valid_data is None or len(valid_data) == 0:
                continue

            # 计算频率直方图
            counts, bin_edges = np.histogram(valid_data, bins=bins, density=False)

            # 转换为百分比
            total_samples = len(valid_data)
            percentages = (counts / total_samples) * 100

            # 计算每个区间的中点
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # ===== 关键：使用移动平均或插值来平滑曲线 =====
            # 方法1：使用移动平均平滑
            window_size = 5  # 滑动窗口大小
            smoothed_percentages = np.convolve(
                percentages,
                np.ones(window_size) / window_size,
                mode='same'
            )

            # 方法2：使用样条插值（更平滑）
            # from scipy.interpolate import make_interp_spline
            # spline = make_interp_spline(bin_centers, percentages, k=3)
            # xs_smooth = np.linspace(bin_centers[0], bin_centers[-1], 200)
            # smoothed_percentages = spline(xs_smooth)

            # 这里使用方法1，不需要额外依赖
            frequency_data[method_key] = {
                'xs': bin_centers.astype(np.float32),
                'ys': smoothed_percentages.astype(np.float32),
                'total_samples': total_samples
            }

        return model_name, frequency_data

    except Exception as e:
        print(f"   ❌ 平滑频率计算失败 {model_name}: {e}")
        return model_name, None


# ==================== 修改绘图函数 ====================
def plot_comparison_frequency_curves(force_recompute=False, use_multiprocessing=True, n_workers=None):
    """绘制 bias 频率百分比曲线图"""
    config = Config()

    n_models = len(config.MODELS)
    print(f"\n🎨 开始处理 {n_models} 个模型...")

    # ===== 1️⃣ 检查是否需要重新计算bias数据 =====
    if force_recompute:
        print("🔄 重新计算Bias数据...")
        loader = DataLoader(config)
        loader.load_geo_info()

        if use_multiprocessing:
            if n_workers is None:
                n_workers = min(cpu_count() - 1, n_models)

            print(f"🚀 使用 {n_workers} 个进程并行计算Bias...")

            with Pool(processes=n_workers) as pool:
                process_func = partial(process_single_model,
                                       config=config,
                                       force_recompute=force_recompute)
                results = pool.map(process_func, config.MODELS)

            print("✅ Bias数据计算完成")
        else:
            for i, model_name in enumerate(config.MODELS, 1):
                print(f"   [{i}/{n_models}] 处理: {model_name}")
                loader.load_or_compute_bias_for_model(model_name, force_recompute)

    # ===== 2️⃣ 计算平滑的频率百分比曲线 =====
    print("\n📊 计算平滑的频率百分比曲线...")

    all_freq_data = {}
    tasks = [(model_name, config) for model_name in config.MODELS]

    if use_multiprocessing:
        if n_workers is None:
            n_workers = min(cpu_count() - 1, n_models)

        print(f"🚀 使用 {n_workers} 个进程并行计算频率曲线...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_model = {
                executor.submit(compute_smooth_frequency_curve, task): task[0]
                for task in tasks
            }

            completed = 0
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_name, freq_data = future.result()
                    if freq_data is not None:
                        all_freq_data[model_name] = freq_data
                        completed += 1
                        print(f"   [{completed}/{n_models}] 完成: {model_name}")
                except Exception as e:
                    print(f"   ❌ 处理失败 {model_name}: {e}")
    else:
        # 串行计算
        for i, task in enumerate(tasks, 1):
            print(f"   [{i}/{len(tasks)}] 计算频率曲线: {task[0]}")
            model_name, freq_data = compute_smooth_frequency_curve(task)
            if freq_data is not None:
                all_freq_data[model_name] = freq_data

    print(f"\n✅ 频率曲线数据计算完成: {len(all_freq_data)}/{n_models} 个模型")

    # ===== 3️⃣ ✅ 加载/计算所有模型的平均bias =====
    all_mean_bias = load_or_compute_all_mean_bias(config, use_multiprocessing, n_workers)

    # ===== 4️⃣ 开始绘图（频率百分比曲线）=====
    print("\n🖼️ 开始绘制频率百分比曲线图...")

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

        # 从并行计算结果中获取频率数据
        model_freq = all_freq_data.get(model_name, {})

        # ✅ 获取平均bias数据
        model_mean_bias = all_mean_bias.get(model_name, {})

        stats_lines = []

        for method_key, label_name, color, linestyle in config.COMPARE_METHODS:
            freq_data = model_freq.get(method_key)
            if freq_data is None:
                continue

            xs = freq_data["xs"]
            ys = freq_data["ys"]

            # 绘制频率曲线（平滑曲线）
            line, = ax.plot(xs, ys,
                            color=color,
                            linestyle=linestyle,
                            linewidth=2.0,  # 稍微粗一点，更清晰
                            label=label_name,
                            alpha=0.8)

            if i == 0 and label_name not in labels:
                handles.append(line)
                labels.append(label_name)

            # ✅ 右上角：显示平均bias
            mean_bias = model_mean_bias.get(method_key)
            if mean_bias is not None:
                if abs(mean_bias) < 0.005:  # 如果非常接近0
                    stats_lines.append(f"{label_name}:  0.00")
                else:
                    stats_lines.append(f"{label_name}: {mean_bias: .2f}")

        # 统计文字
        if stats_lines:
            stats_text = "\n".join(stats_lines)
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=17, linespacing=1.5)

        # 装饰
        ax.set_title(model_name, fontsize=24, fontweight='bold', pad=8)
        ax.set_xlim(-2, 2)
        ax.set_xticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

        # ===== 固定纵坐标范围为 0-25% =====
        Y_MAX = 20.0
        ax.set_ylim(0, Y_MAX)
        ax.set_yticks([0, 4, 8, 12, 16, 20])

        # 修改纵坐标标签为百分比
        if i % config.N_COLS == 0:
            ax.set_ylabel('Percentage (%)', fontsize=19)

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)

        if i // config.N_COLS == config.N_ROWS - 1:
            ax.set_xlabel('Error (°C)', fontsize=19)

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

    # 修改输出文件名
    save_path = Path(config.OUTPUT_DIR) / 'bias_frequency_curves_7x3_day.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    plot_comparison_frequency_curves(
        force_recompute=False,
        use_multiprocessing=True,
        n_workers=10
    )