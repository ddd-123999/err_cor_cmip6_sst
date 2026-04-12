import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pickle
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count
from functools import partial


# ==================== 配置部分 ====================
class Config:
    # --- 基础配置 ---
    START_DATE = '2020-01-01'
    PERIODS = 1827
    OUTPUT_DIR = './seasonal_metrics'
    DPI = 600

    # 【缓存配置】
    CACHE_DIR = './cache_data_seasonal'
    CACHE_FILE = 'seasonal_metrics_all_cache_yearly.pkl'  # ✅ 改名避免与旧缓存冲突

    # ✅ 【重要】缓存中存储所有4个指标
    ALL_METRICS = ['rmse', 'mae', 'bias', 'pcc']

    # 【关键1】要绘制的指标列表 (从所有指标中选择，最多2个)
    TARGET_METRICS = ['bias', 'pcc']  # ✅ 可随时修改这里来切换绘图指标

    METRIC_LABELS = {
        'rmse': 'RMSE (°C)',
        'mae': 'MAE (°C)',
        'bias': 'Bias (°C)',
        'pcc': 'PCC'
    }

    # 【关键2】纵坐标范围
    Y_LIMS = {
        'rmse': (0, 1.0),
        'bias': (-1.5, 1.5),
        'mae': (0, 1.0),
        'pcc': (0, 1.0),
        'default': None
    }

    # 【关键3】要绘制的方法列表
    METHODS_TO_PLOT = [
        ('base', 'Control'),
        ('EDCDF', 'EDCDF'),
        # ('linear_reg', 'Linear Regression'),
        ('ConvLSTM', 'ConvLSTM'),
        ('UNet', 'UNet'),
        ('MambaUNet', 'Mamba-TempNet'),
    ]

    # --- 数据路径配置 ---
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

    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
        'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
        'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
        'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'


# ==================== 数据加载与计算 ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mask = None
        self.full_time_index = pd.date_range(start=config.START_DATE, periods=config.PERIODS, freq='D')
        self.load_mask()

    def load_mask(self):
        """加载掩码数据"""
        if Path(self.config.MASK_PATH).exists():
            self.mask = np.load(self.config.MASK_PATH).astype(bool)

    def get_spatial_mean_series(self, data):
        """计算空间平均值的时间序列"""
        data = data[:self.config.PERIODS]

        if self.mask is not None and data.shape[1:] == self.mask.shape:
            data_masked = data.copy()
            data_masked[:, ~self.mask] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                spatial_mean = np.nanmean(data_masked, axis=(1, 2))
        else:
            spatial_mean = np.nanmean(data, axis=(1, 2))

        idx = self.full_time_index[:len(spatial_mean)]
        return pd.Series(spatial_mean, index=idx)

    def get_data_pair(self, method_key, model_name):
        """获取预测值和真实值的空间平均序列"""
        paths = self.config.METHODS_CONFIG.get(method_key)
        if not paths:
            return None, None
        p = {k: v.format(model=model_name) for k, v in paths.items()}

        try:
            if 'cmip' in p:
                pred_data = np.load(p['cmip'])['sst']
            elif 'pred' in p:
                pred_data = np.load(p['pred'])
            else:
                return None, None

            if 'obs' in p:
                true_data = np.load(p['obs'])['sst']
                if true_data.shape[0] > self.config.PERIODS:
                    true_data = true_data[-self.config.PERIODS:]
            elif 'true' in p:
                true_data = np.load(p['true'])
            else:
                return None, None

            T = min(pred_data.shape[0], true_data.shape[0])
            pred_s = self.get_spatial_mean_series(pred_data[:T])
            true_s = self.get_spatial_mean_series(true_data[:T])

            if pred_s.empty or true_s.empty:
                return None, None

            return pred_s, true_s

        except Exception as e:
            return None, None

    def calculate_monthly_metric(self, pred_s, true_s, metric):
        """
        ✅ 修改为：先分别计算每年的月度指标（5年×12月=60个值），再按月平均

        步骤：
        1. 对每一年，计算12个月的指标 → 得到5组12个月的值
        2. 对每个月（1-12），平均5年的该月指标值
        3. 返回12个月的平均指标值
        """
        # 获取所有年份
        years = pred_s.index.year.unique()
        monthly_values_by_year = []  # 存储每年的月度指标列表

        # 先计算每年的月度指标
        for year in years:
            year_mask = pred_s.index.year == year
            pred_y = pred_s[year_mask]
            true_y = true_s[year_mask]
            diff_y = pred_y - true_y

            year_monthly_vals = []

            for month in range(1, 13):
                month_mask = pred_y.index.month == month
                p_ym = pred_y[month_mask].dropna()
                t_ym = true_y[month_mask].dropna()
                d_ym = diff_y[month_mask].dropna()

                if len(d_ym) == 0:
                    year_monthly_vals.append(np.nan)
                    continue

                if metric == 'rmse':
                    val = np.sqrt(np.mean(d_ym ** 2))
                elif metric == 'mae':
                    val = np.mean(np.abs(d_ym))
                elif metric == 'bias':
                    val = np.mean(d_ym)
                elif metric == 'pcc':
                    if len(p_ym) > 1:
                        val = np.corrcoef(p_ym, t_ym)[0, 1]
                    else:
                        val = np.nan
                else:
                    val = np.nan

                year_monthly_vals.append(val)

            monthly_values_by_year.append(year_monthly_vals)

        # 对每个月，平均5年的值
        results = []
        monthly_values_by_year = np.array(monthly_values_by_year)  # 形状: (5年, 12月)

        for month in range(12):
            month_vals = monthly_values_by_year[:, month]  # 该月5年的值
            month_vals = month_vals[~np.isnan(month_vals)]  # 去掉NaN

            if len(month_vals) > 0:
                results.append(np.mean(month_vals))
            else:
                results.append(np.nan)

        return results

    def load_or_calculate_all_metrics(self, force_recompute=False, n_workers=10):
        """✅ 改进：使用并行计算加速，缓存所有4个指标的数据"""
        cache_path = Path(self.config.CACHE_DIR) / self.config.CACHE_FILE

        # 1. 尝试加载缓存
        if not force_recompute and cache_path.exists():
            print(f"💾 正在从缓存加载数据: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                # ✅ 检查缓存是否包含所有4个指标
                all_keys_present = True
                required_keys = [(m[0], mk) for m in self.config.METHODS_TO_PLOT
                                 for mk in self.config.ALL_METRICS]
                for key in required_keys:
                    if key not in cached_data or not cached_data[key]:
                        all_keys_present = False
                        break

                if all_keys_present:
                    print("✅ 缓存完整，已加载所有4个指标的数据")
                    return cached_data
                else:
                    print("⚠️ 缓存文件不完整或过期，将重新计算。")
            except Exception as e:
                print(f"❌ 缓存文件加载失败 ({e})，将重新计算。")

        # 2. ✅ 使用并行计算所有4个指标的数据
        print(f"💻 正在使用 {n_workers} 个进程并行计算所有指标和月度数据...")

        # 准备所有任务
        tasks = []
        for method_key, method_label in self.config.METHODS_TO_PLOT:
            for metric_key in self.config.ALL_METRICS:
                tasks.append((method_key, method_label, metric_key))

        total_tasks = len(tasks)
        print(f"   总任务数: {total_tasks}")

        # 使用进程池并行计算
        all_data_cache = {}

        with Pool(processes=n_workers) as pool:
            # 使用偏函数传递self.config
            process_func = partial(self._process_single_task, config=self.config)
            results = pool.map(process_func, tasks)

        # 整理结果
        for (method_key, metric_key), model_results_list in results:
            all_data_cache[(method_key, metric_key)] = model_results_list
            print(f"   ✅ 完成: {method_key} - {metric_key}")

        # 3. 保存缓存
        Path(self.config.CACHE_DIR).mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data_cache, f)
        print(f"✅ 数据计算完成，已保存所有4个指标至缓存: {cache_path}")

        return all_data_cache

    @staticmethod
    def _process_single_task(task, config):
        """
        ✅ 静态方法：处理单个任务（方法+指标）
        用于多进程并行计算
        """
        method_key, method_label, metric_key = task
        print(f"   🔄 处理: {method_label} - {metric_key}")

        # 创建临时DataLoader实例
        loader = DataLoader(config)

        model_results_list = []
        for model in config.MODELS:
            pred_s, true_s = loader.get_data_pair(method_key, model)

            if pred_s is not None and true_s is not None:
                monthly_vals = loader.calculate_monthly_metric(pred_s, true_s, metric_key)
                model_results_list.append(monthly_vals)

        return (method_key, metric_key), model_results_list


# ==================== 绘图主程序 ====================
def plot_multi_metric_comparison(force_recompute=False, n_workers=10):
    """
    绘制多指标对比图

    Args:
        force_recompute: 是否强制重新计算（忽略缓存）
        n_workers: 并行进程数，默认10
    """
    c = Config()
    loader = DataLoader(c)

    # ✅ 加载包含所有4个指标的缓存数据（带并行参数）
    all_metric_data = loader.load_or_calculate_all_metrics(
        force_recompute=force_recompute,
        n_workers=n_workers
    )

    # ✅ 从缓存中提取要绘制的指标
    metric_keys = c.TARGET_METRICS
    n_metrics = len(metric_keys)
    n_methods = len(c.METHODS_TO_PLOT)

    if n_metrics > 2:
        print("⚠️ 警告: TARGET_METRICS 最多支持两个指标进行两列绘制。")
        return

    print(f"🚀 开始绘制 {metric_keys} 的 Monthly Climatology 对比图")

    fig, axes = plt.subplots(n_methods, n_metrics, figsize=(5.5 * n_metrics, 3.5 * n_methods),
                             dpi=c.DPI, sharex=True, squeeze=False)

    colors = plt.cm.jet(np.linspace(0, 1, len(c.MODELS)))
    x_positions = range(1, 13)

    legend_handles = []
    legend_labels = []
    legend_collected = False

    for r, (method_key, method_label) in enumerate(c.METHODS_TO_PLOT):
        for col, metric_key in enumerate(metric_keys):
            ax = axes[r, col]
            metric_label = c.METRIC_LABELS.get(metric_key, metric_key.upper())

            data_key = (method_key, metric_key)
            all_models_monthly = all_metric_data.get(data_key, [])

            if not all_models_monthly:
                ax.text(0.5, 0.5, "Data Missing", transform=ax.transAxes,
                        ha='center', va='center')
                continue

            # 绘制每个模型
            for j, model in enumerate(c.MODELS):
                if j < len(all_models_monthly):
                    monthly_vals = all_models_monthly[j]
                    line, = ax.plot(x_positions, monthly_vals,
                                    color=colors[j], linewidth=1, alpha=0.6)

                    if not legend_collected:
                        legend_handles.append(line)
                        legend_labels.append(model)

            # 绘制集合平均线
            avg_vals = np.nanmean(all_models_monthly, axis=0)
            mean_line, = ax.plot(x_positions, avg_vals,
                                 color='black', linewidth=2.5, linestyle='-',
                                 label='MMM', zorder=100)

            if not legend_collected:
                legend_handles.append(mean_line)
                legend_labels.append('MMM')
                legend_collected = True

            # 设置子图样式
            if r == 0:
                ax.set_title(metric_label, fontsize=18, fontweight='normal')
                # 2. 根据列索引 col 设置左侧的编号 (a), (b)
                if col == 0:
                    ax.set_title('(a)', fontsize=18, fontweight='normal', loc='left')
                elif col == 1:
                    ax.set_title('(b)', fontsize=18, fontweight='normal', loc='left')

            if col == 0:
                ax.set_ylabel(f"{method_label}", fontsize=18, fontweight='normal')

            if r == n_methods - 1:
                ax.set_xlabel("Time (months)", fontsize=18)

            ax.set_xticks(x_positions)
            ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
            ax.set_xlim(x_positions[0], x_positions[-1])
            ax.grid(True, linestyle='--', alpha=0.3)

            y_lim = c.Y_LIMS.get(metric_key, c.Y_LIMS['default'])
            if y_lim:
                ax.set_ylim(y_lim)

            ax.tick_params(axis='y', labelsize=13)
            ax.tick_params(axis='x', labelsize=13)

    # 全局图例
    ncol = 6
    total_items = len(legend_handles)
    nrows = (total_items + ncol - 1) // ncol

    reordered_handles = []
    reordered_labels = []
    empty_handle = Line2D([0], [0], visible=False, label='')

    for c_idx in range(ncol):
        for r_idx in range(nrows):
            index = r_idx * ncol + c_idx
            if index < total_items:
                reordered_handles.append(legend_handles[index])
                reordered_labels.append(legend_labels[index])
            else:
                reordered_handles.append(empty_handle)
                reordered_labels.append('')

    fig.legend(handles=reordered_handles, labels=reordered_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               ncol=6,
               fontsize=11,
               frameon=False,
               columnspacing=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    Path(c.OUTPUT_DIR).mkdir(exist_ok=True)
    metric_str = '_vs_'.join(metric_keys)
    filename = f"monthly_climatology_compare_{metric_str}_yearly.png"
    save_path = Path(c.OUTPUT_DIR) / filename
    plt.savefig(save_path, bbox_inches='tight')

    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 首次运行设为 True 计算所有指标，后续设为 False 使用缓存
    # n_workers 参数控制并行进程数，默认10
    plot_multi_metric_comparison(force_recompute=False, n_workers=10)