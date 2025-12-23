import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import pickle
from matplotlib.lines import Line2D


# ==================== 配置部分 ====================
class Config:
    # --- 基础配置 ---
    START_DATE = '2020-01-01'
    PERIODS = 1827
    OUTPUT_DIR = './seasonal_metrics'
    DPI = 300

    # 【缓存配置】
    CACHE_DIR = './cache_data_seasonal'
    CACHE_FILE = 'seasonal_metrics_all_cache.pkl'  # ✅ 改名避免冲突

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
        'pcc': (-1.0, 1.0),
        'default': None
    }

    # 【关键3】要绘制的方法列表
    METHODS_TO_PLOT = [
        ('base', 'Control'),
        ('EDCDF', 'EDCDF'),
        # ('linear_reg', 'Linear Regression'),
        ('ConvLSTM', 'ConvLSTM'),
        ('UNet', 'UNet'),
        ('MambaUNet', 'MambaUNet'),
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
        """计算12个月的指标值"""
        results = []
        diff = pred_s - true_s

        for m in range(1, 13):
            mask = pred_s.index.month == m
            p_m = pred_s[mask].dropna()
            t_m = true_s[mask].dropna()
            d_m = diff[mask].dropna()

            if len(d_m) == 0:
                results.append(np.nan)
                continue

            if metric == 'rmse':
                val = np.sqrt(np.mean(d_m ** 2))
            elif metric == 'mae':
                val = np.mean(np.abs(d_m))
            elif metric == 'bias':
                val = np.mean(d_m)
            elif metric == 'pcc':
                if len(p_m) > 1:
                    val = np.corrcoef(p_m, t_m)[0, 1]
                else:
                    val = np.nan
            else:
                val = np.nan

            results.append(val)

        return results

    def load_or_calculate_all_metrics(self, force_recompute=False):
        """✅ 改进：缓存所有4个指标的数据"""
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

        # 2. ✅ 计算所有4个指标的数据
        print("💻 正在计算所有指标和月度数据（包含所有4个指标）...")
        all_data_cache = {}
        total_tasks = len(self.config.METHODS_TO_PLOT) * len(self.config.ALL_METRICS)
        task_count = 0

        for method_key, method_label in self.config.METHODS_TO_PLOT:
            for metric_key in self.config.ALL_METRICS:  # ✅ 使用 ALL_METRICS
                task_count += 1
                print(f"   [{task_count}/{total_tasks}] 计算 {method_label} - {metric_key}...")

                model_results_list = []
                for model in self.config.MODELS:
                    pred_s, true_s = self.get_data_pair(method_key, model)

                    if pred_s is not None and true_s is not None:
                        monthly_vals = self.calculate_monthly_metric(pred_s, true_s, metric_key)
                        model_results_list.append(monthly_vals)

                all_data_cache[(method_key, metric_key)] = model_results_list

        # 3. 保存缓存
        Path(self.config.CACHE_DIR).mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data_cache, f)
        print(f"✅ 数据计算完成，已保存所有4个指标至缓存: {cache_path}")

        return all_data_cache


# ==================== 绘图主程序 ====================
def plot_multi_metric_comparison(force_recompute=False):
    c = Config()
    loader = DataLoader(c)

    # ✅ 加载包含所有4个指标的缓存数据
    all_metric_data = loader.load_or_calculate_all_metrics(force_recompute=force_recompute)

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
                                 label='Multi-model Mean', zorder=100)

            if not legend_collected:
                legend_handles.append(mean_line)
                legend_labels.append('Multi-model Mean')
                legend_collected = True

            # 设置子图样式
            if r == 0:
                ax.set_title(metric_label, fontsize=15, fontweight='normal')
                # 2. 根据列索引 col 设置左侧的编号 (a), (b)
                if col == 0:
                    ax.set_title('(a)', fontsize=15, fontweight='normal', loc='left')
                elif col == 1:
                    ax.set_title('(b)', fontsize=15, fontweight='normal', loc='left')

            if col == 0:
                ax.set_ylabel(f"{method_label}", fontsize=15, fontweight='normal')

            if r == n_methods - 1:
                ax.set_xlabel("Time (months)", fontsize=12)



            ax.set_xticks(x_positions)
            ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
            ax.set_xlim(x_positions[0], x_positions[-1])
            ax.grid(True, linestyle='--', alpha=0.3)

            y_lim = c.Y_LIMS.get(metric_key, c.Y_LIMS['default'])
            if y_lim:
                ax.set_ylim(y_lim)

            ax.tick_params(axis='y', labelsize=11)
            ax.tick_params(axis='x', labelsize=11)

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
    filename = f"monthly_climatology_compare_{metric_str}.png"
    save_path = Path(c.OUTPUT_DIR) / filename
    plt.savefig(save_path, bbox_inches='tight')

    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 首次运行设为 True 计算所有指标，后续设为 False 使用缓存
    plot_multi_metric_comparison(force_recompute=False)
