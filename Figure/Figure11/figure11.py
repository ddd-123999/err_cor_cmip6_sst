import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import pickle


# ==================== 配置部分 ====================
class Config:
    # --- 缓存配置 ---
    CACHE_DIR = './cache_data_metrics'
    CACHE_FILE = 'global_metrics_mixed_cache.pkl'
    PERIODS = 1827

    # --- 模型列表 (21个) ---
    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
        'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC',
        'EC-Earth3-Veg-LR', 'EC-Earth3-veg', 'EC-Earth3', 'GFDL-CM4',
        'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    # --- 方法配置 ---
    METHODS = [
        # ('base', 'Control', 'black', '-', 3.5),
        # ('EDCDF', 'EDCDF', '#4baf73', '-', 3.0),
        # # ('linear_reg', 'Linear Reg', '#fc945d', '-', 2.0),
        # ('ConvLSTM', 'ConvLSTM', '#fc945d', '-', 3.0),
        # ('UNet', 'UNet', '#D8211C', '-', 3.0),
        # ('MambaUNet', 'MambaUNet', '#299BCF', '-', 3.0),

        ('base', 'Control', 'black', '-', 3.5),
        ('EDCDF', 'EDCDF', '#385b8c', '-', 3.0),
        # ('linear_reg', 'Linear Reg', '#fc945d', '-', 2.0),
        ('ConvLSTM', 'ConvLSTM', '#bfe4ef', '--', 3.0),
        ('UNet', 'UNet', '#dd878f', '--', 3.0),
        ('MambaUNet', 'Mamba-TempNet', '#bc2325', '-', 3.0),

    ]

    # --- 【恢复】预计算的指标文件路径 (用于读取 RMSE, MAE, PCC) ---
    METRICS_FILES = {
        'base': '../../Baseline/Base/base_metrics_all.npy',
        'EDCDF': '../../Baseline/QM/qm_edcdf_q100_metrics_all.npy',
        # 'linear_reg', 'Linear Reg', '../../Baseline/Linear_regression/lr_results_data_s3_p1/lr_metrics_s3_p1_all.npy',
        'ConvLSTM': '../../Baseline/ConvLSTM/first/convlstm_metrics_s3_p1_all.npy',
        'UNet': '../../Baseline/UNet/first/unet_metrics_s3_p1_all.npy',
        'MambaUNet': '../../Baseline/MambaUNet/first/mambaunet_metrics_s3_p1_all.npy',
    }

    # --- 【重要】原始数据路径配置 (仅用于计算 Bias) ---
    RAW_DATA_PATHS = {
        'base': {
            'pred_file': '../../Preprocessing/dataset/{model}/ssp245_test.npz',
            'pred_key': 'sst',
            'true_file': '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz',
            'true_key': 'sst'
        },
        'EDCDF': {
            'pred_file': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_corrections.npy',
            'true_file': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_trues.npy'
        },
        # 'linear_reg': {
        #     'pred_file': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_corrections.npy',
        #     'true_file': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_trues.npy'
        # },
        'ConvLSTM': {
             'pred_file': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
             'true_file': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        },
        'UNet': {
            'pred_file': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true_file': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        },
        'MambaUNet': {
            'pred_file': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true_file': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        },
    }

    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'

    # --- 要绘制的4个指标 ---
    TARGET_METRICS = [
        ('rmse', 'RMSE (°C)', False),
        ('mae', 'MAE (°C)', False),
        ('bias', 'Bias (°C)', False),
        ('pcc', 'PCC', True),
    ]

    READ_FROM_FILE_METRICS = ['rmse', 'mae', 'pcc']
    CALCULATE_FROM_RAW_METRICS = ['bias']

    # --- 图片设置 ---
    OUTPUT_DIR = './metric_figures'
    OUTPUT_FILENAME = 'metrics_curves_mixed_cache_final.png'
    DPI = 600
    FIG_WIDTH = 20
    FIG_HEIGHT = 12


# ==================== 数据加载与计算 ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.metrics_data = {}
        self.mask = None
        self._load_global_mask()

    def _load_global_mask(self):
        """加载全局掩码数据"""
        if Path(self.config.MASK_PATH).exists():
            self.mask = np.load(self.config.MASK_PATH).astype(bool)

    def _get_raw_data_pair(self, method_key, model_name):
        """获取预测值和真实值的原始 3D 数据"""
        paths = self.config.RAW_DATA_PATHS.get(method_key)
        if not paths:
            return None, None

        try:
            pred_path = paths['pred_file'].format(model=model_name)

            # 加载预测数据
            if Path(pred_path).suffix == '.npz':
                pred_data = np.load(pred_path)[paths.get('pred_key', 'sst')]
            else:
                pred_data = np.load(pred_path)

            # 加载真实数据
            true_path = paths['true_file'].format(model=model_name) if '{model}' in paths['true_file'] else paths[
                'true_file']
            if Path(true_path).suffix == '.npz':
                true_data = np.load(true_path)[paths.get('true_key', 'sst')]
            else:
                true_data = np.load(true_path)

            # 截断到相同长度
            T = min(pred_data.shape[0], true_data.shape[0], self.config.PERIODS)
            pred_data = pred_data[:T]
            true_data = true_data[:T]

            # 应用掩码
            if self.mask is not None and pred_data.shape[1:] == self.mask.shape:
                pred_data = pred_data.copy()
                true_data = true_data.copy()
                pred_data[:, ~self.mask] = np.nan
                true_data[:, ~self.mask] = np.nan

            return pred_data, true_data

        except Exception as e:
            print(f"⚠️ 加载原始数据失败 [{method_key} - {model_name}]: {e}")
            return None, None

    def _calculate_global_bias(self, pred_data, true_data):
        """计算 Bias"""
        T = min(len(pred_data), len(true_data))
        diff = pred_data[:T] - true_data[:T]

        d_flat = diff.flatten()
        valid_mask = ~np.isnan(d_flat)

        if np.sum(valid_mask) == 0:
            return np.nan

        d_flat = d_flat[valid_mask]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = np.mean(d_flat)
        return val

    def _read_metric_from_file(self, method_key, model_name):
        """从 .npy 文件中读取所有指标"""
        path = self.config.METRICS_FILES.get(method_key)
        results = {}

        key_map = {'rmse': 'rmse', 'mae': 'mae', 'pcc': 'pcc'}

        if not path or not Path(path).exists():
            return {m: np.nan for m in self.config.READ_FROM_FILE_METRICS}

        try:
            method_data = np.load(path, allow_pickle=True).item()
            model_metrics = method_data.get(model_name, {})

            for metric_key in self.config.READ_FROM_FILE_METRICS:
                search_keys = [key_map[metric_key], metric_key.upper(), f'mean_{metric_key}']
                val = np.nan
                for k in search_keys:
                    if k in model_metrics:
                        val = model_metrics[k]
                        if isinstance(val, (np.ndarray, list)):
                            val = float(val) if val else np.nan
                        break
                results[metric_key] = val
        except Exception as e:
            print(f"⚠️ 读取文件指标失败 [{method_key} - {model_name}]: {e}")
            return {m: np.nan for m in self.config.READ_FROM_FILE_METRICS}

        return results

    def load_or_calculate_all_metrics(self, force_recompute=False):
        """执行混合加载和计算逻辑,并处理缓存"""
        cache_path = Path(self.config.CACHE_DIR) / self.config.CACHE_FILE
        target_metric_keys = [m[0] for m in self.config.TARGET_METRICS]

        # 1. 尝试加载缓存
        if not force_recompute and cache_path.exists():
            print(f"💾 正在从缓存加载数据: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                required_metrics_check = all(
                    metric_key in cached_data.get(method_key, {}).get(model, {})
                    for method_key, _, _, _, _ in self.config.METHODS
                    for model in self.config.MODELS
                    for metric_key in target_metric_keys
                )

                if required_metrics_check:
                    print("✅ 缓存数据完整,已加载")
                    return cached_data

                print("⚠️ 缓存文件不完整或过期,将重新计算。")
            except Exception as e:
                print(f"❌ 缓存文件加载失败 ({e}),将重新计算。")

        # 2. 混合计算和加载
        print("💻 正在执行混合数据加载和 Bias 计算...")
        all_data_cache = {m[0]: {} for m in self.config.METHODS}
        total_tasks = len(self.config.METHODS) * len(self.config.MODELS)
        task_count = 0

        for method_key, method_label, _, _, _ in self.config.METHODS:
            for model in self.config.MODELS:
                task_count += 1
                print(f"   [{task_count}/{total_tasks}] 处理: {method_label} - {model}")

                # a) 从文件读取 RMSE, MAE, PCC
                model_metrics = self._read_metric_from_file(method_key, model)

                # b) 专门计算 Bias
                pred_data, true_data = self._get_raw_data_pair(method_key, model)

                if pred_data is not None and true_data is not None:
                    bias_val = self._calculate_global_bias(pred_data, true_data)
                    model_metrics['bias'] = bias_val
                else:
                    print(f"      ⚠️ {model} 的原始数据缺失,Bias 设为 NaN")
                    model_metrics['bias'] = np.nan

                all_data_cache[method_key][model] = model_metrics

        # 3. 保存缓存
        Path(self.config.CACHE_DIR).mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(all_data_cache, f)
        print(f"✅ 数据处理完成,已保存至缓存: {cache_path}")

        return all_data_cache


# ==================== 绘图主程序 ====================
def plot_metric_curves(force_recompute=False):
    """绘制4个子图的指标曲线"""
    cfg = Config()
    loader = DataLoader(cfg)

    # 1. 加载或计算所有数据
    data_cache = loader.load_or_calculate_all_metrics(force_recompute=force_recompute)

    print(f"\n🎨 开始绘制曲线图...")

    # 2. 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(cfg.FIG_WIDTH, cfg.FIG_HEIGHT), dpi=cfg.DPI)
    axes_flat = axes.flatten()

    legend_handles = []

    # 3. 遍历4个指标
    for idx, (metric_key, metric_label, larger_better) in enumerate(cfg.TARGET_METRICS):
        ax = axes_flat[idx]

        # a) 提取 Base 方法的数据用于排序
        base_values = []
        for model in cfg.MODELS:
            val = data_cache.get('base', {}).get(model, {}).get(metric_key, np.nan)
            base_values.append(val)

        base_values = np.array(base_values)

        # b) 获取排序索引
        if metric_key.lower() == 'bias':
            sort_values = np.abs(base_values)
            sorted_indices = np.argsort(sort_values)[::-1]
        elif larger_better:
            sorted_indices = np.argsort(base_values)
        else:
            sorted_indices = np.argsort(base_values)[::-1]

        sorted_models = [cfg.MODELS[i] for i in sorted_indices]
        x_pos = np.arange(len(sorted_models))

        # c) 绘制每个方法的曲线
        for method_key, method_label, color, linestyle, linewidth in cfg.METHODS:
            values = []
            for model in sorted_models:
                val = data_cache.get(method_key, {}).get(model, {}).get(metric_key, np.nan)
                values.append(val)

            line, = ax.plot(x_pos, values,
                            color=color,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            label=method_label,
                            alpha=0.85)

            if idx == 0:
                legend_handles.append(line)

        # d) 装饰子图
        ax.set_title(metric_label, fontsize=24, fontweight='normal', pad=10)
        # ax.set_ylabel(metric_label, fontsize=16)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=15)
        ax.set_xlim(-0.5, len(sorted_models) - 0.5)

        # ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='y', labelsize=15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if metric_key.lower() == 'pcc':
            ax.set_ylim(0.6, 1.0)
        elif metric_key.lower() == 'mae':
            ax.set_ylim(0.4, 2.0)
        elif metric_key.lower() == 'rmse':
            ax.set_ylim(0.8, 2.4)
        elif metric_key.lower() == 'bias':
            ax.set_ylim(-0.5, 1.75)

        if metric_key.lower() == 'bias':
            ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.set_yticks(np.arange(-0.5, 2.0, 0.25))

    # 4. 全局图例和保存
    legend_labels = [m[1] for m in cfg.METHODS]
    fig.legend(handles=legend_handles,
               labels=legend_labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.02),
               ncol=len(legend_handles),
               fontsize=24,
               frameon=False,
               columnspacing=1.5,
               handlelength=3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    # plt.subplots_adjust(hspace=0.25, wspace=0.10)

    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = Path(cfg.OUTPUT_DIR) / cfg.OUTPUT_FILENAME
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    # 第一次运行时设置为 True 来计算 Bias 并生成缓存
    plot_metric_curves(force_recompute=False)