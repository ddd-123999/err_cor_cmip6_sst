import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import pickle


# ==================== 配置部分 ====================
class Config:
    # 1. 基础配置
    DPI = 300
    OUTPUT_DIR = './combined_figures'
    OUTPUT_FILENAME = 'seasonal_and_regional_comparison.png'

    # 【新增】缓存配置
    CACHE_DIR = './cache_data'
    CACHE_FILE = 'figure7_cache.pkl'

    # 2. 只有原始 CMIP6 和 观测
    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
        'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
        'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
        'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    # 3. 海域映射
    REGION_MAP = {
        'CA': 'Central Arctic',
        'CS': 'Chukchi Sea',
        'ESS': 'East Siberian Sea',
        'LS': 'Laptev Sea',
        'KS': 'Kara Sea',
        'BaS': 'Barents Sea',
        'EGS': 'East Greenland Sea',
        'BB': 'Baffin Bay',
        'CAA': 'Canadian Archipelago',
        'BS': 'Beaufort Sea'
    }
    REGIONS = list(REGION_MAP.values())
    REGION_KEYS = list(REGION_MAP.keys())

    # 4. 路径配置
    REGION_MASK_DIR = 'Region_Masks'
    OBS_PATH = '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
    CMIP6_PATH_TEMPLATE = '../../Preprocessing/dataset/{model}/ssp245_test.npz'

    # 5. 时间配置
    START_DATE = '2020-01-01'
    PERIODS = 1827


# ==================== 数据加载类 (带缓存) ====================
class DataLoader:
    def __init__(self, config):
        self.c = config
        self.region_masks = {}
        self.time_index = pd.date_range(start=config.START_DATE, periods=config.PERIODS, freq='D')
        self.data_shape = None

        # 缓存路径
        Path(config.CACHE_DIR).mkdir(exist_ok=True)
        self.cache_path = Path(config.CACHE_DIR) / config.CACHE_FILE

    def resize_mask(self, mask, target_shape):
        """调整掩码尺寸以匹配数据"""
        if mask.shape == target_shape:
            return mask
        r_idx = np.linspace(0, mask.shape[0] - 1, target_shape[0]).astype(int)
        c_idx = np.linspace(0, mask.shape[1] - 1, target_shape[1]).astype(int)
        return mask[r_idx[:, None], c_idx]

    def load_masks(self, target_shape):
        """加载所有分海域掩码"""
        self.data_shape = target_shape
        abs_path = os.path.abspath(self.c.REGION_MASK_DIR)

        print(f"📂 加载海域掩码...")
        for abbr, full_name in self.c.REGION_MAP.items():
            path = os.path.join(abs_path, f"mask_{abbr}.npy")
            if os.path.exists(path):
                m = np.load(path)
                if m.shape != target_shape:
                    m = self.resize_mask(m, target_shape)
                self.region_masks[full_name] = m.astype(bool)
            else:
                print(f"   ⚠️ 警告: 找不到 {full_name} 的掩码文件 ({path})")
                self.region_masks[full_name] = np.zeros(target_shape, dtype=bool)

    def process_data(self, data):
        """
        输入: (Time, Lat, Lon)
        输出:
           1. monthly_clim: (12,) 1-12月的区域平均值
           2. regional_means: (10,) 10个海域的多年平均值
        """
        if not self.region_masks:
            self.load_masks(data.shape[1:])

        # 全海域月平均
        global_mean_series = pd.Series(np.nanmean(data, axis=(1, 2)), index=self.time_index[:len(data)])
        monthly_clim = global_mean_series.groupby(global_mean_series.index.month).mean()

        # 分海域多年平均
        regional_vals = []
        for region_name in self.c.REGIONS:
            mask = self.region_masks.get(region_name)
            if mask is not None and np.any(mask):
                masked_data = data[:, mask]
                spatial_mean = np.nanmean(masked_data, axis=1)
                total_mean = np.nanmean(spatial_mean)
                regional_vals.append(total_mean)
            else:
                regional_vals.append(np.nan)

        return monthly_clim, regional_vals

    def compute_all_data(self):
        """计算所有模型和观测的数据"""
        print("🔄 开始计算所有数据...")
        all_data = {'models': {}, 'obs': None}

        # 处理观测数据
        print("   处理观测数据...")
        try:
            obs_raw = np.load(self.c.OBS_PATH)['sst'][-self.c.PERIODS:]
            all_data['obs'] = self.process_data(obs_raw)
        except Exception as e:
            print(f"   ❌ 观测数据处理失败: {e}")

        # 处理所有CMIP6模型
        for i, model in enumerate(self.c.MODELS, 1):
            print(f"   [{i}/{len(self.c.MODELS)}] 处理: {model}")
            path = self.c.CMIP6_PATH_TEMPLATE.format(model=model)
            try:
                if os.path.exists(path):
                    data = np.load(path)['sst']
                    if len(data) > self.c.PERIODS:
                        data = data[:self.c.PERIODS]
                    all_data['models'][model] = self.process_data(data)
            except Exception as e:
                print(f"      ⚠️ 跳过: {e}")

        return all_data

    def load_or_compute_data(self, force_recompute=False):
        """加载缓存或重新计算"""
        if not force_recompute and self.cache_path.exists():
            print(f"📦 发现缓存文件，正在加载: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"   ✅ 缓存加载成功！")
                return data
            except Exception as e:
                print(f"   ⚠️ 缓存加载失败: {e}，将重新计算")

        # 重新计算
        data = self.compute_all_data()

        # 保存缓存
        print(f"\n💾 保存缓存到: {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"   ✅ 缓存保存成功！")

        return data


# ==================== 绘图主程序 ====================
def plot_combined_figure(force_recompute=False):
    """
    绘制组合图

    Args:
        force_recompute: 是否强制重新计算数据
    """
    loader = DataLoader(Config)

    # 加载或计算数据
    all_data = loader.load_or_compute_data(force_recompute=force_recompute)

    print("\n🎨 开始绘制图表...")

    # 准备绘图
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7), dpi=Config.DPI)

    # 定义颜色
    cmap = plt.cm.jet
    norm = mcolors.Normalize(vmin=0, vmax=len(Config.MODELS) - 1)
    model_colors = [cmap(norm(i)) for i in range(len(Config.MODELS))]

    # 绘制模型数据
    for i, model in enumerate(Config.MODELS):
        model_data = all_data['models'].get(model)
        if model_data:
            monthly_clim, regional_means = model_data

            # 左图
            ax_left.plot(monthly_clim.index, monthly_clim.values,
                         color=model_colors[i], linewidth=1.5, alpha=0.7)

            # 右图
            ax_right.plot(range(len(regional_means)), regional_means,
                          color=model_colors[i], linewidth=1.5, alpha=0.7)

    # 绘制观测数据
    if all_data['obs']:
        obs_monthly, obs_regional = all_data['obs']

        ax_left.plot(obs_monthly.index, obs_monthly.values,
                     color='black', linewidth=3, zorder=10, label='Observation')

        ax_right.plot(range(len(obs_regional)), obs_regional,
                      color='black', linewidth=3, zorder=10)

    # 设置样式 (保持原有样式代码)
    ax_left.set_title("(a)", fontsize=20, fontweight='normal', loc='left')
    ax_left.set_ylabel("SST (°C)", fontsize=20)
    ax_left.set_xlabel("Time (months)", fontsize=20)
    ax_left.set_xticks(range(1, 13))
    ax_left.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], fontsize=14)
    ax_left.grid(True, linestyle='--', alpha=0.3)
    ax_left.set_xlim(1, 12)
    ax_left.set_ylim(-2, 5)
    ax_left.tick_params(axis='y', labelsize=12)

    ax_right.set_title("(b)", fontsize=20, fontweight='normal', loc='left')
    ax_right.set_xlabel("Region", fontsize=20)
    ax_right.set_xticks(range(len(Config.REGION_KEYS)))
    ax_right.set_xticklabels(Config.REGION_KEYS, fontsize=14)
    ax_right.grid(True, linestyle='--', alpha=0.3)
    ax_right.set_ylim(-2, 5)
    ax_right.tick_params(axis='y', labelsize=12)

    # 图例 (保持原有图例代码)
    handles = []
    for i, model in enumerate(Config.MODELS):
        line = Line2D([0], [0], color=model_colors[i], linewidth=2, label=model)
        handles.append(line)
    obs_handle = Line2D([0], [0], color='black', linewidth=3, label='Observation')
    handles.append(obs_handle)

    ncol = 7
    total_items = len(handles)
    nrows = (total_items + ncol - 1) // ncol
    reordered_handles = []
    empty_handle = Line2D([0], [0], visible=False, label='')

    for c in range(ncol):
        for r in range(nrows):
            index = r * ncol + c
            if index < total_items:
                reordered_handles.append(handles[index])
            else:
                reordered_handles.append(empty_handle)

    fig.legend(handles=reordered_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.0),
               ncol=7,
               fontsize=12,
               frameon=False,
               columnspacing=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.82])

    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = Path(Config.OUTPUT_DIR) / Config.OUTPUT_FILENAME
    plt.savefig(save_path)
    print(f"\n✅ 图片已保存: {save_path}")


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    # 第一次运行设置为 True，后续设置为 False
    plot_combined_figure(force_recompute=False)