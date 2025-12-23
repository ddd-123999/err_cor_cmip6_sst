import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker


# ==================== 配置部分 ====================
class Config:
    N_ROWS = 5
    N_COLS = 5
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

    MODEL_PATH_TEMPLATE = '../../Preprocessing/dataset/{model}/ssp245_test.npz'
    OBS_PATH = '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'

    EXTENT = [-180, 180, 66, 90]
    VMIN = -2
    VMAX = 14
    CMAP = 'RdYlBu_r'

    OUTPUT_DIR = './sst_figures'
    OUTPUT_FILENAME = f'sst_grid_{N_ROWS}x{N_COLS}.png'
    DPI = 300


# ==================== 【新增】辅助函数: 创建偏移色标 ====================
def create_shifted_cmap(cmap_name, vmin, vmax, center=0):
    """
    创建一个“偏移”的颜色表，使得颜色的中心点（如白色）对应数据中的 center 值。
    保持色标刻度线性均匀分布。
    """
    # 1. 计算 center 在整个范围中的相对位置 (0~1之间)
    if vmin >= center or vmax <= center:
        # 如果范围不包含 center，直接返回原色标
        return plt.get_cmap(cmap_name)

    midpoint = (center - vmin) / (vmax - vmin)

    # 2.以此位置为界，重采样原色标
    fname = plt.get_cmap(cmap_name)
    N = 256
    new_colors = []

    for i in range(N):
        val = i / (N - 1)  # 当前位置 0~1

        if val < midpoint:
            # 左半部分：映射到原色标的 0 ~ 0.5
            # val / midpoint 归一化到 0~1，再 * 0.5
            orig_idx = (val / midpoint) * 0.5
        else:
            # 右半部分：映射到原色标的 0.5 ~ 1
            # (val - midpoint) / (1 - midpoint) 归一化到 0~1，再 * 0.5 + 0.5
            orig_idx = 0.5 + ((val - midpoint) / (1 - midpoint)) * 0.5

        new_colors.append(fname(orig_idx))

    new_cmap = mcolors.LinearSegmentedColormap.from_list('shifted', new_colors)
    return new_cmap


# ==================== 数据加载器 ====================
class DataLoader:
    def __init__(self, config):
        self.config = config
        self.mask = None
        self.lon = None
        self.lat = None

    def load_base_info(self):
        if Path(self.config.MASK_PATH).exists():
            self.mask = np.load(self.config.MASK_PATH)
        else:
            self.mask = np.ones((10, 10))
        if Path(self.config.OBS_PATH).exists():
            f = np.load(self.config.OBS_PATH)
            self.lon = f['lon']
            self.lat = f['lat']

    def load_model_sst(self, model_name):
        path = self.config.MODEL_PATH_TEMPLATE.format(model=model_name)
        try:
            data = np.load(path)['sst']
            sst_mean = np.nanmean(data, axis=0)
            if self.mask.shape == sst_mean.shape:
                sst_mean[~self.mask.astype(bool)] = np.nan
            return sst_mean
        except:
            return None

    def load_obs_sst(self):
        try:
            data = np.load(self.config.OBS_PATH)['sst'][-1827:]
            sst_mean = np.nanmean(data, axis=0)
            if self.mask.shape == sst_mean.shape:
                sst_mean[~self.mask.astype(bool)] = np.nan
            return sst_mean
        except:
            return None


# ==================== 绘图主程序 ====================
def plot_sst_grid():
    conf = Config()
    loader = DataLoader(conf)
    loader.load_base_info()

    print(f"🎨 开始绘制: {conf.N_ROWS}行 x {conf.N_COLS}列 (线性偏移色标)...")

    fig, axes = plt.subplots(conf.N_ROWS, conf.N_COLS,
                             figsize=(7 * conf.N_COLS, 7 * conf.N_ROWS),
                             subplot_kw={'projection': ccrs.NorthPolarStereo()},
                             dpi=conf.DPI)

    axes_flat = axes.flatten()
    total_slots = len(axes_flat)
    model_idx = 0

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)

    # --- 【关键修改】生成偏移后的新色标 ---
    shifted_cmap = create_shifted_cmap(conf.CMAP, conf.VMIN, conf.VMAX, center=0)

    for i in range(total_slots):
        ax = axes_flat[i]
        title_text = ""
        sst_data = None

        # === 观测数据放在模型后面，而不是最后一格 ===
        if i == len(conf.MODELS):
            sst_data = loader.load_obs_sst()
            title_text = "Observation (OISST)"

        # === 模式数据 ===
        elif i < len(conf.MODELS):
            model_name = conf.MODELS[i]
            sst_data = loader.load_model_sst(model_name)
            title_text = model_name

        # === 超出部分全部隐藏 ===
        else:
            ax.set_visible(False)
            continue

        ax.set_extent(conf.EXTENT, crs=ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
        ax.coastlines(linewidth=0.5, color='black', zorder=3)

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=4)
        gl.ylocator = mticker.FixedLocator([60, 70, 80])

        lon_labels = [(0, '0°'), (60, '60°E'), (120, '120°E'), (180, '180°'), (-120, '120°W'), (-60, '60°W')]
        label_lat = 65
        for lon, label in lon_labels:
            # 计算旋转角度：为了让文字贴合圆弧
            if lon == 0 or lon == 180:
                rotation = 0
            elif abs(lon) == 60:
                rotation = lon
            elif abs(lon) == 120:
                rotation = lon + 180
            else:
                rotation = lon  # 默认逻辑

            # 微调位置 (防止60度和120度离得太近或太远)
            current_lat = label_lat
            if lon == 0:
                current_lat = label_lat - 0.4
            if lon == 60:
                current_lat = label_lat - 0.2
            if lon == 120:
                current_lat = label_lat + 0.5
            if lon == -60:
                current_lat = label_lat - 0.4
            if lon == -120:
                current_lat = label_lat - 0.3
            ax.text(lon, current_lat, label,
                    transform=ccrs.PlateCarree(),
                    ha='center', va='center',
                    rotation=rotation,
                    fontsize=18,  # 字体稍微调小一点适应小图
                    fontweight='normal',
                    color='black',
                    zorder=10)

        if sst_data is not None:
            # --- 【关键修改】使用新色标，去掉 TwoSlopeNorm ---
            im = ax.pcolormesh(loader.lon, loader.lat, sst_data,
                               transform=ccrs.PlateCarree(),
                               cmap=shifted_cmap,  # 使用新色标
                               vmin=conf.VMIN,  # 使用线性范围
                               vmax=conf.VMAX,
                               shading='auto')
            ax.set_title(title_text, fontsize=30, fontweight='bold', pad=30)

        # ============================
        # 🚀 通用版：色标自动放在最后一行空白 subplot 区域
        # ============================

        fig.canvas.draw()  # 必须调用才能获得正确的 subplot 位置

        total_used = len(conf.MODELS) + 1  # 模式数量 + 观测图
        total_slots = conf.N_ROWS * conf.N_COLS

        # 找到最后一个使用的 subplot 的 index
        last_used_index = total_used - 1

        # 所在行（从 0 开始）
        last_row = last_used_index // conf.N_COLS

        # 最后一行 subplot 的所有 index
        last_row_start = last_row * conf.N_COLS
        last_row_end = last_row_start + conf.N_COLS
        last_row_indices = list(range(last_row_start, last_row_end))

        # 找到这一行中未使用的 subplot
        empty_indices = [i for i in last_row_indices if i >= total_used]

        # 如果这一行没有空白 subplot，则退到下一行，否则保持空白区域
        if len(empty_indices) == 0:
            print("⚠ 无空白 subplot 可用于放色标，采用默认底部色标")
            cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
        else:
            empty_axes = [axes_flat[i] for i in empty_indices]

            # 自动计算色标的 bbox
            xmin = min(ax.get_position().x0 for ax in empty_axes)
            xmax = max(ax.get_position().x1 for ax in empty_axes)
            ymin = min(ax.get_position().y0 for ax in empty_axes)

            cbar_height = 0.015
            cbar_width = xmax - xmin
            cbar_left = xmin
            cbar_bottom = ymin + 0.02  # 稍微上移一点

            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

        # 画色标
        cb = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
        cb.set_label('SST (°C)', fontsize=30)
        cb.ax.tick_params(labelsize=22)

    plt.subplots_adjust(top=0.95, bottom=0.08, wspace=0.07, hspace=0.2)

    Path(conf.OUTPUT_DIR).mkdir(exist_ok=True)
    save_path = Path(conf.OUTPUT_DIR) / conf.OUTPUT_FILENAME
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {save_path}")
    # plt.show()


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plot_sst_grid()