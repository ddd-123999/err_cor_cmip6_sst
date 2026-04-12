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
from PIL import Image


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

    # GMT地图路径
    GMT_MAP_PATH = 'Arctic_Regions_GMT.png'

    EXTENT = [-180, 180, 66, 90]
    VMIN = -2
    VMAX = 14
    CMAP = 'RdYlBu_r'

    OUTPUT_DIR = './sst_figures'
    OUTPUT_FILENAME = f'sst_grid_{N_ROWS}x{N_COLS}_with_gmt.png'
    DPI = 300


# ==================== 【新增】辅助函数: 创建偏移色标 ====================
def create_shifted_cmap(cmap_name, vmin, vmax, center=0):
    """
    创建一个"偏移"的颜色表，使得颜色的中心点(如白色)对应数据中的 center 值。
    保持色标刻度线性均匀分布。
    """
    if vmin >= center or vmax <= center:
        return plt.get_cmap(cmap_name)

    midpoint = (center - vmin) / (vmax - vmin)
    fname = plt.get_cmap(cmap_name)
    N = 256
    new_colors = []

    for i in range(N):
        val = i / (N - 1)
        if val < midpoint:
            orig_idx = (val / midpoint) * 0.5
        else:
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

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)

    # --- 【关键修改】生成偏移后的新色标 ---
    shifted_cmap = create_shifted_cmap(conf.CMAP, conf.VMIN, conf.VMAX, center=0)

    for i in range(total_slots):
        ax = axes_flat[i]
        title_text = ""
        sst_data = None
        is_gmt_map = False

        # ===============================
        # 固定 5×5 布局下的内容分配
        # ===============================

        # OISST 固定在最后一行第 2 个位置（index=21）
        if i == 21:
            sst_data = loader.load_obs_sst()
            title_text = "Observation (OISST)"

        # GMT 海域图固定在最后一个位置（index=24）
        elif i == 24:
            is_gmt_map = True
            title_text = "Arctic Regions"

        # CMIP6 模式：顺序填充前 21 个位置（0–20）
        elif i < len(conf.MODELS):
            model_name = conf.MODELS[i]
            sst_data = loader.load_model_sst(model_name)
            title_text = model_name

        # 色标位置（22, 23）先占位，不画任何东西
        elif i in [22, 23, 20]:
            ax.set_visible(False)
            continue

        # 其余位置隐藏
        else:
            ax.set_visible(False)
            continue

        # 处理GMT地图的特殊情况
        if is_gmt_map:
            # 移除投影，创建普通坐标轴用于显示图片
            ax.remove()
            ax = fig.add_subplot(conf.N_ROWS, conf.N_COLS, i + 1)

            try:
                # 读取GMT地图
                gmt_img = Image.open(conf.GMT_MAP_PATH)
                ax.imshow(gmt_img)
                ax.axis('off')
                # ax.set_title(title_text, fontsize=30, fontweight='bold', pad=30)
                print(f"   ✅ 成功插入GMT地图: {conf.GMT_MAP_PATH}")
            except FileNotFoundError:
                print(f"   ⚠️ 找不到GMT地图文件: {conf.GMT_MAP_PATH}")
                print(f"   请先运行 create.bat 生成地图")
                ax.text(0.5, 0.5, 'GMT Map\nNot Found',
                        ha='center', va='center', fontsize=20, color='red')
                ax.axis('off')
            except Exception as e:
                print(f"   ❌ 插入GMT地图失败: {e}")
                ax.text(0.5, 0.5, 'Error Loading\nGMT Map',
                        ha='center', va='center', fontsize=20, color='red')
                ax.axis('off')

            continue

        # 常规地图设置（SST数据）
        ax.set_extent(conf.EXTENT, crs=ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)
        ax.add_feature(cfeature.LAND, facecolor='gray', zorder=2)
        ax.coastlines(linewidth=0.5, color='black', zorder=3)

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=4)
        gl.ylocator = mticker.FixedLocator([60, 70, 80])

        lon_labels = [(0, '0°'), (60, '60°E'), (120, '120°E'), (180, '180°'), (-120, '120°W'), (-60, '60°W')]
        label_lat = 65
        for lon, label in lon_labels:
            if lon == 0 or lon == 180:
                rotation = 0
            elif abs(lon) == 60:
                rotation = lon
            elif abs(lon) == 120:
                rotation = lon + 180
            else:
                rotation = lon

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
                    fontsize=18,
                    fontweight='normal',
                    color='black',
                    zorder=10)

        if sst_data is not None:
            # --- 【关键修改】使用新色标，去掉 TwoSlopeNorm ---
            im = ax.pcolormesh(loader.lon, loader.lat, sst_data,
                               transform=ccrs.PlateCarree(),
                               cmap=shifted_cmap,
                               vmin=conf.VMIN,
                               vmax=conf.VMAX,
                               shading='auto')
            ax.set_title(title_text, fontsize=30, fontweight='bold', pad=30)

    # ============================
    # 🚀 色标放在倒数第二行的空白位置
    # ============================
    fig.canvas.draw()

    total_used = len(conf.MODELS) + 2  # 模式数量 + 观测图 + GMT地图
    total_slots = conf.N_ROWS * conf.N_COLS

    # 找到最后一个使用的 subplot 的 index
    last_used_index = total_used - 1

    # 所在行（从 0 开始）
    last_row = last_used_index // conf.N_COLS

    # 倒数第二行 subplot 的所有 index
    second_last_row = last_row - 1
    second_last_row_start = second_last_row * conf.N_COLS
    second_last_row_end = second_last_row_start + conf.N_COLS
    second_last_row_indices = list(range(second_last_row_start, second_last_row_end))

    # 找到这一行中未使用的 subplot（OISST后面的两个位置）
    # OISST的索引是 len(conf.MODELS)，在倒数第二行
    oisst_index = len(conf.MODELS)
    empty_indices = [oisst_index + 1, oisst_index + 2]  # OISST后面的两个位置

    # 确保这些索引在倒数第二行
    empty_indices = [i for i in empty_indices if i in second_last_row_indices]

    # ============================
    # 🎯 色标固定在最后一行第 3、4 个位置
    # ============================

    ax_cb1 = axes_flat[22]
    ax_cb2 = axes_flat[23]

    # 获取两个轴的位置
    pos1 = ax_cb1.get_position()
    pos2 = ax_cb2.get_position()

    # 合并成一个长色标
    cbar_left = pos1.x0
    cbar_right = pos2.x1
    cbar_width = cbar_right - cbar_left

    cbar_height = pos1.height * 0.12
    cbar_bottom = pos1.y0 + pos1.height / 2 - cbar_height / 2 - 0.05

    # 隐藏原 subplot
    ax_cb1.set_visible(False)
    ax_cb2.set_visible(False)

    # 新建色标轴
    cbar_ax = fig.add_axes([
        cbar_left,
        cbar_bottom,
        cbar_width,
        cbar_height
    ])

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