import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pickle
import warnings

warnings.filterwarnings('ignore')


# ==================== 配置部分 ====================
class Config:
    # 模型列表
    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
        'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
        'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
        'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    # 缓存路径
    TIMESERIES_CACHE = './trend_cache_yearly/timeseries_cache_yearly.pkl'
    SPATIAL_CACHE = './trend_cache_yearly/spatial_trend_cache_yearly.pkl'
    OBS_PATH = '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'

    # 时间配置
    START_YEAR = 2025
    END_YEAR = 2100

    # 输出配置
    OUTPUT_DIR = './sst_figures'
    OUTPUT_FILENAME = 'sst_combined_trend_spatial_final.png'
    DPI = 600

    # 空间图配置
    EXTENT = [-180, 180, 66, 90]

    # 自定义10个颜色(从浅到深)
    # CUSTOM_COLORS = [
    #     '#C8FAFF', '#FFFFEB', '#FFF1BD', '#FFD79A', '#FFAE76',
    #     '#FF7955', '#FF3A3A', '#F72232', '#D90D2B', '#A6001B'
    # ]
    CUSTOM_COLORS = [
        '#FFFFEB', '#FFF1BD', '#FFD79A', '#FFAE76','#FF7955',
        '#FF3A3A', '#F72232', '#D90D2B', '#A6001B','#700012'
    ]

    # 离散色标边界(0.05步长)
    TREND_BOUNDS = np.arange(0, 0.55, 0.05)

    # 颜色配置
    INDIVIDUAL_COLOR = '#D3D3D3'  # 浅灰色
    ORIGINAL_MMM_COLOR = '#539DCC'  # Control MMM 蓝色
    CORRECTED_MMM_COLOR = '#CE4459'  # MambaUNet MMM 红色
    ALPHA_SHADE = 0.2

    @property
    def TREND_CMAP(self):
        """返回自定义的离散颜色映射"""
        return mcolors.ListedColormap(self.CUSTOM_COLORS)

    @property
    def TREND_NORM(self):
        """返回自定义的边界标准化器"""
        return mcolors.BoundaryNorm(self.TREND_BOUNDS, len(self.CUSTOM_COLORS))


# ==================== 数据加载器 ====================
class DataLoader:
    """加载缓存数据"""

    def __init__(self, config):
        self.config = config
        self.lon = None
        self.lat = None
        self.load_geo_info()

    def load_geo_info(self):
        """加载经纬度信息"""
        if Path(self.config.OBS_PATH).exists():
            f = np.load(self.config.OBS_PATH)
            self.lon = f['lon']
            self.lat = f['lat']
            print(f"✅ 加载经纬度: lon{self.lon.shape}, lat{self.lat.shape}")

    def load_timeseries_cache(self):
        """加载时间序列缓存"""
        cache_path = Path(self.config.TIMESERIES_CACHE)
        if not cache_path.exists():
            raise FileNotFoundError(f"❌ 时间序列缓存不存在: {cache_path}")

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 加载时间序列缓存: {len(data)} 个模型")
        return data

    def load_spatial_cache(self):
        """加载空间趋势缓存"""
        cache_path = Path(self.config.SPATIAL_CACHE)
        if not cache_path.exists():
            raise FileNotFoundError(f"❌ 空间趋势缓存不存在: {cache_path}")

        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ 加载空间趋势缓存")
        return data


# ==================== 稀疏显著性点函数 ====================
def sparse_points_by_distance_projection(lon_array, lat_array, values_array,
                                         projection, distance_km=200):
    """
    在投影坐标系下按距离稀疏点

    参数:
        lon_array, lat_array: 经纬度数组
        values_array: 对应的值数组
        projection: cartopy投影对象（如ccrs.NorthPolarStereo()）
        distance_km: 稀疏距离（公里）
    """
    # 1. 将经纬度转换为投影坐标（单位：米）
    import pyproj
    from scipy.spatial import KDTree
    import numpy as np

    # 获取投影的坐标参考系统
    proj_crs = projection.proj4_params

    # 创建转换器：地理坐标(经纬度) -> 投影坐标
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84 经纬度
        proj_crs,  # 投影坐标系
        always_xy=True
    )

    # 转换坐标
    x_coords, y_coords = transformer.transform(lon_array, lat_array)

    # 将公里转换为米
    distance_m = distance_km * 1000

    # 2. 在投影坐标下进行稀疏（使用KDTree加速）
    coords_2d = np.column_stack([x_coords, y_coords])

    n = len(lon_array)
    selected_indices = []

    # 使用KDTree进行快速距离查询
    tree = KDTree(coords_2d)

    for i in range(n):
        if i in selected_indices:
            continue

        # 找到距离当前点小于阈值的所有点
        indices = tree.query_ball_point(coords_2d[i], distance_m)

        # 只保留当前点（删除周围的点）
        # 从列表中去掉已经被选择的点
        if not any(idx in selected_indices for idx in indices if idx != i):
            selected_indices.append(i)

    # 返回稀疏后的数据
    if selected_indices:
        selected_indices = np.array(selected_indices)
        sparse_lon = lon_array[selected_indices]
        sparse_lat = lat_array[selected_indices]
        sparse_values = values_array[selected_indices]

        return np.column_stack([sparse_lon, sparse_lat, sparse_values])
    else:
        return np.array([])


def sparse_significant_points_projection(lon_grid, lat_grid, trend_data, p_data,
                                         projection,
                                         distance_km=200, p_threshold=0.05,
                                         trend_threshold=0.25):
    """在投影坐标下提取并稀疏显著点"""
    sig_mask = (p_data < p_threshold) & (np.abs(trend_data) > trend_threshold)

    if not np.any(sig_mask):
        return np.array([])

    # 提取显著点的经纬度和趋势值
    sig_lons = lon_grid[sig_mask].flatten()
    sig_lats = lat_grid[sig_mask].flatten()
    sig_trends = trend_data[sig_mask].flatten()

    # 在投影坐标下稀疏
    sparse_data = sparse_points_by_distance_projection(
        sig_lons, sig_lats, sig_trends,
        projection, distance_km
    )

    if len(sparse_data) > 0:
        print(f"   投影稀疏处理: 从 {len(sig_lons)} 个显著点中选择了 {len(sparse_data)} 个点")
        return sparse_data[:, :2]  # 只返回经纬度
    else:
        return np.array([])


# ==================== 主绘图函数 ====================
def plot_combined_figure():
    """绘制组合图: 左侧线性趋势 + 右侧空间分布"""

    config = Config()
    loader = DataLoader(config)

    print("=" * 60)
    print("🚀 开始绘制组合图")
    print("=" * 60)

    # 加载缓存数据
    timeseries_data = loader.load_timeseries_cache()
    spatial_data = loader.load_spatial_cache()

    # 提取空间趋势数据
    orig_trend = spatial_data['original_trend']
    orig_p = spatial_data['original_p']
    corr_trend = spatial_data['corrected_trend']
    corr_p = spatial_data['corrected_p']

    # 提取时间序列数据
    original_ts_list = []
    corrected_ts_list = []

    for model_name in config.MODELS:
        if model_name in timeseries_data:
            model_data = timeseries_data[model_name]
            if 'original' in model_data:
                original_ts_list.append(model_data['original']['time_series'])
            if 'corrected' in model_data:
                corrected_ts_list.append(model_data['corrected']['time_series'])

    # 转换为数组
    original_ts_array = np.array(original_ts_list)  # [n_models, n_years]
    corrected_ts_array = np.array(corrected_ts_list)

    # 计算多模式平均和标准差
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        original_mmm = np.nanmean(original_ts_array, axis=0)
        original_std = np.nanstd(original_ts_array, axis=0)
        corrected_mmm = np.nanmean(corrected_ts_array, axis=0)
        corrected_std = np.nanstd(corrected_ts_array, axis=0)

    # 生成年份数组
    n_years = len(original_mmm)
    years = np.arange(config.START_YEAR, config.START_YEAR + n_years)

    # ========== 计算线性趋势斜率 (℃/decade) ==========
    def calculate_trend_slope(years, values):
        """计算线性趋势斜率，返回 ℃/decade"""
        # 移除NaN值
        mask = ~np.isnan(values)
        if np.sum(mask) < 2:
            return np.nan

        y = values[mask]
        x = years[mask]

        # 线性回归
        coeff = np.polyfit(x, y, 1)
        slope_per_year = coeff[0]  # ℃/year
        slope_per_decade = slope_per_year * 10  # ℃/decade

        return slope_per_decade

    # 计算斜率
    original_slope = calculate_trend_slope(years, original_mmm)
    corrected_slope = calculate_trend_slope(years, corrected_mmm)

    print(f"📊 数据统计:")
    print(f"   原始数据: {len(original_ts_list)} 个模型, {n_years} 年")
    print(f"   校正数据: {len(corrected_ts_list)} 个模型, {n_years} 年")
    print(f"   原始MMM趋势斜率: {original_slope:.3f} ℃/decade")
    print(f"   校正MMM趋势斜率: {corrected_slope:.3f} ℃/decade")

    # 计算全局Y轴范围
    all_values = np.concatenate([
        original_ts_array.flatten(),
        corrected_ts_array.flatten(),
        original_mmm,
        corrected_mmm
    ])
    all_values = all_values[~np.isnan(all_values)]

    y_min = np.min(all_values) - 0.5  # 留点边距
    y_max = np.max(all_values) + 0.5
    # ========== 创建图形 ==========
    fig = plt.figure(figsize=(13, 10), dpi=config.DPI)

    # 创建网格布局: 2行2列
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 0.8], height_ratios=[1, 1],
                          hspace=0.12, wspace=0.1,
                          left=0.08, right=0.92, top=0.95, bottom=0.08)

    # ========== 子图1: Control 线性趋势 (左上) ==========
    ax_ts_orig = fig.add_subplot(gs[0, 0])

    # 绘制21条个体模型线(浅灰色)
    for i in range(len(original_ts_list)):
        ax_ts_orig.plot(years, original_ts_array[i],
                        color=config.INDIVIDUAL_COLOR,
                        linewidth=1.0, alpha=0.5, zorder=1)

    # 绘制多模式平均线
    ax_ts_orig.plot(years, original_mmm,
                    color=config.ORIGINAL_MMM_COLOR,
                    linewidth=2.5, zorder=3, label='MMM')

    # 绘制标准差阴影
    ax_ts_orig.fill_between(years,
                            original_mmm - original_std,
                            original_mmm + original_std,
                            color=config.ORIGINAL_MMM_COLOR,
                            alpha=config.ALPHA_SHADE, zorder=2)

    # 添加一条invisible线用于图例
    ax_ts_orig.plot([], [], color=config.INDIVIDUAL_COLOR,
                    linewidth=2.0, label='Individual models')

    # ========== 添加趋势斜率文本标注 ==========
    slope_text_orig = f'linear trend = {original_slope:.3f} °C/decade'
    ax_ts_orig.text(0.02, 0.83, slope_text_orig,
                    transform=ax_ts_orig.transAxes,
                    fontsize=16, fontweight='normal',
                    verticalalignment='top',
                    color=config.ORIGINAL_MMM_COLOR)

    ax_ts_orig.set_ylabel('SST (°C)', fontsize=18)
    ax_ts_orig.set_title('(a)', fontsize=18, fontweight='normal', loc='left')
    ax_ts_orig.text(0.98, 0.98, 'Control',
                    transform=ax_ts_orig.transAxes,
                    fontsize=18, fontweight='normal',
                    ha='right', va='top')
    ax_ts_orig.set_xlim(config.START_YEAR, config.END_YEAR)
    ax_ts_orig.set_ylim(y_min, y_max)
    ax_ts_orig.tick_params(labelbottom=False)
    ax_ts_orig.grid(True, alpha=0.25, linestyle='--')
    ax_ts_orig.legend(loc='upper left', fontsize=13, frameon=False)
    ax_ts_orig.tick_params(axis='both', labelsize=12)

    # ========== 子图2: Control 空间分布 (右上) ==========
    ax_spatial_orig = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
    ax_spatial_orig.set_extent(config.EXTENT, crs=ccrs.PlateCarree())

    # 设置圆形边界
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * radius + center)
    ax_spatial_orig.set_boundary(circle, transform=ax_spatial_orig.transAxes)
    # ax_spatial_orig.text(0.5, 0.5, 'control',
    #                      transform=ax_spatial_orig.transAxes,
    #                      fontsize=14, fontweight='normal',
    #                      ha='center', va='center',
    #                      color=config.ORIGINAL_MMM_COLOR)
    # 绘制趋势
    if orig_trend is not None:
        im_orig = ax_spatial_orig.pcolormesh(
            loader.lon, loader.lat, orig_trend,
            transform=ccrs.PlateCarree(),
            cmap=config.TREND_CMAP,
            norm=config.TREND_NORM,
            shading='auto', zorder=1
        )

        # 添加显著性点
        if orig_p is not None:
            lon_grid, lat_grid = np.meshgrid(loader.lon, loader.lat, indexing='ij')

            # 使用投影坐标稀疏
            sparse_points = sparse_significant_points_projection(
                lon_grid, lat_grid, orig_trend.T, orig_p.T,
                projection=ccrs.NorthPolarStereo(),
                distance_km=200, p_threshold=0.05, trend_threshold=0.0
            )

            if len(sparse_points) > 0:
                ax_spatial_orig.scatter(
                    sparse_points[:, 0], sparse_points[:, 1],
                    s=15.0, c='black', alpha=0.8,
                    transform=ccrs.PlateCarree(), zorder=10,
                    marker='.', edgecolors='none'
                )

    ax_spatial_orig.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax_spatial_orig.coastlines(linewidth=0.5, color='black', zorder=3)
    gl_orig = ax_spatial_orig.gridlines(draw_labels=False, linewidth=0.5,
                                        color='gray', alpha=0.5, linestyle='--', zorder=4)
    gl_orig.ylocator = mticker.FixedLocator([60, 70, 80])
    ax_spatial_orig.set_title('(c)                   Control', fontsize=18, fontweight='normal', loc='left')

    # ========== 子图3: MambaUNet 线性趋势 (左下) ==========
    ax_ts_corr = fig.add_subplot(gs[1, 0], sharex=ax_ts_orig)

    # 绘制21条个体模型线(浅灰色)
    for i in range(len(corrected_ts_list)):
        ax_ts_corr.plot(years, corrected_ts_array[i],
                        color=config.INDIVIDUAL_COLOR,
                        linewidth=1.0, alpha=0.5, zorder=1)

    # 绘制多模式平均线
    ax_ts_corr.plot(years, corrected_mmm,
                    color=config.CORRECTED_MMM_COLOR,
                    linewidth=2.5, zorder=3, label='MMM')

    # 绘制标准差阴影
    ax_ts_corr.fill_between(years,
                            corrected_mmm - corrected_std,
                            corrected_mmm + corrected_std,
                            color=config.CORRECTED_MMM_COLOR,
                            alpha=config.ALPHA_SHADE, zorder=2)

    # 添加invisible线用于图例
    ax_ts_corr.plot([], [], color=config.INDIVIDUAL_COLOR,
                    linewidth=2.0, label='Individual models')

    # ========== 添加趋势斜率文本标注 ==========
    slope_text_corr = f'linear trend = {corrected_slope:.3f} °C/decade'
    ax_ts_corr.text(0.02, 0.83, slope_text_corr,
                    transform=ax_ts_corr.transAxes,
                    fontsize=16, fontweight='normal',
                    verticalalignment='top',
                    color=config.CORRECTED_MMM_COLOR)

    ax_ts_corr.set_xlabel('Time (years)', fontsize=18, fontweight='normal')
    ax_ts_corr.set_ylabel('SST (°C)', fontsize=18, fontweight='normal')
    ax_ts_corr.set_title('(b)', fontsize=18, fontweight='normal', loc='left')
    ax_ts_corr.text(0.98, 0.98, 'Mamba-TempNet',
                    transform=ax_ts_corr.transAxes,
                    fontsize=18, fontweight='normal',
                    ha='right', va='top')
    ax_ts_corr.set_xlim(config.START_YEAR, config.END_YEAR)
    ax_ts_corr.set_ylim(y_min, y_max)
    ax_ts_corr.grid(True, alpha=0.25, linestyle='--')
    ax_ts_corr.legend(loc='upper left', fontsize=13, frameon=False)
    ax_ts_corr.tick_params(axis='both', labelsize=12)

    # ========== 子图4: MambaUNet 空间分布 (右下) ==========
    ax_spatial_corr = fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo())
    ax_spatial_corr.set_extent(config.EXTENT, crs=ccrs.PlateCarree())
    ax_spatial_corr.set_boundary(circle, transform=ax_spatial_corr.transAxes)
    # ax_spatial_corr.text(0.5, 0.5, 'Mamba-TempNet',
    #                      transform=ax_spatial_corr.transAxes,
    #                      fontsize=14, fontweight='normal',
    #                      ha='center', va='center',
    #                      color=config.CORRECTED_MMM_COLOR)
    # 绘制趋势
    if corr_trend is not None:
        im_corr = ax_spatial_corr.pcolormesh(
            loader.lon, loader.lat, corr_trend,
            transform=ccrs.PlateCarree(),
            cmap=config.TREND_CMAP,
            norm=config.TREND_NORM,
            shading='auto', zorder=1
        )

        # 添加显著性点
        if corr_p is not None:
            lon_grid, lat_grid = np.meshgrid(loader.lon, loader.lat, indexing='ij')

            # 使用投影坐标稀疏
            sparse_points = sparse_significant_points_projection(
                lon_grid, lat_grid, corr_trend.T, corr_p.T,
                projection=ccrs.NorthPolarStereo(),
                distance_km=200, p_threshold=0.05, trend_threshold=0.0
            )

            if len(sparse_points) > 0:
                ax_spatial_corr.scatter(
                    sparse_points[:, 0], sparse_points[:, 1],
                    s=15.0, c='black', alpha=0.8,
                    transform=ccrs.PlateCarree(), zorder=10,
                    marker='.', edgecolors='none'
                )

    ax_spatial_corr.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax_spatial_corr.coastlines(linewidth=0.5, color='black', zorder=3)
    gl_corr = ax_spatial_corr.gridlines(draw_labels=False, linewidth=0.5,
                                        color='gray', alpha=0.5, linestyle='--', zorder=4)
    gl_corr.ylocator = mticker.FixedLocator([60, 70, 80])
    ax_spatial_corr.set_title('(d)          Mamaba-TempNet', fontsize=18, fontweight='normal', loc='left')

    # ========== 添加色标 (竖向,放在右侧空间图的右边) ==========
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])  # [left, bottom, width, height]

    cbar = fig.colorbar(im_corr if corr_trend is not None else im_orig,
                        cax=cbar_ax, orientation='vertical', extend='neither')
    cbar.set_label('SST trend (°C/decade)', fontsize=18, fontweight='normal')
    cbar.ax.tick_params(labelsize=11)

    # ========== 保存图片 ==========
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True, parents=True)
    save_path = Path(config.OUTPUT_DIR) / config.OUTPUT_FILENAME
    plt.savefig(save_path, bbox_inches='tight', dpi=config.DPI)

    print(f"\n✅ 图片已保存: {save_path}")
    print("=" * 60)
    # plt.show()

# ==================== 主程序 ====================
if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    print("\n🎨 绘制 CMIP6 线性趋势 + 空间分布组合图")
    print("📂 数据来源: trend_cache_yearly/ 缓存文件")
    print("-" * 60)

    try:
        plot_combined_figure()
        print("\n🎉 绘图完成!")
    except Exception as e:
        print(f"\n❌ 绘图失败: {e}")
        import traceback

        traceback.print_exc()