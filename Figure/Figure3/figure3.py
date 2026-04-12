import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath
from pathlib import Path
import warnings
import pickle


# ==================== 配置部分 ====================
class Config:
    """配置类:所有参数集中管理"""

    # --- ✅ 全部21个模型(用于缓存计算) ---
    ALL_MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
        'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
        'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
        'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
        'NorESM2-MM'
    ]

    # --- 🎨 本次要画的模型(随时修改这里切换) ---
    MODELS = [
        # 'ACCESS-CM2',
        # 'ACCESS-ESM1-5',
        # 'BCC-CSM2-MR',
        # 'CanESM5',
        # 'CESM2-WACCM',
        # 'CMCC-CM2-SR5',
        # 'CMCC-ESM2',

        # 'EC-Earth3-CC',
        # 'EC-Earth3-Veg-LR',
        # 'EC-Earth3-veg',
        # 'EC-Earth3',
        # 'GFDL-CM4',
        # 'GFDL-ESM4',
        # 'IPSL-CM6A-LR',

        'MIROC6',
        'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'NESM3',
        'NorESM2-LM',
        'NorESM2-MM'
    ]

    # --- ✅ 新增:缓存配置 ---
    CACHE_DIR = './cache_data_bias'
    CACHE_FILE = 'figure2_bias_cache.pkl'

    # --- 方法配置(数据路径) ---
    METHODS = [
        ('base', 'Control', {
            'cmip': '../../Preprocessing/dataset/{model}/ssp245_test.npz',
            'obs': '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
        }, True),

        ('EDCDF', 'EDCDF', {
            'pred': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_corrections.npy',
            'true': '../../Baseline/QM/qm_edcdf_q100_results_data/{model}/test_trues.npy'
        }, True),

        # ('linear_reg', 'Linear Reg', {
        #     'pred': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_corrections.npy',
        #     'true': '../../Baseline/Linear_regression/lr_results_data_s3_p1/{model}/test_trues.npy'
        # }, True),

        ('ConvLSTM', 'ConvLSTM', {
            'pred': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true': '../../Baseline/ConvLSTM/first/md-ConvLSTM_cn-{model}_bs-8_pt-10_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        }, True),

        ('UNet', 'UNet', {
            'pred': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true': '../../Baseline/UNet/first/md-UNet_cn-{model}_bs-32_pt-15_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        }, True),

        ('MambaUNet', 'Mamba-TempNet', {
            'pred': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_corrections.npy',
            'true': '../../Baseline/MambaUNet/first/md-MambaUNet_new_cn-{model}_bs-32_pt-20_sl-3_cl-1_dp-0.0_ln-mse_norm-True/test_trues.npy'
        }, True),
    ]

    # --- 指标文件路径(用于读取指标值)---
    METRICS_FILES = {
        'base': '../../Baseline/Base/base_metrics_all.npy',
        'EDCDF': '../../Baseline/QM/qm_edcdf_q100_metrics_all.npy',
        # 'linear_reg': '../../Baseline/Linear_regression/lr_results_data_s3_p1/lr_metrics_s3_p1_all.npy',
        'ConvLSTM': '../../Baseline/ConvLSTM/first/convlstm_metrics_s3_p1_all.npy',
        'UNet': '../../Baseline/UNet/first/unet_metrics_s3_p1_all.npy',
        'MambaUNet': '../../Baseline/MambaUNet/first/mambaunet_metrics_s3_p1_all.npy',
    }

    # --- 地理信息 ---
    MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'
    GEO_INFO_PATH = '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'

    # --- 图片配置 ---
    # OUTPUT_FILENAME = 'models_1-7.png'
    # OUTPUT_FILENAME = 'models_8-14.png'
    OUTPUT_FILENAME = 'models_15-21.png'  # 💡 根据MODELS列表手动修改文件名
    OUTPUT_DIR = './comparison_figures'

    # --- 色标配置 ---
    BIAS_CMAP = 'RdBu_r'
    BIAS_VMIN = -4
    BIAS_VMAX = 4

    # --- 图像质量 ---
    DPI = 600
    FIGURE_WIDTH_PER_METHOD = 4.5
    FIGURE_HEIGHT_PER_MODEL = 4.5
    PLOT_SHAPE = 'circle'


# ==================== 数据加载器 (✅ 添加缓存功能) ====================
class DataLoader:
    """数据加载和处理"""

    def __init__(self, config):
        self.config = config
        self.mask = None
        self.lon = None
        self.lat = None
        self.metrics_dict = {}

        # ✅ 缓存路径
        Path(config.CACHE_DIR).mkdir(exist_ok=True)
        self.cache_path = Path(config.CACHE_DIR) / config.CACHE_FILE

    def load_geo_info(self):
        """加载地理信息和mask"""
        print("📂 加载地理信息...")
        self.mask = np.load(self.config.MASK_PATH)
        geo_file = np.load(self.config.GEO_INFO_PATH)
        self.lon = geo_file['lon']
        self.lat = geo_file['lat']
        print(f"   ✓ Mask形状: {self.mask.shape}")
        print(f"   ✓ 经度形状: {self.lon.shape}")
        print(f"   ✓ 纬度形状: {self.lat.shape}")

    def load_metrics(self):
        """加载所有方法的指标值"""
        print("\n📂 加载指标数据...")
        for method_name, metrics_file in self.config.METRICS_FILES.items():
            if not Path(metrics_file).exists():
                print(f"   ⚠️  指标文件不存在: {metrics_file}")
                self.metrics_dict[method_name] = {}
                continue
            try:
                data = np.load(metrics_file, allow_pickle=True).item()
                self.metrics_dict[method_name] = data
                print(f"   ✓ {method_name}: 加载了 {len(data)} 个模型的指标")
            except Exception as e:
                print(f"   ❌ 加载失败 {metrics_file}: {e}")
                self.metrics_dict[method_name] = {}

    def get_metric_value(self, method_name, model_name, metric_name):
        """获取特定模型、方法、指标的值"""
        try:
            method_data = self.metrics_dict.get(method_name, {})
            model_data = method_data.get(model_name, {})

            if metric_name in model_data:
                value = model_data[metric_name]
            elif metric_name == 'mean_pcc' and 'pcc' in model_data:
                value = model_data['pcc']
            elif metric_name == 'mean_ssim' and 'ssim' in model_data:
                value = model_data['ssim']
            else:
                return np.nan

            if isinstance(value, np.ndarray):
                value = float(value)
            return value
        except Exception as e:
            return np.nan

    def _calculate_single_bias(self, method_name, model_name, method_config):
        """计算单个模型的bias数据 (内部方法)"""
        try:
            paths = method_config.copy()
            for key in paths:
                if isinstance(paths[key], str):
                    paths[key] = paths[key].format(model=model_name)

            if 'cmip' in paths and 'obs' in paths:
                cmip_data = np.load(paths['cmip'])['sst']
                obs_data = np.load(paths['obs'])['sst'][-1827:]
                min_len = min(cmip_data.shape[0], obs_data.shape[0])
                cmip_data = cmip_data[:min_len]
                obs_data = obs_data[:min_len]
                bias = cmip_data - obs_data
                bias_map = np.nanmean(bias, axis=0)

            elif 'pred' in paths and 'true' in paths:
                pred_data = np.load(paths['pred'])
                true_data = np.load(paths['true'])
                bias = pred_data - true_data
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bias_map = np.nanmean(bias, axis=0)
            else:
                return None

            bias_map[~self.mask.astype(bool)] = np.nan
            return bias_map

        except Exception as e:
            print(f"      ⚠️ 计算失败: {e}")
            return None

    def load_or_calculate_all_bias(self, force_recompute=False):
        """
        ✅ 核心方法:加载或计算所有bias数据 (针对ALL_MODELS,不是MODELS)

        Returns:
            bias_cache: {(model_name, method_name): bias_map_array}
        """
        # 1. 尝试加载缓存
        if not force_recompute and self.cache_path.exists():
            print(f"💾 发现缓存文件,正在加载: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    bias_cache = pickle.load(f)

                # ✅ 检查缓存是否包含全部21个模型
                required_keys = [(m, mt[0]) for m in self.config.ALL_MODELS
                                 for mt in self.config.METHODS]
                all_present = all(k in bias_cache for k in required_keys)

                if all_present:
                    print(f"   ✅ 缓存加载成功! (包含 {len(self.config.ALL_MODELS)} 个模型)")
                    return bias_cache
                else:
                    print(f"   ⚠️ 缓存不完整,将重新计算全部 {len(self.config.ALL_MODELS)} 个模型")
            except Exception as e:
                print(f"   ❌ 缓存加载失败 ({e}),将重新计算")

        # 2. ✅ 计算全部21个模型的bias数据
        print(f"\n💻 开始计算全部 {len(self.config.ALL_MODELS)} 个模型的bias数据...")
        print("   (一次性缓存所有模型,后续切换模型列表时无需重新计算)")
        bias_cache = {}
        total_tasks = len(self.config.ALL_MODELS) * len(self.config.METHODS)
        task_count = 0

        # 使用 ALL_MODELS 而不是 MODELS
        for model_name in self.config.ALL_MODELS:
            for method_name, display_name, method_config, _ in self.config.METHODS:
                task_count += 1
                print(f"   [{task_count}/{total_tasks}] 处理: {model_name} - {display_name}")

                bias_map = self._calculate_single_bias(method_name, model_name, method_config)
                bias_cache[(model_name, method_name)] = bias_map

        # 3. 保存缓存
        print(f"\n💾 保存缓存到: {self.cache_path}")
        with open(self.cache_path, 'wb') as f:
            pickle.dump(bias_cache, f)
        print(f"   ✅ 缓存保存成功! (包含 {len(self.config.ALL_MODELS)} 个模型)")

        return bias_cache

    def load_bias_data(self, method_name, model_name, bias_cache):
        """
        ✅ 从缓存中获取bias数据

        Args:
            method_name: 方法名
            model_name: 模型名
            bias_cache: 缓存字典

        Returns:
            bias_map: (lat, lon) 的偏差空间分布
        """
        return bias_cache.get((model_name, method_name))


# ==================== 图表生成器 ====================
class FigureGenerator:
    """生成对比图表"""

    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def create_comparison_figure(self, bias_cache):
        """创建对比图(多年平均偏差空间分布)"""
        n_models = len(self.config.MODELS)
        n_methods = len(self.config.METHODS)

        print(f"\n🎨 生成对比图 ({n_models}行 × {n_methods}列)...")

        fig_width = self.config.FIGURE_WIDTH_PER_METHOD * n_methods * 0.6
        fig_height = self.config.FIGURE_HEIGHT_PER_MODEL * n_models * 0.6

        fig, axes = plt.subplots(
            n_models, n_methods,
            figsize=(fig_width, fig_height),
            subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=0)},
            dpi=300
        )

        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)

        for i, model_name in enumerate(self.config.MODELS):
            for j, (method_name, display_name, method_config, _) in enumerate(self.config.METHODS):
                ax = axes[i, j]

                ax.set_extent([-180, 180, 66, 90], crs=ccrs.PlateCarree())

                if self.config.PLOT_SHAPE == 'circle':
                    theta = np.linspace(0, 2 * np.pi, 100)
                    center, radius = [0.5, 0.5], 0.5
                    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
                    circle = mpath.Path(verts * radius + center)
                    ax.set_boundary(circle, transform=ax.transAxes)
                    ax.spines['geo'].set_visible(True)
                    ax.spines['geo'].set_edgecolor('black')
                    ax.spines['geo'].set_linewidth(0.6)

                print(f"   处理: {model_name} - {display_name}...")

                # ✅ 从缓存中获取数据
                bias_map = self.data_loader.load_bias_data(method_name, model_name, bias_cache)

                if bias_map is not None:
                    c = ax.pcolormesh(
                        self.data_loader.lon,
                        self.data_loader.lat,
                        bias_map,
                        transform=ccrs.PlateCarree(),
                        cmap=self.config.BIAS_CMAP,
                        vmin=self.config.BIAS_VMIN,
                        vmax=self.config.BIAS_VMAX,
                        shading='auto'
                    )

                ax.add_feature(cfeature.LAND, facecolor='lightgray',
                               edgecolor='none', zorder=2)
                ax.coastlines(resolution='50m', linewidth=0.4, color='#555555', zorder=3)
                ax.gridlines(draw_labels=False, linewidth=0.4,
                             color='gray', alpha=0.5, linestyle='--', zorder=4)

                lon_labels = [(0, '0°'), (60, '60°E'), (120, '120°E'),
                              (180, '180°'), (-120, '120°W'), (-60, '60°W')]
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
                            fontsize=10,
                            fontweight='normal',
                            color='black',
                            zorder=10)

                title_text = ""
                if i == 0:
                    title_text = f"{display_name}"

                if j == 0:
                    ax.text(-0.18, 0.5, model_name,
                            transform=ax.transAxes,
                            fontsize=18, fontweight='bold',
                            rotation=90, va='center', ha='center')

                if i == 0:
                    ax.set_title(display_name, fontsize=20, fontweight='bold', pad=25)
                else:
                    ax.set_title("", pad=0)

        cax = fig.add_axes([0.20, 0.02, 0.60, 0.012])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=self.config.BIAS_CMAP,
                norm=plt.Normalize(vmin=self.config.BIAS_VMIN,vmax=self.config.BIAS_VMAX
                )
            ),
            cax=cax,
            orientation='horizontal'
        )
        cb.ax.set_xlabel('Bias (°C)', fontsize=18, fontweight='normal', labelpad=8)
        cb.ax.xaxis.set_label_position('bottom')
        cb.ax.xaxis.set_label_coords(0.5, -0.9)
        cb.ax.tick_params(labelsize=14)

        plt.subplots_adjust(
            wspace=0.03, hspace=0.20,
            left=0.08, right=0.98,
            top=0.96, bottom=0.05
        )

        output_path = Path(self.config.OUTPUT_DIR) / self.config.OUTPUT_FILENAME
        plt.savefig(
            output_path,
            dpi=self.config.DPI,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"\n✅ 图片已保存: {output_path} (分辨率: {self.config.DPI} DPI)")
        plt.close(fig)


# ==================== 主函数 ====================
def main(force_recompute=False):
    """
    主执行流程

    Args:
        force_recompute: 是否强制重新计算数据(忽略缓存)
    """
    print("=" * 60)
    print("🚀 多方法海表温度偏差对比图生成器")
    print("=" * 60)

    config = Config()
    data_loader = DataLoader(config)
    data_loader.load_geo_info()
    data_loader.load_metrics()

    # ✅ 核心:加载或计算所有bias数据
    bias_cache = data_loader.load_or_calculate_all_bias(force_recompute=force_recompute)

    # 生成图表
    figure_generator = FigureGenerator(config, data_loader)
    figure_generator.create_comparison_figure(bias_cache)

    print("\n" + "=" * 60)
    print("🎉 图表生成完成!")
    print("=" * 60)
    print(f"📊 配置信息:")
    print(f"   - 模型数量: {len(config.MODELS)}")
    print(f"   - 方法数量: {len(config.METHODS)}")
    print(f"   - 总子图数: {len(config.MODELS) * len(config.METHODS)}")
    print(f"   - 输出文件: {config.OUTPUT_FILENAME}")
    print("=" * 60)


if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'

    # ✅ 第一次运行设为 True 来生成缓存,后续设为 False 使用缓存
    main(force_recompute=False)