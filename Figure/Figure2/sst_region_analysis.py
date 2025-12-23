import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

# ==================== 配置 ====================
class Config:
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
    MASK_BASE_PATH = '../../Preprocessing/observation/obs/mask.npy'

    # 区域掩码路径（根据create_mask.py生成的）
    REGION_MASK_DIR = 'Region_Masks'

    OUTPUT_DIR = './sst_analysis_results'


# ==================== 数据加载 ====================
def load_region_masks(region_names=['EG', 'BaS']):
    """加载区域掩码"""
    masks = {}
    for region in region_names:
        mask_path = Path(Config.REGION_MASK_DIR) / f'mask_{region}.npy'
        if mask_path.exists():
            masks[region] = np.load(mask_path).astype(bool)
            print(f"✓ 加载 {region} 掩码: {np.sum(masks[region])} 个海洋网格点")
        else:
            print(f"⚠️  未找到 {region} 掩码文件: {mask_path}")
            print(f"   请先运行 create_mask.py 生成区域掩码")
            return None
    return masks


def load_obs_sst():
    """加载观测数据"""
    try:
        data = np.load(Config.OBS_PATH)['sst']
        # 使用测试集时间段: 最后1827天 (2020-2024)
        sst_test = data[-1827:]
        print(f"✓ 观测数据形状: {sst_test.shape}")
        return sst_test
    except Exception as e:
        print(f"✗ 加载观测数据失败: {e}")
        return None


def load_model_sst(model_name):
    """加载单个模型的SST数据"""
    path = Config.MODEL_PATH_TEMPLATE.format(model=model_name)
    try:
        data = np.load(path)['sst']
        return data
    except Exception as e:
        print(f"✗ 加载 {model_name} 失败: {e}")
        return None


# ==================== 区域统计 ====================
def calculate_gridpoint_sst_range(sst_data, region_mask):
    """
    计算逐格点多年平均SST的范围
    sst_data: (time, lat, lon)
    region_mask: (lat, lon) boolean

    返回: (min_sst, max_sst) - 区域内所有格点多年平均SST的最小值和最大值
    """
    if sst_data is None or region_mask is None:
        return np.nan, np.nan

    # 1. 先计算每个格点的多年平均 (time维度求平均)
    temporal_mean = np.nanmean(sst_data, axis=0)  # shape: (lat, lon)

    # 2. 提取区域内的格点
    regional_values = temporal_mean[region_mask]

    # 3. 过滤NaN值
    valid_values = regional_values[~np.isnan(regional_values)]

    if len(valid_values) == 0:
        return np.nan, np.nan

    # 4. 返回该区域所有格点的最小值和最大值
    return np.min(valid_values), np.max(valid_values)


# ==================== 主分析函数 ====================
def analyze_regional_sst():
    """分析EG和BaS海域的逐格点多年平均SST范围"""

    # 1. 加载掩码
    print("\n" + "=" * 60)
    print("步骤 1: 加载区域掩码")
    print("=" * 60)
    region_masks = load_region_masks(['EG', 'BaS'])
    if region_masks is None:
        return

    # 合并EG和BaS掩码
    combined_mask = np.logical_or(region_masks['EG'], region_masks['BaS'])
    print(f"✓ EG+BaS 合并区域: {np.sum(combined_mask)} 个海洋网格点")

    # 2. 加载观测数据
    print("\n" + "=" * 60)
    print("步骤 2: 计算观测数据的逐格点多年平均SST范围")
    print("=" * 60)
    obs_sst = load_obs_sst()
    obs_min, obs_max = calculate_gridpoint_sst_range(obs_sst, combined_mask)
    obs_range = obs_max - obs_min
    print(f"✓ 观测数据 (OISST) 逐格点多年平均SST范围: [{obs_min:.2f}, {obs_max:.2f}]°C")
    print(f"  范围跨度: {obs_range:.2f}°C")

    # 3. 加载模型数据
    print("\n" + "=" * 60)
    print("步骤 3: 计算21个模型的逐格点多年平均SST范围")
    print("=" * 60)
    model_ranges = {}
    for i, model in enumerate(Config.MODELS, 1):
        model_sst = load_model_sst(model)
        model_min, model_max = calculate_gridpoint_sst_range(model_sst, combined_mask)
        model_ranges[model] = {
            'min': model_min,
            'max': model_max,
            'range': model_max - model_min
        }
        print(f"  [{i:2d}/21] {model:20s}: [{model_min:6.2f}, {model_max:6.2f}]°C  "
              f"(范围: {model_max - model_min:.2f}°C)")

    # 4. 统计分析
    print("\n" + "=" * 60)
    print("步骤 4: 统计结果")
    print("=" * 60)

    valid_models = {k: v for k, v in model_ranges.items()
                    if not (np.isnan(v['min']) or np.isnan(v['max']))}

    if len(valid_models) == 0:
        print("✗ 没有有效的模型数据")
        return

    # 收集所有模型的最小值和最大值
    all_mins = [v['min'] for v in valid_models.values()]
    all_maxs = [v['max'] for v in valid_models.values()]
    all_ranges = [v['range'] for v in valid_models.values()]

    results = {
        'Obs_Min': obs_min,
        'Obs_Max': obs_max,
        'Obs_Range': obs_range,
        'Model_Min_Lowest': np.min(all_mins),
        'Model_Min_Highest': np.max(all_mins),
        'Model_Max_Lowest': np.min(all_maxs),
        'Model_Max_Highest': np.max(all_maxs),
        'Model_Range_Min': np.min(all_ranges),
        'Model_Range_Max': np.max(all_ranges),
        'Model_Range_Mean': np.mean(all_ranges)
    }

    print(f"\n观测数据 (OISST) 逐格点多年平均SST范围:")
    print(f"  最小值:                 {results['Obs_Min']:.2f}°C")
    print(f"  最大值:                 {results['Obs_Max']:.2f}°C")
    print(f"  范围跨度:               {results['Obs_Range']:.2f}°C")

    print(f"\n模型集合统计 (逐格点多年平均SST):")
    print(f"  最小值范围:             [{results['Model_Min_Lowest']:.2f}, {results['Model_Min_Highest']:.2f}]°C")
    print(f"  最大值范围:             [{results['Model_Max_Lowest']:.2f}, {results['Model_Max_Highest']:.2f}]°C")
    print(f"  范围跨度:               [{results['Model_Range_Min']:.2f}, {results['Model_Range_Max']:.2f}]°C")
    print(f"  平均范围跨度:           {results['Model_Range_Mean']:.2f}°C")

    # 5. 保存结果
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)

    # 保存详细数据
    df = pd.DataFrame({
        'Model': ['Observation'] + list(valid_models.keys()),
        'SST_Min': [obs_min] + all_mins,
        'SST_Max': [obs_max] + all_maxs,
        'SST_Range': [obs_range] + all_ranges
    })
    csv_path = Path(Config.OUTPUT_DIR) / 'EG_BaS_SST_gridpoint_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 详细结果已保存到: {csv_path}")

    # 6. 可视化
    plot_results(obs_min, obs_max, valid_models)

    return results, valid_models


# ==================== 可视化 ====================
def plot_results(obs_min, obs_max, model_ranges):
    """绘制逐格点SST范围对比图"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # 准备数据
    models = ['Observation'] + list(model_ranges.keys())
    n_models = len(models)
    y_positions = np.arange(n_models)

    # 观测数据的范围
    mins = [obs_min] + [v['min'] for v in model_ranges.values()]
    maxs = [obs_max] + [v['max'] for v in model_ranges.values()]
    ranges = [maxs[i] - mins[i] for i in range(len(mins))]

    # 绘制水平条形图（显示范围）
    colors = ['green'] + ['blue'] * len(model_ranges)
    alphas = [0.8] + [0.5] * len(model_ranges)

    for i, (y, min_val, max_val, color, alpha) in enumerate(zip(y_positions, mins, maxs, colors, alphas)):
        # 绘制从min到max的水平线段
        ax.plot([min_val, max_val], [y, y], color=color, linewidth=8, alpha=alpha, solid_capstyle='round')
        # 在两端添加标记
        ax.scatter([min_val, max_val], [y, y], color=color, s=100, alpha=alpha, zorder=3)

        # 添加数值标签
        if i == 0:  # 观测数据
            ax.text(max_val + 0.3, y, f'[{min_val:.1f}, {max_val:.1f}]°C',
                    va='center', fontsize=9, fontweight='bold', color='green')
        else:
            ax.text(max_val + 0.3, y, f'[{min_val:.1f}, {max_val:.1f}]°C',
                    va='center', fontsize=8, color='black')

    # 添加观测数据的参考线
    ax.axvline(obs_min, color='green', linewidth=1.5, linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(obs_max, color='green', linewidth=1.5, linestyle='--', alpha=0.5, zorder=1)

    # 设置坐标轴
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('SST (°C)', fontsize=13, fontweight='bold')
    ax.set_title('EG+BaS Region: Gridpoint Multi-year Mean SST Range\n(Green: Observation, Blue: CMIP6 Models)',
                 fontsize=14, fontweight='bold', pad=15)

    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 反转y轴，让观测数据在顶部
    ax.invert_yaxis()

    plt.tight_layout()

    save_path = Path(Config.OUTPUT_DIR) / 'EG_BaS_SST_gridpoint_range_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存到: {save_path}")

    # plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10

    print("\n" + "=" * 60)
    print("  EG和BaS海域逐格点多年平均SST范围分析")
    print("  (对比观测数据与21个CMIP6模型)")
    print("=" * 60)
    print("\n分析方法:")
    print("  1. 对每个格点计算多年平均SST (时间维度平均)")
    print("  2. 统计区域内所有格点的最小值和最大值")
    print("  3. 对比观测与模型的空间温度范围")
    print("=" * 60)

    results, model_ranges = analyze_regional_sst()

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)