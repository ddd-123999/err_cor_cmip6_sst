# obs_processing_utils.py (重命名自 preprocessing_cmip6.py)
# 包含从 preprocessing.py 重构的函数

import numpy as np
import os

# ----------------- 归一化函数 -----------------

def _min_max_normalize_with_mask(data, data_min, data_max, mask):
    """
    内部函数：Min-Max归一化到[0,1]范围，并确保陆地数据为0
    """
    normalized_data = np.zeros_like(data)
    ocean_mask = mask.astype(bool)

    if len(data.shape) == 3:  # 3D数据: (时间, 纬度, 经度)
        for t in range(data.shape[0]):
            ocean_data = data[t][ocean_mask]
            if len(ocean_data) > 0:
                normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                normalized_data[t][ocean_mask] = normalized_ocean
    else:  # 2D数据
        ocean_data = data[ocean_mask]
        if len(ocean_data) > 0:
            normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
            normalized_data[ocean_mask] = normalized_ocean

    return normalized_data


def normalize_obs_for_model(model_name, base_param_dir, raw_data_dir, final_output_dir,
                            output_filename):
    """
    归一化观测数据。
    加载原始 sst.npz，加载特定模型的参数，
    并保存归一化后的 sst.npz。
    """

    # 1. 定义路径
    raw_sst_path = os.path.join(raw_data_dir, 'sst_daily_not_to_be_normalized.npz')
    mask_path = os.path.join(raw_data_dir, 'mask.npy')

    # 构建归一化参数的路径
    param_file_path = os.path.join(
        base_param_dir,
        f"{model_name}_normalized",
        "normalization_params.npz"
    )

    # 构建最终输出目录和文件
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, output_filename)

    # 2. 检查所需文件
    if not os.path.exists(param_file_path):
        raise FileNotFoundError(f"找不到 {model_name} 的参数文件: {param_file_path}")
    if not os.path.exists(raw_sst_path):
        raise FileNotFoundError(f"找不到原始观测数据: {raw_sst_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"找不到 mask 文件: {mask_path}")

    # 3. 加载归一化参数
    print(f"  加载参数: {param_file_path}")
    norm_params = np.load(param_file_path)
    data_min = norm_params['min']
    data_max = norm_params['max']
    print(f"  参数 (Min: {data_min:.4f}, Max: {data_max:.4f})")

    # 4. 加载原始数据
    print(f"  加载原始观测数据: {raw_sst_path}")
    data = np.load(raw_sst_path)
    sst_data = data['sst']
    mask = np.load(mask_path)

    # 5. 应用归一化
    print("  正在归一化...")
    sst_normalized = _min_max_normalize_with_mask(sst_data, data_min, data_max, mask)

    # 6. 验证归一化
    ocean_mask = mask.astype(bool)
    ocean_data_normalized = sst_normalized[:, ocean_mask]
    normalized_min = np.nanmin(ocean_data_normalized)
    normalized_max = np.nanmax(ocean_data_normalized)

    land_mask = mask == 0
    land_values = sst_normalized[:, land_mask]
    land_all_zero = np.all(land_values == 0)

    print(f"  验证: 海洋范围 [{normalized_min:.4f}, {normalized_max:.4f}]")
    print(f"  验证: 陆地全为0 = {land_all_zero}")

    # 7. 保存归一化后的数据
    np.savez(final_output_path,
             lon=data['lon'],
             lat=data['lat'],
             x_proj=data['x_proj'],
             y_proj=data['y_proj'],
             sst=sst_normalized)

    data.close()
    return final_output_path