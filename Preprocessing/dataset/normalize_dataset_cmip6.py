import numpy as np
import os
from datetime import date


def normalize_model_data(original_dir, mask_path):
    """
    归一化单个模型的数据集。

    参数:
    original_dir (str): 包含 ssp245_*.npz 文件的目录 (例如 './model_npz_raw/EC-Earth3')
    mask_path (str): 'mask.npy' 文件的路径
    """

    # 1. 创建归一化数据目录
    # 例如: './model_npz_raw/EC-Earth3' -> './model_npz_raw/EC-Earth3_normalized'
    normalized_dir = f"{original_dir}_normalized"
    os.makedirs(normalized_dir, exist_ok=True)

    # 2. 加载mask文件
    print(f"  加载mask文件: {mask_path}")
    if not os.path.exists(mask_path):
        print(f"  ❌ 错误: 归一化时找不到Mask文件: {mask_path}")
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = np.load(mask_path)
    ocean_mask = mask.astype(bool)
    print(f"  Mask形状: {mask.shape}")
    print(f"  海洋点数: {np.sum(mask == 1)}, 陆地点数: {np.sum(mask == 0)}")

    # 3. 计算训练集的统计量（用于所有数据集的归一化）
    print("  计算训练集统计量...")
    train_npz_path = os.path.join(original_dir, "ssp245_train.npz")
    if not os.path.exists(train_npz_path):
        print(f"  ❌ 错误: 找不到训练集文件: {train_npz_path}")
        raise FileNotFoundError(f"Training set not found: {train_npz_path}")

    train_data_245 = np.load(train_npz_path)['sst']

    # 4. 只使用海洋区域的数据计算统计量
    if len(train_data_245.shape) == 3:  # 时间, 纬度, 经度
        # 对每个时间步应用mask
        ocean_data_list = []
        for i in range(train_data_245.shape[0]):
            ocean_data = train_data_245[i][ocean_mask]
            ocean_data_list.append(ocean_data)
        all_train_ocean_data = np.concatenate(ocean_data_list)
    else:
        all_train_ocean_data = train_data_245[ocean_mask]

    # Min-Max归一化参数（只基于海洋数据，忽略NaN）
    data_min = np.nanmin(all_train_ocean_data)
    data_max = np.nanmax(all_train_ocean_data)

    print(f"  训练集统计量 - Min: {data_min:.4f}, Max: {data_max:.4f}")
    print(f"  数据范围: {data_min:.4f} 到 {data_max:.4f}")

    # 保存归一化参数
    np.savez(os.path.join(normalized_dir, "normalization_params.npz"),
             min=data_min, max=data_max)

    # 5. Min-Max归一化函数（修复陆地数据问题）
    def min_max_normalize_with_mask(data, data_min, data_max, mask):
        """
        Min-Max归一化到[0,1]范围，并确保陆地数据为0
        """
        # 创建归一化后的数组，初始化为0（陆地默认为0）
        normalized_data = np.zeros_like(data)

        # 只对海洋区域进行归一化
        ocean_mask_bool = mask.astype(bool)

        if len(data.shape) == 3:  # 时间, 纬度, 经度
            for i in range(data.shape[0]):
                # 检查此时间步是否有数据
                if i < data.shape[0]:
                    ocean_data = data[i][ocean_mask_bool]
                    if len(ocean_data) > 0:
                        normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                        normalized_data[i][ocean_mask_bool] = normalized_ocean
        else:  # 2D数据
            ocean_data = data[ocean_mask_bool]
            if len(ocean_data) > 0:
                normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                normalized_data[ocean_mask_bool] = normalized_ocean

        return normalized_data

    # 6. 处理所有数据集
    datasets = {
        'ssp245': ['train', 'val', 'test', 'pre']
    }

    for scenario, splits in datasets.items():
        for split in splits:
            input_path = os.path.join(original_dir, f"{scenario}_{split}.npz")
            output_path = os.path.join(normalized_dir, f"{scenario}_{split}.npz")

            if os.path.exists(input_path):
                print(f"  处理: {scenario}_{split}")

                # 加载原始数据
                data_dict = np.load(input_path)
                sst_data = data_dict['sst']

                # Min-Max归一化（使用修复后的函数）
                sst_normalized = min_max_normalize_with_mask(sst_data, data_min, data_max, mask)

                # 保存归一化后的数据
                np.savez(output_path, sst=sst_normalized)

                # 验证 (使用节约内存的模式)
                print("     正在验证 (节约内存模式)...")
                normalized_min = np.inf
                normalized_max = -np.inf
                land_is_zero = True
                ocean_mask_bool = ocean_mask.astype(bool)
                land_mask_bool = ~ocean_mask_bool

                for i in range(sst_normalized.shape[0]):
                    ocean_slice = sst_normalized[i][ocean_mask_bool]
                    if ocean_slice.size > 0:
                        current_min = np.nanmin(ocean_slice)
                        current_max = np.nanmax(ocean_slice)
                        if current_min < normalized_min: normalized_min = current_min
                        if current_max > normalized_max: normalized_max = current_max

                    if land_is_zero:
                        land_slice = sst_normalized[i][land_mask_bool]
                        if land_slice.size > 0 and not np.all(land_slice == 0):
                            land_is_zero = False

                print(f"    ✅ 已保存: {output_path}")
                print(f"       海洋数据范围: [{normalized_min:.4f}, {normalized_max:.4f}]")
                print(f"       陆地数据全为0: {land_is_zero}")
            else:
                print(f"    ⚠️ 文件不存在: {input_path}")

    print(f"\n  🎉 {original_dir} 的所有数据集已Min-Max归一化。")
    print(f"  📊 归一化范围: [0, 1]")
    return normalized_dir  # 返回归一化目录的路径


# 允许此脚本被导入，或被独立运行（用于测试）
if __name__ == "__main__":
    pass
    # print("--- 正在以独立模式测试 normalize_dataset.py ---")
    #
    # # --- !! 测试配置 !! ---
    # # 假设 'dataset_split.py' 已经运行并创建了 'model_npz_raw_TEST/EC-Earth3' 目录
    # TEST_ORIGINAL_DIR = r"./EC-Earth3"
    # TEST_MASK_PATH = r"../observation/mask.npy"
    # # -------------------------
    #
    # if not os.path.exists(TEST_ORIGINAL_DIR):
    #     print(f"测试目录 '{TEST_ORIGINAL_DIR}' 未找到，跳过。")
    # elif not os.path.exists(TEST_MASK_PATH):
    #     print(f"测试Mask文件 '{TEST_MASK_PATH}' 未找到，跳过。")
    # else:
    #     normalize_model_data(
    #         TEST_ORIGINAL_DIR,
    #         TEST_MASK_PATH
    #     )
