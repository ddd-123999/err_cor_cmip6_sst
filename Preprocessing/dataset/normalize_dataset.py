# normalize_dataset.py
import numpy as np
import os
import shutil
from datetime import date


def normalize_and_save_datasets():
    """归一化所有数据集并保存到新文件夹"""

    # 创建归一化数据目录
    normalized_dir = "./ACCESS-CM2_normalized"
    os.makedirs(normalized_dir, exist_ok=True)

    # 原始数据目录
    original_dir = "./ACCESS-CM2"

    # 加载mask文件
    print("加载mask文件...")
    mask = np.load('../observation/obs/mask.npy')  # 假设mask.npy在同一目录
    ocean_mask = mask.astype(bool)
    print(f"Mask形状: {mask.shape}")
    print(f"海洋点数: {np.sum(mask == 1)}, 陆地点数: {np.sum(mask == 0)}")

    # 计算训练集的统计量（用于所有数据集的归一化）
    print("计算训练集统计量...")
    train_data_245 = np.load(os.path.join(original_dir, "ssp245_train.npz"))['sst']

    # 只使用海洋区域的数据计算统计量
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

    print(f"训练集统计量 - Min: {data_min:.4f}, Max: {data_max:.4f}")
    print(f"数据范围: {data_min:.4f} 到 {data_max:.4f}")

    # 保存归一化参数
    np.savez(os.path.join(normalized_dir, "normalization_params.npz"),
             min=data_min, max=data_max)

    # Min-Max归一化函数（修复陆地数据问题）
    def min_max_normalize_with_mask(data, data_min, data_max, mask):
        """
        Min-Max归一化到[0,1]范围，并确保陆地数据为0
        """
        # 创建归一化后的数组，初始化为0（陆地默认为0）
        normalized_data = np.zeros_like(data)

        # 只对海洋区域进行归一化
        ocean_mask = mask.astype(bool)

        if len(data.shape) == 3:  # 时间, 纬度, 经度
            for i in range(data.shape[0]):
                ocean_data = data[i][ocean_mask]
                if len(ocean_data) > 0:
                    normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                    normalized_data[i][ocean_mask] = normalized_ocean
        else:  # 2D数据
            ocean_data = data[ocean_mask]
            if len(ocean_data) > 0:
                normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                normalized_data[ocean_mask] = normalized_ocean

        return normalized_data

    # 处理所有数据集
    datasets = {
        'ssp245': ['train', 'val', 'test', 'pre']
    }

    for scenario, splits in datasets.items():
        for split in splits:
            input_path = os.path.join(original_dir, f"{scenario}_{split}.npz")
            output_path = os.path.join(normalized_dir, f"{scenario}_{split}.npz")

            if os.path.exists(input_path):
                print(f"处理: {scenario}_{split}")

                # 加载原始数据
                data_dict = np.load(input_path)
                sst_data = data_dict['sst']

                # Min-Max归一化（使用修复后的函数）
                sst_normalized = min_max_normalize_with_mask(sst_data, data_min, data_max, mask)

                # ... (代码前略) ...

                # 保存归一化后的数据
                np.savez(output_path, sst=sst_normalized)

                print("     正在验证 (节约内存模式)...")
                normalized_min = np.inf
                normalized_max = -np.inf
                land_is_zero = True
                ocean_mask_bool = ocean_mask.astype(bool)
                land_mask_bool = ~ocean_mask_bool

                # 逐个时间步迭代，避免内存爆炸
                for i in range(sst_normalized.shape[0]):
                    # 提取当前时间步的海洋数据
                    ocean_slice = sst_normalized[i][ocean_mask_bool]
                    if ocean_slice.size > 0:
                        current_min = np.nanmin(ocean_slice)
                        current_max = np.nanmax(ocean_slice)
                        if current_min < normalized_min:
                            normalized_min = current_min
                        if current_max > normalized_max:
                            normalized_max = current_max

                    # 检查陆地数据
                    if land_is_zero:  # 如果已经发现非0，则跳过
                        land_slice = sst_normalized[i][land_mask_bool]
                        if land_slice.size > 0 and not np.all(land_slice == 0):
                            land_is_zero = False
                # ----------------------------------------------------

                print(f"  ✅ 已保存: {output_path}")
                print(f"     海洋数据范围: [{normalized_min:.4f}, {normalized_max:.4f}]")
                print(f"     陆地数据全为0: {land_is_zero}")
            else:
                print(f"  ⚠️ 文件不存在: {input_path}")

    print(f"\n🎉 所有数据集已Min-Max归一化并保存到: {normalized_dir}")
    print(f"📊 归一化范围: [0, 1]")
    return data_min, data_max


if __name__ == "__main__":
    normalize_and_save_datasets()