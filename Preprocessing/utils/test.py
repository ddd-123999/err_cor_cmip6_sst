import numpy as np
import os


def detailed_land_check_single_file(file_path, mask):
    """
    详细检查单个文件的陆地数据
    """
    print(f"\n🔍 详细检查: {os.path.basename(file_path)}")
    print("-" * 60)

    data_dict = np.load(file_path)

    for key in data_dict.files:
        data = data_dict[key]
        print(f"数组 '{key}': {data.shape}")

        total_non_zero_count = 0
        max_non_zero_value = 0

        # 检查每个时间步
        for t in range(data.shape[0]):
            current_data = data[t]
            land_data = current_data[mask == 0]

            # 找出非零且非NaN的值
            valid_land = land_data[~np.isnan(land_data)]
            non_zero_mask = valid_land != 0
            non_zero_count = np.sum(non_zero_mask)

            if non_zero_count > 0:
                total_non_zero_count += non_zero_count
                current_max = np.max(np.abs(valid_land[non_zero_mask]))
                max_non_zero_value = max(max_non_zero_value, current_max)
                print(f"  时间步 {t:3d}: {non_zero_count:4d} 个非零点, 最大值: {current_max:.8f}")

        if total_non_zero_count == 0:
            print(f"  ✅ 完美! 所有陆地数据均为0或NaN")
        else:
            print(f"  ❗ 总计: {total_non_zero_count} 个非零陆地点")
            print(f"  📈 最大非零绝对值: {max_non_zero_value:.8f}")


# 使用详细检查
mask_path = "../observation/obs/mask.npy"
mask = np.load(mask_path)

base_path = "../dataset/ACCESS-CM2_normalized/"

# 选择一个文件进行详细检查
test_file = os.path.join(base_path, "ssp245_test.npz")
if os.path.exists(test_file):
    detailed_land_check_single_file(test_file, mask)
else:
    print(f"测试文件不存在: {test_file}")


# def verify_mask_quality(mask_path, data_path, first_timestep_only=True):
#     """
#     验证 mask 的质量
#     """
#     import numpy as np
#
#     print("=" * 60)
#     print("Mask 质量验证")
#     print("=" * 60)
#
#     # 加载 mask 和数据
#     mask = np.load(mask_path)
#     data = np.load(data_path)['sst']
#
#     if first_timestep_only:
#         data = data[0, :, :]
#
#     print(f"\n1. Mask 基本信息")
#     print(f"   形状: {mask.shape}")
#     print(f"   dtype: {mask.dtype}")
#     print(f"   唯一值: {np.unique(mask)}")
#
#     # 统计信息
#     valid_count = np.sum(mask == 1)
#     invalid_count = np.sum(mask == 0)
#     total_count = mask.size
#
#     print(f"\n2. Mask 统计")
#     print(f"   有效点(mask=1): {valid_count} ({100 * valid_count / total_count:.1f}%)")
#     print(f"   无效点(mask=0): {invalid_count} ({100 * invalid_count / total_count:.1f}%)")
#     print(f"   总点数: {total_count}")
#     print(f"   验证: {valid_count + invalid_count} == {total_count} ✓")
#
#     # 检查海洋数据
#     ocean_data = data[mask == 1]
#     print(f"\n3. 海洋数据质量(mask=1)")
#     print(f"   NaN数量: {np.sum(np.isnan(ocean_data))}")
#     print(f"   数据范围: [{np.nanmin(ocean_data):.4f}, {np.nanmax(ocean_data):.4f}]")
#     print(f"   数据统计: 均值={np.nanmean(ocean_data):.4f}, 方差={np.nanvar(ocean_data):.4f}")
#
#     # 检查陆地数据
#     land_data = data[mask == 0]
#     print(f"\n4. 陆地数据质量(mask=0)")
#     print(f"   NaN数量: {np.sum(np.isnan(land_data))}")
#     print(f"   全为0: {np.all(land_data == 0) if len(land_data) > 0 else 'N/A'}")
#     print(f"   数据范围: [{np.nanmin(land_data):.4f}, {np.nanmax(land_data):.4f}]")
#
#     non_nan_land = land_data[~np.isnan(land_data)]
#     if len(non_nan_land) > 0 and not np.all(non_nan_land == 0):
#         print(f"   ⚠️ 警告：陆地不全为0，有 {len(non_nan_land)} 个非零值")
#     else:
#         print(f"   ✅ 陆地正确设置为0或NaN")
#
#     print("\n" + "=" * 60)
#     return mask, data
#
#
# # 使用验证工具
# verify_mask_quality(
#     mask_path="../observation/mask.npy",
#     data_path="../observation/sst_daily_not_to_be_normalized.npz",
#     first_timestep_only=True
# )