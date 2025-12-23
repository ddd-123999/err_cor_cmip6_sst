# 处理SST观测数据，生成sst.npz和mask.npy，并进行北极区筛选
import netCDF4
import numpy as np
import os
import sys

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Preprocessing.utils.filter_northern_hemisphere import filter_northern_hemisphere

# 导入 process_and_save 模块
sys.path.append(os.path.dirname(__file__))
from Preprocessing.utils.process_and_save import load_nc_meta, process_and_save_chunk

# ---------------- 配置参数 ----------------
CHUNK_SIZE = 365  # 每次读取365天（1年）
NORMALIZE_DATA = True # 是否进行归一化处理
NORMALIZATION_PARAMS_PATH = "../dataset/ACCESS-CM2_normalized/normalization_params.npz"  # 训练集归一化参数路径

# 读取SST数据文件
print("读取SST观测数据...")
input_file = 'obs/oisst_daily_19820101-20241231.nc'

# 使用 process_and_save 中的函数读取元数据
sst_first, lon, lat, total_time = load_nc_meta(input_file)
print(f"原始SST数据形状: ({total_time}, {lat.shape[0]}, {lon.shape[0]})")
print(f"原始经度形状: {lon.shape}, 纬度形状: {lat.shape}")

# 获取投影坐标
dataset = netCDF4.Dataset(input_file, 'r')
x_proj = dataset.variables['x_proj'][:]
y_proj = dataset.variables['y_proj'][:]
dataset.close()
print(f"原始x_proj形状: {x_proj.shape}, y_proj形状: {y_proj.shape}")

# ---------------- 筛选北极区数据 ----------------
print("\n筛选北极区数据...")
# 使用第一个时间步筛选北极区
sst_first_nh, lon_nh, lat_nh = filter_northern_hemisphere(sst_first, lon, lat)
print(f"北极区SST数据形状: {sst_first_nh.shape}")
print(f"北极区经度形状: {lon_nh.shape}, 纬度形状: {lat_nh.shape}")
print(f"纬度范围: {np.min(lat_nh):.2f}° 到 {np.max(lat_nh):.2f}°")

# 筛选投影坐标（与经纬度使用相同的筛选逻辑）
print("\n筛选投影坐标...")
if len(lat.shape) == 1:  # 一维纬度
    # 创建北极区掩码
    nh_mask = lat >= 66
    nh_lat_indices = np.where(nh_mask)[0]  # 获取纬度索引
    print(f"北极区掩码True数量: {np.sum(nh_mask)} / {len(nh_mask)}")

    # 筛选投影坐标
    if len(x_proj.shape) == 2 and x_proj.shape[0] == len(lat):
        x_proj_nh = x_proj[nh_mask, :]
        print(f"北极区x_proj形状: {x_proj_nh.shape}")
    else:
        x_proj_nh = x_proj
        print(f"x_proj保持原形状: {x_proj_nh.shape}")

    if len(y_proj.shape) == 2 and y_proj.shape[0] == len(lat):
        y_proj_nh = y_proj[nh_mask, :]
        print(f"北极区y_proj形状: {y_proj_nh.shape}")
    else:
        y_proj_nh = y_proj
        print(f"y_proj保持原形状: {y_proj_nh.shape}")

# ---------------- 创建mask ----------------
print("\n创建mask...")
# 使用第一个时间步来创建mask
mask = np.where(~np.isnan(sst_first_nh[0]), 1, 0)
mask = mask.astype(np.int8)
print(f"Mask形状: {mask.shape}")
print(f"有效数据点数: {np.sum(mask)}")

# ---------------- 加载训练集归一化参数 ----------------
if NORMALIZE_DATA:
    print("\n加载训练集归一化参数...")
    if os.path.exists(NORMALIZATION_PARAMS_PATH):
        norm_params = np.load(NORMALIZATION_PARAMS_PATH)
        data_min = norm_params['min']
        data_max = norm_params['max']
        print(f"✅ 成功加载训练集归一化参数")
        print(f"训练集统计量 - Min: {data_min:.4f}, Max: {data_max:.4f}")
        print(f"归一化范围: {data_min:.4f} 到 {data_max:.4f}")
    else:
        print(f"❌ 未找到训练集归一化参数文件: {NORMALIZATION_PARAMS_PATH}")
        print("将使用观测数据自身的统计量进行归一化")

        # 计算观测数据的统计量作为备选
        print("计算观测数据统计量...")
        all_sst_data = []

        # 分批读取所有数据计算统计量
        for start_idx in range(0, total_time, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, total_time)
            print(f"  读取数据块: {start_idx} 到 {end_idx}")

            dataset = netCDF4.Dataset(input_file, 'r')
            sst_chunk = dataset.variables['sst'][start_idx:end_idx, :, :]
            dataset.close()

            # 筛选北极区
            if len(lat.shape) == 1:  # 一维纬度
                sst_chunk_nh = sst_chunk[:, nh_lat_indices, :]
            else:
                # 二维纬度处理
                sst_chunk_nh = sst_chunk[:, nh_mask, :]

            # 应用mask，只保留有效数据点
            sst_chunk_valid = sst_chunk_nh[:, mask.astype(bool)]
            all_sst_data.append(sst_chunk_valid)

        # 合并所有数据计算统计量
        all_sst_data = np.concatenate(all_sst_data, axis=0)

        # 计算Min-Max归一化参数（忽略NaN）
        data_min = np.nanmin(all_sst_data)
        data_max = np.nanmax(all_sst_data)

        print(f"观测数据统计量 - Min: {data_min:.4f}, Max: {data_max:.4f}")


    # 定义归一化函数 - 修复陆地数据问题
    def min_max_normalize_with_mask(data, data_min, data_max, mask):
        """
        Min-Max归一化到[0,1]范围，并确保陆地数据为0
        支持2D和3D数据
        """
        # 创建归一化后的数组，初始化为0（陆地默认为0）
        normalized_data = np.zeros_like(data)

        # 只对海洋区域进行归一化
        ocean_mask = mask.astype(bool)

        if len(data.shape) == 3:  # 3D数据: (时间, 纬度, 经度)
            # 对每个时间步单独处理
            for t in range(data.shape[0]):
                # 获取当前时间步的海洋数据
                ocean_data = data[t][ocean_mask]
                if len(ocean_data) > 0:
                    # 归一化海洋数据
                    normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                    # 将归一化后的海洋数据放回对应位置
                    normalized_data[t][ocean_mask] = normalized_ocean
        else:  # 2D数据
            ocean_data = data[ocean_mask]
            if len(ocean_data) > 0:
                normalized_ocean = (ocean_data - data_min) / (data_max - data_min)
                normalized_data[ocean_mask] = normalized_ocean

        return normalized_data

# ---------------- 使用 process_and_save_chunk 分批处理所有数据 ----------------
print("\n开始分批处理所有数据...")
output_file = 'sst_daily.npz'

# 处理经纬度和投影坐标中的NaN值
lon_nh = np.nan_to_num(lon_nh, nan=0.0)
lat_nh = np.nan_to_num(lat_nh, nan=0.0)
x_proj_nh = np.nan_to_num(x_proj_nh, nan=0.0)
y_proj_nh = np.nan_to_num(y_proj_nh, nan=0.0)

# 调用 process_and_save_chunk 处理所有数据
stats = process_and_save_chunk(
    input_file=input_file,
    start_idx=0,
    end_idx=total_time,
    output_file=output_file,
    lon=lon_nh,
    lat=lat_nh,
    mask=mask,
    nh_lat_indices=nh_lat_indices,
    chunk_size=CHUNK_SIZE
)

# ---------------- 对保存的数据进行归一化处理 ----------------
if NORMALIZE_DATA:
    print("\n对已保存的数据进行归一化处理...")

    # 加载已保存的数据
    data = np.load(output_file)
    sst_data = data['sst']

    # 应用归一化（使用修复后的函数，确保陆地数据为0）
    sst_normalized = min_max_normalize_with_mask(sst_data, data_min, data_max, mask)

    # 验证归一化范围
    ocean_mask = mask.astype(bool)
    if len(sst_normalized.shape) == 3:  # 3D数据
        # 取所有时间步的海洋数据
        ocean_data_normalized = sst_normalized[:, ocean_mask]
    else:  # 2D数据
        ocean_data_normalized = sst_normalized[ocean_mask]

    normalized_min = np.nanmin(ocean_data_normalized) if len(ocean_data_normalized) > 0 else 0
    normalized_max = np.nanmax(ocean_data_normalized) if len(ocean_data_normalized) > 0 else 0

    # 检查陆地数据
    if len(sst_normalized.shape) == 3:  # 3D数据
        land_data_normalized = sst_normalized[:, ~ocean_mask]
    else:  # 2D数据
        land_data_normalized = sst_normalized[~ocean_mask]

    land_is_zero = np.all(land_data_normalized == 0) if len(land_data_normalized) > 0 else True

    print(f"海洋数据归一化后范围: [{normalized_min:.4f}, {normalized_max:.4f}]")
    print(f"陆地数据全为0: {land_is_zero}")

    # 重新保存归一化后的数据
    np.savez(output_file,
             lon=data['lon'],
             lat=data['lat'],
             x_proj=data['x_proj'] if 'x_proj' in data else x_proj_nh,
             y_proj=data['y_proj'] if 'y_proj' in data else y_proj_nh,
             sst=sst_normalized)
    data.close()

    print("✅ 观测数据归一化完成")

# ---------------- 保存投影坐标和其他数据 ----------------
print("\n保存投影坐标数据...")
# 重新加载保存的npz文件，添加投影坐标
data = np.load(output_file)
np.savez(output_file,
         lon=data['lon'],
         lat=data['lat'],
         x_proj=x_proj_nh,
         y_proj=y_proj_nh,
         sst=data['sst'])
data.close()

np.save('obs/mask.npy', mask)

# ---------------- 输出统计信息 ----------------
print("\n" + "=" * 50)
print("数据处理完成！")
print("=" * 50)
print(f"SST数据形状: {stats['shape']}")
print(f"经度数据形状: {lon_nh.shape}")
print(f"纬度数据形状: {lat_nh.shape}")
print(f"x_proj形状: {x_proj_nh.shape}")
print(f"y_proj形状: {y_proj_nh.shape}")
print(f"Mask形状: {mask.shape}")
print(f"有效数据点数: {np.sum(mask)}")

if NORMALIZE_DATA:
    print(f"数据状态: 已归一化到 [0, 1] 范围")
    print(f"使用的归一化参数 - Min: {data_min:.4f}, Max: {data_max:.4f}")
    # 检查是否有超出训练集范围的数据
    original_min = np.nanmin(sst_data) if 'sst_data' in locals() else stats['min']
    original_max = np.nanmax(sst_data) if 'sst_data' in locals() else stats['max']
    if original_min < data_min or original_max > data_max:
        print(f"⚠️  注意: 观测数据范围 [{original_min:.4f}, {original_max:.4f}] 超出训练集范围")

    # 最终验证陆地数据
    final_data = np.load(output_file)
    final_sst = final_data['sst']
    land_mask = mask == 0
    land_values = final_sst[:, land_mask] if len(final_sst.shape) == 3 else final_sst[land_mask]
    land_all_zero = np.all(land_values == 0)
    print(f"✅ 陆地数据验证: 全为0 = {land_all_zero}")
    final_data.close()
else:
    if stats['min'] != 0 and stats['max'] != 0:
        print(f"海洋SST数据范围: {stats['min']:.2f} 到 {stats['max']:.2f} °C")
    else:
        print("警告：没有找到有效的海洋数据")

print(f"纬度范围: {np.min(lat_nh):.2f}° 到 {np.max(lat_nh):.2f}°")
print(f"时间跨度: 1982-01-01 到 2024-12-31 ({total_time} 天)")
print("\n数据已保存到:")
print(f"  - {output_file} (包含 lon, lat, x_proj, y_proj, sst)")
print("  - mask.npy")
print("=" * 50)