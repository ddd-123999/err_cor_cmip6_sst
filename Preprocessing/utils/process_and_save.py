"""
数据处理和保存模块
提供分批读取、处理NetCDF数据并保存为npz格式的功能
"""

import netCDF4
import numpy as np
import os
import gc # 导入gc模块

# ---------------- 配置参数 ----------------
CHUNK_SIZE = 730  # 每次读取730天（约2年）
# MEMORY_LIMIT_GB 在这个新方案中不再需要，可以注释掉或删除
# MEMORY_LIMIT_GB = 5  # 累积数据超过5GB就保存到临时文件


# ---------------- 读取函数 ----------------
# load_nc_meta 和 read_nc_chunk 函数保持不变
def load_nc_meta(path):
    """
    只读取元数据和第一个时间步（用于获取mask）
    """
    ds = netCDF4.Dataset(path, "r")
    sst_first = ds.variables["sst"][0:1, :, :]
    lon = ds.variables["lon"][:]
    lat = ds.variables["lat"][:]
    time_len = ds.variables["sst"].shape[0]
    ds.close()
    return sst_first, lon, lat, time_len


def read_nc_chunk(path, start_idx, end_idx):
    """
    分批读取SST数据
    """
    ds = netCDF4.Dataset(path, "r")
    sst = ds.variables["sst"][start_idx:end_idx, :, :]
    ds.close()
    return sst

# ---------------- 统一的分批处理函数 (修改版) ----------------
def process_and_save_chunk(input_file, start_idx, end_idx, output_file,
                           lon, lat, mask, nh_lat_indices,
                           chunk_size=CHUNK_SIZE): # 移除了 memory_limit_gb
    """
    分批读取、处理并保存数据 (优化版：使用memmap预分配)
    """
    print(f"\n处理 {output_file}...")
    total_days = end_idx - start_idx
    print(f"  时间范围: {start_idx} 到 {end_idx} (共{total_days}天)")

    min_sst = float('inf')
    max_sst = float('-inf')

    # 1. 确定最终输出数据的形状
    # 假设 mask 的形状是 (height, width)
    final_shape = (total_days, mask.shape[0], mask.shape[1])
    # 确定数据类型，例如 float32
    final_dtype = np.float32

    # 2. 创建一个内存映射文件 (memmap)
    # 这会在磁盘上创建一个指定大小的文件，但只将访问的部分加载到内存
    temp_file = output_file.replace('.npz', '_temp.npy')
    if os.path.exists(temp_file):
        os.remove(temp_file) # 确保每次都是新文件

    # 使用 'w+' 模式创建可读写的memmap对象
    memmapped_array = np.memmap(temp_file, dtype=final_dtype, mode='w+', shape=final_shape)

    # 3. 分块读写数据
    for i in range(0, total_days, chunk_size):
        chunk_start = start_idx + i
        chunk_end = min(start_idx + i + chunk_size, end_idx)
        actual_chunk_size = chunk_end - chunk_start

        print(f"  处理第 {chunk_start}-{chunk_end} 天 ({i + actual_chunk_size}/{total_days})...")

        # 读取数据块
        sst_chunk = read_nc_chunk(input_file, chunk_start, chunk_end)

        # 直接使用纬度索引筛选北半球
        sst_chunk_nh = sst_chunk[:, nh_lat_indices, :]

        # 处理NaN并应用mask
        sst_chunk_nh = np.nan_to_num(sst_chunk_nh, nan=0.0)
        sst_chunk_nh = sst_chunk_nh * mask[np.newaxis, :, :]

        # 更新统计信息
        ocean_mask = (mask == 1)
        ocean_vals = sst_chunk_nh[:, ocean_mask]
        if ocean_vals.size > 0:
            min_sst = min(min_sst, np.min(ocean_vals))
            max_sst = max(max_sst, np.max(ocean_vals))

        # 4. 直接将处理好的数据块写入内存映射文件的正确位置
        memmapped_array[i : i + actual_chunk_size, :, :] = sst_chunk_nh.astype(final_dtype)
        memmapped_array.flush() # 将更改写入磁盘

        # 清理内存
        del sst_chunk, sst_chunk_nh, ocean_vals
        gc.collect()

    # 5. 保存最终的 .npz 文件
    print(f"  最终保存到 {output_file}...")
    # 从 memmap 文件中读取最终数据并保存
    np.savez(output_file, sst=memmapped_array, lon=lon, lat=lat)

    # 6. 清理
    del memmapped_array # 关闭 memmap 对象
    gc.collect()
    os.remove(temp_file) # 删除临时文件

    print(f"  ✅ 完成! 形状: {final_shape}")

    stats = {
        'shape': final_shape,
        'min': min_sst if min_sst != float('inf') else 0,
        'max': max_sst if max_sst != float('-inf') else 0
    }

    return stats