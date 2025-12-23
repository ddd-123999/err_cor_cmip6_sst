import numpy as np
import os
from datetime import date
from Preprocessing.utils.filter_northern_hemisphere import filter_northern_hemisphere
from Preprocessing.utils.process_and_save import load_nc_meta, process_and_save_chunk

# ---------------- 文件路径 ----------------
files = {
    "hist": r"F:/SST/CMIP6/EC-Earth3/interpolated/sst_EC-Earth3_historical_interp_19820101-20141231.nc",
    "ssp245": r"F:/SST/CMIP6/EC-Earth3/interpolated/sst_EC-Earth3_ssp245_interp_20150101-21001231.nc",
}

# ---------------- 读取元数据 ----------------
print("读取元数据...")
sst_hist_first, lon_hist, lat_hist, hist_len = load_nc_meta(files["hist"])
_, lon_245, lat_245, ssp245_len = load_nc_meta(files["ssp245"])

print(f"历史数据长度: {hist_len} 天")
print(f"SSP245数据长度: {ssp245_len} 天")

# ---------------- 筛选北极区（只用第一个时间步） ----------------
print("\n筛选北极区数据...")
print(f"经度维度: {lon_hist.shape}, 纬度维度: {lat_hist.shape}")

sst_hist_first_nh, lon_hist_nh, lat_hist_nh = filter_northern_hemisphere(
    sst_hist_first, lon_hist, lat_hist
)

# 获取北极区的纬度索引（用于后续批量筛选）
if lat_hist.ndim == 1:
    # 一维纬度数组
    nh_lat_indices = np.where(lat_hist >= 66)[0]
    print(f"北极区纬度索引范围: {nh_lat_indices[0]} 到 {nh_lat_indices[-1]}")
else:
    # 二维纬度数组
    nh_lat_indices = np.where(lat_hist[:, 66] >= 0)[0]
    print(f"北极区纬度索引范围: {nh_lat_indices[0]} 到 {nh_lat_indices[-1]}")

# ---------------- 加载或创建mask ----------------
print("\n加载观测数据的mask...")
mask_path = "../observation/obs/mask.npy"
if os.path.exists(mask_path):
    mask = np.load(mask_path)
    print(f"✅ 成功加载观测mask: {mask_path}")
else:
    print(f"⚠️ 未找到观测mask文件，将根据历史数据创建新mask")
    mask = np.where(~np.isnan(sst_hist_first_nh[0]), 1, 0)
    mask = mask.astype(np.int8)

print(f"Mask形状: {mask.shape}")
print(f"有效数据点数: {np.sum(mask)}")

# ---------------- 计算时间划分 ----------------
print("\n计算时间划分...")
train_days = (date(2015, 1, 1) - date(1982, 1, 1)).days  # 12053
val_days = (date(2020, 1, 1) - date(2015, 1, 1)).days    # 1826
test_days = (date(2025, 1, 1) - date(2020, 1, 1)).days   # 1827

print(f"训练集: {train_days}天, 验证集: {val_days}天, 测试集: {test_days}天")

# 确定每个数据集对应的文件和时间索引
# 训练集: hist[0:12053]
# 验证集: ssp[0:1827]
# 测试集: ssp[1827:3654]
# 预测集: ssp[3654:]

# ---------------- 创建输出目录 ----------------
os.makedirs("./EC-Earth3", exist_ok=True)

# ---------------- 处理SSP245数据集 ----------------
print("\n" + "=" * 60)
print("开始处理 SSP245 数据集")
print("=" * 60)

# 训练集（历史数据）
stats_245_train = process_and_save_chunk(
    files["hist"], 0, train_days,
    "./EC-Earth3/ssp245_train.npz",
    lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
)

# 验证集
stats_245_val = process_and_save_chunk(
    files["ssp245"], 0, val_days,
    "./EC-Earth3/ssp245_val.npz",
    lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
)

# 测试集
stats_245_test = process_and_save_chunk(
    files["ssp245"], val_days, val_days + test_days,
    "./EC-Earth3/ssp245_test.npz",
    lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
)

# 预测集
stats_245_pre = process_and_save_chunk(
    files["ssp245"], val_days + test_days, ssp245_len,
    "./EC-Earth3/ssp245_pre.npz",
    lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
)

# ---------------- 输出最终统计信息 ----------------
print("\n" + "=" * 60)
print("🎉 数据集划分完成！")
print("=" * 60)
print(f"训练集: {stats_245_train['shape'][0]} 天 (1982-01-01 至 2014-12-31)")
print(f"验证集: {stats_245_val['shape'][0]} 天 (2015-01-01 至 2019-12-31)")
print(f"测试集: {stats_245_test['shape'][0]} 天 (2020-01-01 至 2024-12-31)")
print(f"预测集: {stats_245_pre['shape'][0]} 天 (2025-01-01 至 2100-12-31)")
print(f"北极区空间维度: {stats_245_train['shape'][1:]}")
print(f"Mask形状: {mask.shape}")
print(f"有效数据点数: {np.sum(mask)}")

print(f"\n纬度范围: {np.min(lat_hist_nh):.2f}° 到 {np.max(lat_hist_nh):.2f}°")
print(f"训练集海洋SST范围: {stats_245_train['min']:.2f}°C 到 {stats_245_train['max']:.2f}°C")

print("\n数据已保存到:")
print("  - ./EC-Earth3_xy/ssp245_*.npz")
print("=" * 60)