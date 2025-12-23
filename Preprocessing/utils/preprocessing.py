import os
import netCDF4
import numpy as np

# 从 polar_convert.py 导入函数
from Preprocessing.utils.polar_convert import polar_ij_to_lonlat

# --- 修改开始 ---
# 1. 移除文件列表循环，直接指定您的文件名
# 假设您的文件与此脚本位于同一目录或正确的相对路径中
file_path = 'E:/SST/sic_psn25_197901_n07_v05r00.nc'
print(f"正在处理单个文件: {file_path}")

# 2. 打开 NetCDF 文件
try:
    dataset = netCDF4.Dataset(file_path, 'r')
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
    exit()

# 3. 获取坐标 x 和 y
# (与 create_grid.m 一致)
try:
    x = dataset.variables['x'][:]  # 1D 投影坐标 (m)
    y = dataset.variables['y'][:]  # 1D 投影坐标 (m)
    print("成功读取 'x' 和 'y' 坐标。")
except KeyError:
    print("错误：文件中未找到 'x' 或 'y' 变量。")
    dataset.close()
    exit()

# 4. 获取掩码
# 原始脚本查找 'surface_type_mask'。
# 您的文件可能使用不同的名称，例如 'mask' 或 'land'。
# 我们在这里尝试读取 'mask'。
try:
    # 假设 'mask' 变量存在于您的文件中
    # 并且 1 表示有效区域 (非陆地)
    mask = dataset.variables['mask'][:]
    # 确保掩码是 (y, x) 形状
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    # NSIDC 掩码通常 0=陆地, 1=海洋/冰。如果您的掩码定义相反，请调整。
    print("成功读取 'mask' 变量。")
except KeyError:
    print("警告：文件中未找到 'mask' 变量。")
    print("将创建一个空的（全 True）掩码。")
    # 如果找不到掩码，创建一个全为1的掩码 (y, x)
    mask = np.ones((len(y), len(x)), dtype=int)

# 5. 移除 SIC 数据堆叠逻辑
# 您的文件只是网格文件，不包含 'cdr_seaice_conc_monthly' 数据。
# 我们将只保存网格信息。
print("已跳过 SIC 数据读取。")

dataset.close()
# --- 修改结束 ---


# --- xy2ll (此部分保持不变) ---
# 此部分使用格网 *索引* (i, j) 来计算 lon/lat
# 它依赖于 x 和 y 的长度来确定网格大小
print("正在从格网索引计算经纬度 (xy2ll)...")
length_x = len(x)
length_y = len(y)

lon = np.zeros((length_y, length_x))
lat = np.zeros((length_y, length_x))

# 循环 (i, j) 索引
# i 对应 x 轴 (1 到 length_x)
# j 对应 y 轴 (1 到 length_y)
for i in range(length_x):  # 对应 i
    for j in range(length_y): # 对应 j
        # polar_ij_to_lonlat 使用 1-based 索引
        # (j, i) 顺序匹配 (y, x) 布局
        lon[j][i], lat[j][i] = polar_ij_to_lonlat(i + 1, j + 1, 25, 'NORTH')

print("经纬度计算完成。")


# --- 保存数据 (修改) ---
# 6. 保存网格数据
# inter.py 需要 'lon', 'lat', 'x', 'y'
# 我们不保存 'sic' 变量，因为它不存在
np.savez('grid.npz', lon=lon, lat=lat, x=x, y=y)
print("已保存 'grid.npz' (仅含网格信息)。")

# 7. 保存掩码
np.save('mask.npy', mask)
print("已保存 'mask.npy'。")
print("处理完成。")
# # 将所有SIC数据整合到一个文件夹中，获取mask
# # 将xy转化为ll
#
# import os
# import netCDF4
# import numpy as np
#
# from Preprocessing.utils.polar_convert import polar_ij_to_lonlat
#
# # 获取文件名称
# directory = '../data0/'
# extension = '.nc'
# file_name = [f for f in os.listdir(directory) if f.endswith(extension)]
#
# # 获取SIC、mask、xy
# SIC = []
# for i, file in enumerate(file_name):
#     dataset = netCDF4.Dataset(f'../data0/{file}', 'r')
#     sic = dataset.variables['cdr_seaice_conc_monthly'][:]
#     if i != 0:
#         SIC = np.vstack((SIC, sic))
#         if i == 551:
#             group = dataset.groups['cdr_supplementary']
#             mask = group.variables['surface_type_mask'][:]
#             mask = np.where(np.logical_or(mask == 50, mask == 100), 1, 0)
#             mask = np.squeeze(mask)
#     else:
#         SIC = sic
#         x = dataset.variables['x'][:]
#         y = dataset.variables['y'][:]
#
# SIC = SIC * mask
#
# # xy2ll
# length_x = len(x)
# length_y = len(y)
#
# lon = np.zeros((length_y, length_x))
# lat = np.zeros((length_y, length_x))
# for i in range(length_x):
#     for j in range(length_y):
#         lon[j][i], lat[j][i] = polar_ij_to_lonlat(i + 1, j + 1, 25, 'NORTH')
#
# # 保存数据
# np.savez('sic.npz', lon=lon, lat=lat, x=x, y=y, sic=SIC)
# np.save('mask.npy', mask)
#
#
# # import matplotlib
# # matplotlib.use('TkAgg')
# # import matplotlib.pyplot as plt
# # import cartopy.crs as ccrs
# # import cartopy.feature as cfeature
# #
# # # 数据预处理
# # SIC = SIC[1]
# #
# # # 1. 创建画布
# # fig = plt.figure(figsize=(8, 8))
# # proj = ccrs.NorthPolarStereo(central_longitude=-45)
# # ax = plt.axes(projection=proj)
# #
# # # 3. 设置地图显示范围
# # ax.set_extent([-3580000, 3750000, -5350000, 5850000], crs=proj)
# #
# # # 4. 添加地图要素
# # ax.coastlines(resolution='110m')
# # ax.gridlines(draw_labels=False)
# # ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
# # ax.add_feature(cfeature.OCEAN, zorder=0)
# #
# # # 5. 画图
# # # c = ax.pcolormesh(x, y, SIC, cmap='viridis', vmin=0, vmax=1)  # xy
# # c = ax.pcolormesh(lon, lat, SIC, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=1)  # ll
# # plt.colorbar(c, orientation='horizontal', pad=0.05, label='Sea Ice Concentration')
# #
# # plt.title('Sea Ice Concentration (North Polar Projection)')
# # plt.tight_layout()
# # plt.show()
