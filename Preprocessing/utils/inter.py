import netCDF4
import numpy as np

from Preprocessing.utils.fc_inter import idw_with_mask_and_radius
from Preprocessing.utils.polar_convert import polar_lonlat_to_xy

# 加载插值格网
ob_x = np.load(r'grid.npz')['x']  # 单位 m
ob_y = np.load(r'grid.npz')['y']
ob_lon = np.load(r'grid.npz')['lon']  # 单位 度数
ob_lat = np.load(r'grid.npz')['lat']

ob_xx, ob_yy = np.meshgrid(ob_x, ob_y)

# 读取CMIP6格网信息和数据
file = '../CMIP6/GFDL-CM4/siconc_SImon_GFDL-CM4_historical_r1i1p1f1_gn_195001-201412.nc'
cmip_dataset = netCDF4.Dataset(f'{file}', 'r')

sic = cmip_dataset.variables['siconc'][:]
sic = sic[-432:]
lon = cmip_dataset.variables['lon'][:]
lat = cmip_dataset.variables['lat'][:]

# ll2xy
length_y = lon.shape[0]
length_x = lon.shape[1]
xx = np.zeros((length_y, length_x))
yy = np.zeros((length_y, length_x))
for i in range(length_y):
    for j in range(length_x):
        [xx[i, j], yy[i, j]] = polar_lonlat_to_xy(lon[i, j], lat[i, j])

xx *= 1000
yy *= 1000

# 读取掩码
mask = np.load(r'../observation/data1/mask.npy')

print(">"*20 + "基础数据加载完毕,开始插值" + "<"*20)

# 开始插值
interpolated = idw_with_mask_and_radius(
    ob_xx, ob_yy,
    xx, yy,
    sic,
    initial_radius=75000,
    max_radius=150000,
    mask=mask,
    expand_ratio=1.5,
    p=2
)
interpolated = np.where(mask == 1, interpolated, 0)
interpolated = np.where(np.isnan(interpolated), 0, interpolated)

print(">"*20 + "历史数据插值完成！" + "<"*20)

file = '../CMIP6/GFDL-CM4/siconc_SImon_GFDL-CM4_ssp245_r1i1p1f1_gn_201501-210012.nc'
cmip_dataset = netCDF4.Dataset(f'{file}', 'r')
sic = cmip_dataset.variables['siconc'][:]
sic = sic[:, 504:, :]

print(">"*20 + "ssp245基础数据加载完毕,开始插值！" + "<"*20)

interpolated_245 = idw_with_mask_and_radius(
    ob_xx, ob_yy,
    xx, yy,
    sic,
    initial_radius=70000,
    max_radius=140000,
    mask=mask,
    expand_ratio=1.5,
    p=2
)
interpolated_245 = np.where(mask == 1, interpolated_245, 0)
interpolated_245 = np.where(np.isnan(interpolated_245), 0, interpolated_245)

print(">"*20 + "ssp245数据插值完成！" + "<"*20)

# file = '../CMIP6/GFDL-CM4/siconc_SImon_GFDL-CM4_ssp585_r1i1p1f1_gn_201501-210012.nc'
# cmip_dataset = netCDF4.Dataset(f'{file}', 'r')
# sic = cmip_dataset.variables['siconc'][:]
# sic = sic[:, 504:, :]
#
# print(">"*20 + "ssp585基础数据加载完毕,开始插值！" + "<"*20)
#
# interpolated_585 = idw_with_mask_and_radius(
#     ob_xx, ob_yy,
#     xx, yy,
#     sic,
#     initial_radius=75000,
#     max_radius=150000,
#     mask=mask,
#     expand_ratio=1.5,
#     p=2
# )
# interpolated_585 = np.where(mask == 1, interpolated_585, 0)
# interpolated_585 = np.where(np.isnan(interpolated_585), 0, interpolated_585)
#
# print(">"*20 + "ssp585数据插值完成！" + "<"*20)
#
# ssp245_train = np.vstack((interpolated, interpolated_245[0:120]))
# ssp585_train = np.vstack((interpolated, interpolated_585[0:120]))
# ssp245_pre = interpolated_245[120:]
# ssp585_pre = interpolated_585[120:]
#
# if ssp245_train.shape[0] != 552:
#     raise ValueError(f"数量错误：期望 552，实际 {ssp245_train.shape[0]}")
#
# np.savez('./GFDL-CM4/ssp245_train.npz', sic=ssp245_train / 100, lon=ob_lon, lat=ob_lat)
# np.savez('./GFDL-CM4/ssp585_train.npz', sic=ssp585_train / 100, lon=ob_lon, lat=ob_lat)
# np.savez('./GFDL-CM4/ssp245_pre.npz', sic=ssp245_pre / 100, lon=ob_lon, lat=ob_lat)
# np.savez('./GFDL-CM4/ssp585_pre.npz', sic=ssp585_pre / 100, lon=ob_lon, lat=ob_lat)
