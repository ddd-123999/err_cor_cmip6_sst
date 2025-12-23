import netCDF4
import numpy as np

def check_if_regular_grid(nc_path):
    ds = netCDF4.Dataset(nc_path, "r")

    # 尝试读取经纬度变量
    if "lon" in ds.variables:
        lon = ds.variables["lon"][:]
    elif "longitude" in ds.variables:
        lon = ds.variables["longitude"][:]
    else:
        print("⚠️ 未找到经度变量 (lon / longitude)")
        return False

    if "lat" in ds.variables:
        lat = ds.variables["lat"][:]
    elif "latitude" in ds.variables:
        lat = ds.variables["latitude"][:]
    else:
        print("⚠️ 未找到纬度变量 (lat / latitude)")
        return False

    # 判断维度是否为一维
    if lon.ndim != 1 or lat.ndim != 1:
        print("❌ 经纬度不是一维数组，可能是不规则（curvilinear）网格。")
        return False

    # 检查间隔是否均匀
    lon_diff = np.diff(lon)
    lat_diff = np.diff(lat)
    lon_uniform = np.allclose(lon_diff, lon_diff[0])
    lat_uniform = np.allclose(lat_diff, lat_diff[0])

    if lon_uniform and lat_uniform:
        print("✅ 检测结果：规则格网（regular lat-lon grid）")
        print(f"  经度范围: {lon[0]:.2f}° ~ {lon[-1]:.2f}°，间隔 {lon_diff[0]:.2f}°")
        print(f"  纬度范围: {lat[0]:.2f}° ~ {lat[-1]:.2f}°，间隔 {lat_diff[0]:.2f}°")
        return True
    else:
        print("⚠️ 经纬度间隔不均匀，可能是不规则网格。")
        print(f"  经度间隔范围: {lon_diff.min():.4f} ~ {lon_diff.max():.4f}")
        print(f"  纬度间隔范围: {lat_diff.min():.4f} ~ {lat_diff.max():.4f}")
        return False


check_if_regular_grid("/Preprocessing/observation/obs/oisst_daily_19820101-20241231.nc")