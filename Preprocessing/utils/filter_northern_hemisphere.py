import numpy as np


def filter_northern_hemisphere(sst, lon, lat):
    """
    筛选北极区数据 - 针对二维坐标的规则格网

    参数:
        sst: SST数据，形状为 [time, lat, lon] 或 [time, lon, lat]
        lon: 经度数据，可以是一维或二维
        lat: 纬度数据，可以是一维或二维

    返回:
        sst_nh: 北极区的SST数据
        lon_nh: 北极区的经度数据
        lat_nh: 北极区的纬度数据
    """
    # 初始化变量
    sst_nh = sst
    lon_nh = lon
    lat_nh = lat

    # 对于二维坐标的规则格网,我们基于纬度值创建掩码
    if len(lat.shape) == 2:  # 二维纬度
        print("检测到二维坐标，使用纬度筛选")

        # 创建北极区掩码 (纬度 >= 0)
        # 由于是规则格网，我们可以用第一列来判断纬度值
        nh_mask = lat[:, 0] >= 66  # 取第一列的纬度值判断

        print(f"北极区掩码True数量: {np.sum(nh_mask)} / {len(nh_mask)}")

        # 应用筛选
        # SST数据维度: [time, lat, lon] -> 筛选lat维度
        sst_nh = sst[:, nh_mask, :]

        # 坐标数据维度: [lat, lon] -> 筛选lat维度
        lon_nh = lon[nh_mask, :]
        lat_nh = lat[nh_mask, :]

    elif len(lat.shape) == 1:  # 一维纬度
        print("检测到一维坐标，使用纬度筛选")
        nh_mask = lat >= 66
        lat_nh = lat[nh_mask]

        # 根据SST数据的维度应用筛选
        if len(sst.shape) == 3:
            if sst.shape[1] == len(lat):  # [time, lat, lon]
                sst_nh = sst[:, nh_mask, :]
                lon_nh = lon  # 经度不变
            elif sst.shape[2] == len(lat):  # [time, lon, lat]
                sst_nh = sst[:, :, nh_mask]
                lon_nh = lon  # 经度不变

    print(f"筛选后维度 - SST: {sst_nh.shape}, lon: {lon_nh.shape}, lat: {lat_nh.shape}")
    if len(lat_nh.shape) == 2:
        print(f"纬度范围: {np.min(lat_nh):.2f} 到 {np.max(lat_nh):.2f}")
    elif len(lat_nh.shape) == 1:
        print(f"纬度范围: {np.min(lat_nh):.2f} 到 {np.max(lat_nh):.2f}")

    return sst_nh, lon_nh, lat_nh