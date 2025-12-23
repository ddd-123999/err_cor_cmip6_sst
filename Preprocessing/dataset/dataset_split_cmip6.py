import numpy as np
import os
from datetime import date

# 假设您的工具函数在您运行环境的 PYTHONPATH 中
# 或者 'Preprocessing' 文件夹与您的脚本在同一级
try:
    from Preprocessing.utils.filter_northern_hemisphere import filter_northern_hemisphere
    from Preprocessing.utils.process_and_save import load_nc_meta, process_and_save_chunk
except ImportError:
    print("警告: 无法导入 'Preprocessing.utils'。")
    print("请确保您的目录结构正确，或已安装 'Preprocessing' 包。")
    # 允许脚本继续运行，以便 batch_dataset_processing.py 可以导入此文件
    # 但如果调用 split_model_data，将会失败。
    pass


def split_model_data(model_name, hist_file_path, ssp_file_path, base_output_dir, mask_path):
    """
    为单个模型执行数据集划分。

    参数:
    model_name (str): 模型的名称 (例如 'EC-Earth3')
    hist_file_path (str): 历史数据 .nc 文件的完整路径
    ssp_file_path (str): ssp245 .nc 文件的完整路径
    base_output_dir (str): 保存 .npz 文件的根目录 (例如 './model_npz_raw')
    mask_path (str): 'mask.npy' 文件的路径
    """

    # ---------------- 文件路径 ----------------
    files = {
        "hist": hist_file_path,
        "ssp245": ssp_file_path,
    }

    # ---------------- 读取元数据 ----------------
    print("  读取元数据...")
    sst_hist_first, lon_hist, lat_hist, hist_len = load_nc_meta(files["hist"])
    _, lon_245, lat_245, ssp245_len = load_nc_meta(files["ssp245"])

    print(f"  历史数据长度: {hist_len} 天")
    print(f"  SSP245数据长度: {ssp245_len} 天")

    # ---------------- 筛选北极区（只用第一个时间步） ----------------
    print("\n  筛选北极区数据...")
    print(f"  经度维度: {lon_hist.shape}, 纬度维度: {lat_hist.shape}")

    sst_hist_first_nh, lon_hist_nh, lat_hist_nh = filter_northern_hemisphere(
        sst_hist_first, lon_hist, lat_hist
    )

    # 获取北极区的纬度索引（用于后续批量筛选）
    if lat_hist.ndim == 1:
        # 一维纬度数组
        nh_lat_indices = np.where(lat_hist >= 66)[0]
        print(f"  北极区纬度索引范围: {nh_lat_indices[0]} 到 {nh_lat_indices[-1]}")
    else:
        # 二维纬度数组
        # 假设 (lon, lat) 或 (x, y) 维度
        lat_to_check = lat_hist[0, :] if lat_hist.shape[1] > lat_hist.shape[0] else lat_hist[:, 0]
        nh_lat_indices = np.where(lat_to_check >= 0)[0]  # 假设北极区是所有 j 索引
        print(f"  北极区纬度索引范围: {nh_lat_indices[0]} 到 {nh_lat_indices[-1]} (假设)")

    # ---------------- 加载或创建mask ----------------
    print("\n  加载观测数据的mask...")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        print(f"  ✅ 成功加载观测mask: {mask_path}")
    else:
        print(f"  ⚠️ 未找到观测mask文件: {mask_path}")
        print("     将根据当前模型的历史数据创建新mask")
        mask = np.where(~np.isnan(sst_hist_first_nh[0]), 1, 0)
        mask = mask.astype(np.int8)

    print(f"  Mask形状: {mask.shape}")
    print(f"  有效数据点数: {np.sum(mask)}")

    # ---------------- 计算时间划分 ----------------
    # (这些日期是固定的，适用于所有模型)
    print("\n  计算时间划分...")
    train_days = (date(2015, 1, 1) - date(1982, 1, 1)).days  # 12053
    val_days = (date(2020, 1, 1) - date(2015, 1, 1)).days  # 1826
    test_days = (date(2025, 1, 1) - date(2020, 1, 1)).days  # 1827

    print(f"  训练集: {train_days}天, 验证集: {val_days}天, 测试集: {test_days}天")

    # ---------------- 创建输出目录 ----------------
    output_dir = os.path.join(base_output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- 处理SSP245数据集 ----------------
    print("\n" + "=" * 40)
    print(f"  开始处理 {model_name} SSP245 数据集")
    print("=" * 40)

    # 训练集（历史数据）
    stats_245_train = process_and_save_chunk(
        files["hist"], 0, train_days,
        os.path.join(output_dir, "ssp245_train.npz"),
        lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
    )

    # 验证集
    stats_245_val = process_and_save_chunk(
        files["ssp245"], 0, val_days,
        os.path.join(output_dir, "ssp245_val.npz"),
        lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
    )

    # 测试集
    stats_245_test = process_and_save_chunk(
        files["ssp245"], val_days, val_days + test_days,
        os.path.join(output_dir, "ssp245_test.npz"),
        lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
    )

    # 预测集
    stats_245_pre = process_and_save_chunk(
        files["ssp245"], val_days + test_days, ssp245_len,
        os.path.join(output_dir, "ssp245_pre.npz"),
        lon_hist_nh, lat_hist_nh, mask, nh_lat_indices
    )

    # ---------------- 输出最终统计信息 ----------------
    print("\n" + "=" * 40)
    print(f"  🎉 {model_name} 数据集划分完成！")
    print("=" * 40)
    print(f"  训练集: {stats_245_train['shape'][0]} 天 (1982-01-01 至 2014-12-31)")
    print(f"  验证集: {stats_245_val['shape'][0]} 天 (2015-01-01 至 2019-12-31)")
    print(f"  测试集: {stats_245_test['shape'][0]} 天 (2020-01-01 至 2024-12-31)")
    print(f"  预测集: {stats_245_pre['shape'][0]} 天 (2025-01-01 至 2100-12-31)")
    print(f"  北极区空间维度: {stats_245_train['shape'][1:]}")
    print(f"  Mask形状: {mask.shape}")
    print(f"  有效数据点数: {np.sum(mask)}")

    print(f"\n  纬度范围: {np.min(lat_hist_nh):.2f}° 到 {np.max(lat_hist_nh):.2f}°")
    print(f"  训练集海洋SST范围: {stats_245_train['min']:.2f}°C 到 {stats_245_train['max']:.2f}°C")

    print("\n  数据已保存到:")
    print(f"    {output_dir}")
    print("=" * 40)

    return output_dir  # 返回创建的目录，供下一步使用


# 允许此脚本被导入，或被独立运行（用于测试）
if __name__ == "__main__":
    pass
    # print("--- 正在以独立模式测试 dataset_split.py ---")
    #
    # # --- !! 测试配置 !! ---
    # TEST_MODEL = "EC-Earth3"
    # TEST_HIST_FILE = r"F:/CMIP6/EC-Earth3/interpolated/sst_EC-Earth3_historical_interp_19820101-20141231.nc"
    # TEST_SSP_FILE = r"F:/CMIP6/EC-Earth3/interpolated/sst_EC-Earth3_ssp245_interp_20150101-21001231.nc"
    # TEST_BASE_OUTPUT = r"./"
    # TEST_MASK_PATH = r"../observation/mask.npy"
    # # -------------------------
    #
    # if not os.path.exists(TEST_HIST_FILE) or not os.path.exists(TEST_SSP_FILE):
    #     print("测试文件未找到，跳过独立运行。")
    # elif not os.path.exists(TEST_MASK_PATH):
    #     print(f"测试Mask文件 '{TEST_MASK_PATH}' 未找到，跳过。")
    # else:
    #     split_model_data(
    #         TEST_MODEL,
    #         TEST_HIST_FILE,
    #         TEST_SSP_FILE,
    #         TEST_BASE_OUTPUT,
    #         TEST_MASK_PATH
    #     )
