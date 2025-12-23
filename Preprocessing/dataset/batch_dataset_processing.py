import os
import sys
from datetime import datetime

# 导入我们重构的函数
# 确保 dataset_split.py 和 normalize_dataset.py 在同一个文件夹中
try:
    from dataset_split_cmip6 import split_model_data
    from normalize_dataset_cmip6 import normalize_model_data
except ImportError:
    print("错误: 无法导入 'dataset_split.py' 或 'normalize_dataset.py'。")
    print("请确保 batch_dataset_processing.py, dataset_split.py, 和 normalize_dataset.py "
          "位于同一个文件夹中。")
    sys.exit(1)

# ========== 1. 配置 ==========

# 定义所有要处理的模型
# (这是您上次从图片中生成的列表)
MODELS_TO_PROCESS = [
    'ACCESS-ESM1-5',
    'EC-Earth3',
    'EC-Earth3-CC',
    'EC-Earth3-veg',
    'EC-Earth3-Veg-LR',
    'IPSL-CM6A-LR',
    'MIROC6',
    'MPI-ESM1-2-HR',
    'MPI-ESM1-2-LR',
    'MRI-ESM2-0',
    'NESM3',
    'BCC-CSM2-MR',
    'CanESM5',
    'CESM2-WACCM',
    'CMCC-CM2-SR5',
    'CMCC-ESM2',
    'NorESM2-LM',
    'NorESM2-MM',
    'GFDL-CM4',
    'GFDL-ESM4'
]

# 包含所有模型插值后 .nc 文件的基础目录
BASE_NC_DIR = r"F:\CMIP6"

# 您希望保存 .npz 文件的基础目录
# (脚本会在此目录下创建子文件夹, e.g., ./model_npz_raw/EC-Earth3)
BASE_NPZ_OUTPUT_DIR = r"./"

# mask 文件的路径 (请确保此路径相对于您运行此脚本的位置是正确的)
MASK_PATH = r"../observation/obs/mask.npy"

# ========== 2. 批量处理循环 ==========

print("=" * 60)
print(f"开始批量处理 {len(MODELS_TO_PROCESS)} 个模型...")
print(f"数据源 (NC): {BASE_NC_DIR}")
print(f"输出 (NPZ): {BASE_NPZ_OUTPUT_DIR}")
print(f"Mask: {MASK_PATH}")
print("=" * 60)

os.makedirs(BASE_NPZ_OUTPUT_DIR, exist_ok=True)
models_failed = []
models_succeeded = []

for model_name in MODELS_TO_PROCESS:
    print(f"\n--- [{MODELS_TO_PROCESS.index(model_name) + 1}/{len(MODELS_TO_PROCESS)}] "
          f"开始处理: {model_name} ---")

    start_time = datetime.now()

    # 1. 构建文件路径
    # (这些路径基于您在MATLAB脚本中使用的插值后文件名)
    hist_file = os.path.join(
        BASE_NC_DIR, model_name, "interpolated",
        f"sst_{model_name}_historical_interp_19820101-20141231.nc"
    )
    ssp_file = os.path.join(
        BASE_NC_DIR, model_name, "interpolated",
        f"sst_{model_name}_ssp245_interp_20150101-21001231.nc"
    )

    # 2. 检查 .nc 文件是否存在
    if not os.path.exists(hist_file) or not os.path.exists(ssp_file):
        print(f"  ❌ 失败: 找不到 .nc 文件。")
        print(f"     - 检查: {hist_file}")
        print(f"     - 检查: {ssp_file}")
        models_failed.append(f"{model_name} (文件缺失)")
        continue

    try:
        # 3. 步骤 1: 划分数据集
        print(f"\n  [1/2] 正在划分 {model_name}...")

        # 调用重构的函数
        # 它将创建 e.g. "./model_npz_raw/EC-Earth3" 目录
        model_raw_dir = split_model_data(
            model_name=model_name,
            hist_file_path=hist_file,
            ssp_file_path=ssp_file,
            base_output_dir=BASE_NPZ_OUTPUT_DIR,
            mask_path=MASK_PATH
        )
        print(f"  ✅ 划分完成，原始数据保存于: {model_raw_dir}")

        # 4. 步骤 2: 归一化数据集
        print(f"\n  [2/2] 正在归一化 {model_name}...")

        # 调用重构的函数
        # 它将读取 "./model_npz_raw/EC-Earth3"
        # 并写入 "./model_npz_raw/EC-Earth3_normalized"
        normalized_output_dir = normalize_model_data(
            original_dir=model_raw_dir,
            mask_path=MASK_PATH
        )
        print(f"  ✅ 归一化完成，数据保存于: {normalized_output_dir}")

        models_succeeded.append(model_name)

    except Exception as e:
        print(f"  ❌ 失败: 处理 {model_name} 时发生严重错误: {e}")
        import traceback

        traceback.print_exc()
        models_failed.append(f"{model_name} (运行时错误)")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"  🕒 {model_name} 处理完毕，耗时 {duration:.1f} 秒。")

# ========== 3. 最终总结 ==========
print("\n" + "=" * 60)
print("🎉 批量处理全部完成！")
print("=" * 60)
print(f"  ✅ 成功: {len(models_succeeded)}")
print(f"  ❌ 失败: {len(models_failed)}")
if models_failed:
    print("\n失败的模型列表:")
    for model in models_failed:
        print(f"  - {model}")
print("=" * 60)
