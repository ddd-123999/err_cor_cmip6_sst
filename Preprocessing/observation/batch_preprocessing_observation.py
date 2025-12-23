import os
import sys
from datetime import datetime

# 导入我们重构的函数
try:
    from preprocessing_cmip6 import normalize_obs_for_model
except ImportError:
    print("错误: 无法导入 'preprocessing_cmip6.py'。")
    print("请确保 'batch_preprocessing_observation.py' 和 'preprocessing_cmip6.py' "
          "位于同一个文件夹中。")
    sys.exit(1)

# ========== 1. 配置 ==========

# 定义所有要处理的模型
MODELS_TO_PROCESS = [
    'ACCESS-CM2',
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

# 包含所有模型归一化参数的目录
BASE_PARAM_DIR = r"../dataset"

# 存放 *未归一化* 观测数据和 mask 的目录
RAW_OBS_OUTPUT_DIR = r"./obs"

# 存放 *已归一化* 的观测数据的最终目录
FINAL_NORMALIZED_DIR = r"./obs_normalized"

# ========== 2. 批量归一化 ==========

print("=" * 60)
print(f"批量归一化观测数据 (共 {len(MODELS_TO_PROCESS)} 个模型)")
print("=" * 60)

# 检查必需的输入文件
raw_sst_path = os.path.join(RAW_OBS_OUTPUT_DIR, 'sst_daily_not_to_be_normalized.npz')
mask_path = os.path.join(RAW_OBS_OUTPUT_DIR, 'mask.npy')

if not os.path.exists(raw_sst_path):
    print(f"  ❌ 错误: 找不到未归一化的观测数据: {raw_sst_path}")
    sys.exit(1)

if not os.path.exists(mask_path):
    print(f"  ❌ 错误: 找不到 mask 文件: {mask_path}")
    sys.exit(1)

print(f"  ✓ 未归一化观测数据: {raw_sst_path}")
print(f"  ✓ Mask文件: {mask_path}")
print()

models_failed = []
models_succeeded = []

for model_name in MODELS_TO_PROCESS:
    print(f"\n--- [{MODELS_TO_PROCESS.index(model_name) + 1}/{len(MODELS_TO_PROCESS)}] "
          f"处理: {model_name} ---")
    start_time = datetime.now()

    try:
        # 定义输出文件名
        output_filename = f"sst_daily_{model_name}.npz"

        normalized_path = normalize_obs_for_model(
            model_name=model_name,
            base_param_dir=BASE_PARAM_DIR,
            raw_data_dir=RAW_OBS_OUTPUT_DIR,
            final_output_dir=FINAL_NORMALIZED_DIR,
            output_filename=output_filename
        )
        print(f"  ✅ 归一化完成，已保存到: {normalized_path}")
        models_succeeded.append(model_name)

    except FileNotFoundError as e:
        print(f"  ❌ 失败: {e}")
        models_failed.append(f"{model_name} (找不到文件)")
    except Exception as e:
        print(f"  ❌ 失败: 处理 {model_name} 时发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        models_failed.append(f"{model_name} (运行时错误)")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"  🕒 耗时 {duration:.1f} 秒。")

# ========== 3. 最终总结 ==========
print("\n" + "=" * 60)
print("🎉 批量归一化全部完成！")
print("=" * 60)
print(f"  ✅ 成功: {len(models_succeeded)}")
print(f"  ❌ 失败: {len(models_failed)}")
if models_failed:
    print("\n失败的模型列表:")
    for model in models_failed:
        print(f"  - {model}")
print("=" * 60)