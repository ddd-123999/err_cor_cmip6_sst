import numpy as np
import os
from datetime import datetime
from Project.utils.metric import metric

# ========== 配置 ==========

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

# 文件路径
MASK_PATH = '../../Preprocessing/observation/obs/mask.npy'
OB_DATA_PATH = '../../Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz'
CMIP_DATA_DIR = '../../Preprocessing/dataset'
OUTPUT_TXT = './base_metrics_results.txt'
OUTPUT_NPY = './base_metrics_all.npy'

# SSP场景（如果有多个场景可以添加）
SSP_SCENARIO = 'ssp245_test'

# ========== 加载观测数据和mask ==========

print("=" * 60)
print("批量计算Base Metrics")
print("=" * 60)

print("\n加载观测数据和mask...")
mask = np.load(MASK_PATH)
ob_sic = np.load(OB_DATA_PATH)['sst'][-1827:]  # 取最后1827天
print(f"  ✓ Mask形状: {mask.shape}")
print(f"  ✓ 观测数据形状: {ob_sic.shape}")

# 统计观测数据中的海洋点数量
ob_ocean_points = np.sum(~np.isnan(ob_sic) & (mask == 1))
print(f"  ✓ 观测数据中海洋点数量: {ob_ocean_points}")

# ========== 批量处理 ==========

all_results = {}
models_failed = []
models_succeeded = []

# 打开txt文件准备写入
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    # 写入文件头
    f.write("=" * 80 + "\n")
    f.write("Base Metrics 计算结果\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"观测数据: {OB_DATA_PATH}\n")
    f.write(f"观测数据时间范围: 最后1827天\n")
    f.write(f"观测数据海洋点数量: {ob_ocean_points}\n")
    f.write("=" * 80 + "\n\n")

    # 写入表头
    f.write(
        f"{'模型名称':<25} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'NSE':<10} {'PCC':<10} {'SSIM':<10} {'状态':<10}\n")
    f.write("-" * 115 + "\n")

    for idx, model_name in enumerate(MODELS_TO_PROCESS, 1):
        print(f"\n[{idx}/{len(MODELS_TO_PROCESS)}] 处理: {model_name}")
        start_time = datetime.now()

        try:
            # 构建CMIP数据路径
            cmip_data_path = os.path.join(CMIP_DATA_DIR, model_name, f'{SSP_SCENARIO}.npz')

            # 检查文件是否存在
            if not os.path.exists(cmip_data_path):
                raise FileNotFoundError(f"找不到文件: {cmip_data_path}")

            # 加载CMIP数据
            print(f"  加载: {cmip_data_path}")
            cmip_sic = np.load(cmip_data_path)['sst']

            # 统计CMIP数据中的海洋点数量
            cmip_ocean_points = np.sum(~np.isnan(cmip_sic) & (mask == 1))
            print(f"  CMIP数据中海洋点数量: {cmip_ocean_points}")

            # 计算指标
            print(f"  计算指标...")
            rmse, mae, mse, nse, mean_pcc, mean_ssim = metric(mask, cmip_sic, ob_sic)

            # 保存结果
            results = {
                "rmse": np.round(rmse, 4),
                "mae": np.round(mae, 4),
                "mse": np.round(mse, 4),
                "nse": np.round(nse, 4),
                "mean_pcc": np.round(mean_pcc, 4),
                "mean_ssim": np.round(mean_ssim, 4),
                "ocean_points": cmip_ocean_points
            }

            all_results[model_name] = results
            models_succeeded.append(model_name)

            # 写入txt文件
            f.write(f"{model_name:<25} {results['rmse']:<10.4f} {results['mae']:<10.4f} "
                    f"{results['mse']:<10.4f} {results['nse']:<10.4f} {results['mean_pcc']:<10.4f} "
                    f"{results['mean_ssim']:<10.4f} {'✓':<10}\n")
            f.flush()  # 立即写入文件

            # 打印结果
            print(f"  ✅ 完成:")
            print(f"     RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}, MSE: {results['mse']:.4f}")
            print(f"     NSE: {results['nse']:.4f}, PCC: {results['mean_pcc']:.4f}, SSIM: {results['mean_ssim']:.4f}")

        except FileNotFoundError as e:
            print(f"  ❌ 失败: {e}")
            models_failed.append(f"{model_name} (找不到文件)")
            f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 文件缺失':<10}\n")
            f.flush()

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback

            traceback.print_exc()
            models_failed.append(f"{model_name} (运行时错误)")
            f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 计算错误':<10}\n")
            f.flush()

        duration = (datetime.now() - start_time).total_seconds()
        print(f"  🕒 耗时 {duration:.1f} 秒")

    # 写入统计摘要
    f.write("\n" + "=" * 80 + "\n")
    f.write("统计摘要\n")
    f.write("=" * 80 + "\n")
    f.write(f"总模型数: {len(MODELS_TO_PROCESS)}\n")
    f.write(f"成功: {len(models_succeeded)}\n")
    f.write(f"失败: {len(models_failed)}\n")

    if models_failed:
        f.write("\n失败的模型列表:\n")
        for model in models_failed:
            f.write(f"  - {model}\n")

# ========== 保存完整结果 ==========

# 保存为.npy文件
np.save(OUTPUT_NPY, all_results)
print(f"\n✅ 完整结果已保存到: {OUTPUT_NPY}")
print(f"✅ 文本结果已保存到: {OUTPUT_TXT}")

# ========== 最终总结 ==========

print("\n" + "=" * 60)
print("🎉 批量计算完成！")
print("=" * 60)
print(f"  ✅ 成功: {len(models_succeeded)}")
print(f"  ❌ 失败: {len(models_failed)}")
if models_failed:
    print("\n失败的模型列表:")
    for model in models_failed:
        print(f"  - {model}")
print("=" * 60)
print(f"\n📊 请查看结果文件: {OUTPUT_TXT}")