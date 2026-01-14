import numpy as np
import os
from datetime import datetime
from tqdm import tqdm

# 从上面创建的文件导入
from quantile_mapping import QuantileMappingCorrector, metric

# ====================================================================
# 1. 配置参数
# ====================================================================

# 定义所有要处理的模型
MODELS_TO_PROCESS = [
    'ACCESS-CM2','ACCESS-ESM1-5', 'EC-Earth3', 'EC-Earth3-CC',
    'EC-Earth3-veg', 'EC-Earth3-Veg-LR', 'IPSL-CM6A-LR', 'MIROC6',
    'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3',
    'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5',
    'CMCC-ESM2', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4', 'GFDL-ESM4'
]

# QM方法选择
QM_METHOD = 'edcdf'  # 'empirical' 或 'parametric' 或 'edcdf'
N_QUANTILES = 100  # 分位数个数

# 文件路径
OBS_BASE_DIR = '../../Preprocessing/observation'
CMIP_BASE_DIR = '../../Preprocessing/dataset'

MASK_PATH = os.path.join(OBS_BASE_DIR, 'obs/mask.npy')
OBS_RAW_PATH = os.path.join(OBS_BASE_DIR, 'obs/sst_daily_not_to_be_normalized.npz')
OBS_NORMALIZED_DIR = os.path.join(OBS_BASE_DIR, 'obs_normalized')

# 输出文件
OUTPUT_TXT = f'./qm_{QM_METHOD}_q{N_QUANTILES}_metrics_results.txt'
OUTPUT_NPY = f'./qm_{QM_METHOD}_q{N_QUANTILES}_metrics_all.npy'
OUTPUT_DATA_DIR = f'./qm_{QM_METHOD}_q{N_QUANTILES}_results_data'

# SSP场景
SSP_SCENARIO = 'ssp245'

# ====================================================================
# 2. 批量处理主程序
# ====================================================================

print("=" * 60)
print(f"批量计算 Quantile Mapping ({QM_METHOD.upper()}, Q={N_QUANTILES}) 修正后指标")
print("=" * 60)

print("\n加载全局数据 (Mask, 观测Test)...")
mask = np.load(MASK_PATH)
y_obs_test_raw = np.load(OBS_RAW_PATH)['sst'][-1827:]
print(f"  ✓ Mask形状: {mask.shape}")
print(f"  ✓ 观测Test(Raw)形状: {y_obs_test_raw.shape}")

all_results = {}
models_failed = []
models_succeeded = []

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write(f"Quantile Mapping ({QM_METHOD.upper()}, Q={N_QUANTILES}) 修正后 Metrics 计算结果\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"观测数据(Raw): {OBS_RAW_PATH}\n")
    f.write(f"观测数据时间范围: 最后1827天\n")
    f.write("=" * 80 + "\n\n")
    f.write(
        f"{'模型名称':<25} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'NSE':<10} {'PCC':<10} {'SSIM':<10} {'状态':<10}\n")
    f.write("-" * 115 + "\n")

    pbar = tqdm(MODELS_TO_PROCESS, desc="处理模型")
    for model_name in pbar:
        pbar.set_description(f"处理: {model_name}")

        try:
            # 1. 定义路径
            model_norm_dir = os.path.join(CMIP_BASE_DIR, f"{model_name}_normalized")
            norm_params_path = os.path.join(model_norm_dir, "normalization_params.npz")
            obs_normalized_path = os.path.join(OBS_NORMALIZED_DIR, f'sst_daily_{model_name}.npz')

            train_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_train.npz")
            val_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_val.npz")
            test_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_test.npz")

            # 2. 加载数据
            X_train = np.load(train_file)['sst']
            X_val = np.load(val_file)['sst']
            X_cmip_train = np.concatenate([X_train, X_val], axis=0)
            X_cmip_test = np.load(test_file)['sst']
            y_obs_train_norm = np.load(obs_normalized_path)['sst'][:X_cmip_train.shape[0]]

            # 3. 训练 (Fit)
            model = QuantileMappingCorrector(mask, method=QM_METHOD, n_quantiles=N_QUANTILES)
            model.fit(X_cmip_train, y_obs_train_norm, norm_params_path)

            # 4. 预测 (Predict)
            y_pred_raw = model.predict(X_cmip_test)

            # 5. 对齐与评估
            min_len = min(y_pred_raw.shape[0], y_obs_test_raw.shape[0])
            y_pred_aligned = y_pred_raw[:min_len]
            y_obs_aligned = y_obs_test_raw[:min_len]

            model_output_dir = os.path.join(OUTPUT_DATA_DIR, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            save_path_preds = os.path.join(model_output_dir, 'test_corrections.npy')
            save_path_trues = os.path.join(model_output_dir, 'test_trues.npy')

            np.save(save_path_preds, y_pred_aligned)
            np.save(save_path_trues, y_obs_aligned)

            if y_pred_aligned.shape[0] != 1827:
                print(f"  Warning: {model_name} 预测长度 {y_pred_aligned.shape[0]} != 1827")

            print(f"  计算指标 (PCC已加速)...")
            rmse, mae, mse, nse, mean_pcc, mean_ssim = metric(mask, y_pred_aligned, y_obs_aligned)

            # 6. 保存结果
            results = {
                "rmse": np.round(rmse, 4),
                "mae": np.round(mae, 4),
                "mse": np.round(mse, 4),
                "nse": np.round(nse, 4),
                "mean_pcc": np.round(mean_pcc, 4),
                "mean_ssim": np.round(mean_ssim, 4),
                "method": QM_METHOD,
                "n_quantiles": N_QUANTILES
            }
            all_results[model_name] = results
            models_succeeded.append(model_name)

            f.write(f"{model_name:<25} {results['rmse']:<10.4f} {results['mae']:<10.4f} "
                    f"{results['mse']:<10.4f} {results['nse']:<10.4f} {results['mean_pcc']:<10.4f} "
                    f"{results['mean_ssim']:<10.4f} {'✓':<10}\n")
            f.flush()

        except FileNotFoundError as e:
            print(f"  ❌ 失败: {model_name} - {e}")
            models_failed.append(f"{model_name} (找不到文件)")
            f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 文件缺失':<10}\n")
            f.flush()
        except Exception as e:
            print(f"  ❌ 失败: {model_name} - {e}")
            import traceback

            traceback.print_exc()
            models_failed.append(f"{model_name} (运行时错误)")
            f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                    f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 计算错误':<10}\n")
            f.flush()

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

# 保存完整结果
np.save(OUTPUT_NPY, all_results)
print(f"\n✅ 完整结果已保存到: {OUTPUT_NPY}")
print(f"✅ 文本结果已保存到: {OUTPUT_TXT}")

# 最终总结
print("\n" + "=" * 60)
print("🎉 批量计算完成! (Quantile Mapping)")
print("=" * 60)
print(f"  ✅ 成功: {len(models_succeeded)}")
print(f"  ❌ 失败: {len(models_failed)}")
if models_failed:
    print("\n失败的模型列表:")
    for model in models_failed:
        print(f"  - {model}")
print("=" * 60)
print(f"\n📊 请查看结果文件: {OUTPUT_TXT}")