import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
from datetime import datetime
from scipy.stats import pearsonr


# ====================================================================
# 1. 拷贝自 linear_regression.py (核心类与辅助函数)
# ====================================================================

# --- 辅助函数 ---
def fit_single_point(k, X_ocean, y_ocean, seq_len, pred_len, n_seq, ridge_alpha):
    """为单个格点 k 拟合模型"""
    x_k = X_ocean[:, k]
    y_k = y_ocean[:, k]
    X_k_seq = np.zeros((n_seq, seq_len), dtype=np.float32)
    y_k_seq = np.zeros((n_seq, pred_len), dtype=np.float32)
    for i in range(n_seq):
        X_k_seq[i] = x_k[i:i + seq_len]
        y_k_seq[i] = y_k[i + seq_len:i + seq_len + pred_len]
    X_k_seq_bias = np.hstack([X_k_seq, np.ones((n_seq, 1), dtype=np.float32)])
    try:
        XtX = X_k_seq_bias.T @ X_k_seq_bias
        Xty = X_k_seq_bias.T @ y_k_seq
        XtX += ridge_alpha * np.eye(XtX.shape[0], dtype=np.float32)
        sol = np.linalg.solve(XtX, Xty)
        coef = sol[:-1, :]
        intercept = sol[-1, :]
        return k, coef, intercept
    except np.linalg.LinAlgError:
        return k, np.zeros((seq_len, pred_len), dtype=np.float32), np.zeros(pred_len, dtype=np.float32)


# --- 辅助函数结束 ---

class LinearRegressionCorrector:
    """逐格点线性回归偏差校正 - 归一化数据版本"""

    def __init__(self, mask):
        self.mask = mask.astype(bool)
        self.is_fitted = False
        self.data_min = None
        self.data_max = None
        self.n_valid_points = np.sum(self.mask)
        self.coefs = None
        self.intercepts = None

    def load_normalization_params(self, norm_params_path):
        if os.path.exists(norm_params_path):
            norm_params = np.load(norm_params_path)
            self.data_min = norm_params['min']
            self.data_max = norm_params['max']
        else:
            raise FileNotFoundError(f"⚠️ 未找到归一化参数文件: {norm_params_path}")

    def denormalize(self, normalized_data):
        if self.data_min is not None and self.data_max is not None:
            return normalized_data * (self.data_max - self.data_min) + self.data_min
        else:
            print("⚠️ 归一化参数未加载, 返回原始数据")
            return normalized_data

    def fit_vectorized(self, X_cmip_train, y_obs_train, seq_len=1, pred_len=1, norm_params_path=None):
        print(f"🚀 逐格点训练 (S{seq_len}/P{pred_len}) (低内存 + 并行)...")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.height = X_cmip_train.shape[1]
        self.width = X_cmip_train.shape[2]
        X_ocean = X_cmip_train[:, self.mask].astype(np.float32)
        y_ocean = y_obs_train[:, self.mask].astype(np.float32)
        n_samples, n_valid_points = X_ocean.shape
        n_seq = n_samples - seq_len - pred_len + 1
        if n_seq <= 0:
            raise ValueError(f"数据长度不足! 需要至少 {seq_len + pred_len} 个样本")

        self.coefs = np.zeros((n_valid_points, seq_len, pred_len), dtype=np.float32)
        self.intercepts = np.zeros((n_valid_points, pred_len), dtype=np.float32)
        ridge_alpha = 1e-6

        print("   启动并行拟合 (使用所有CPU核心)...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_single_point)(k, X_ocean, y_ocean, seq_len, pred_len, n_seq, ridge_alpha)
            for k in tqdm(range(n_valid_points), desc="分派Fit任务", leave=False)
        )
        print("   收集并行计算结果...")
        for k, coef, intercept in results:
            self.coefs[k] = coef
            self.intercepts[k] = intercept
        self.is_fitted = True
        if norm_params_path:
            self.load_normalization_params(norm_params_path)
        print(f"✅ 训练完成!")

    def predict(self, X_cmip_test):
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        print(f"🔮 逐格点预测...")
        X_ocean_test = X_cmip_test[:, self.mask].astype(np.float32)
        n_samples, n_valid_points = X_ocean_test.shape

        # 预测时, 我们滑动窗口直到最后
        n_pred_seq = n_samples - self.seq_len + 1
        y_pred_ocean = np.zeros((n_pred_seq, n_valid_points, self.pred_len), dtype=np.float32)

        # 预测循环 (通常很快, 可选并行)
        for k in tqdm(range(n_valid_points), desc="逐格点Predict", leave=False):
            coef_k = self.coefs[k]
            intercept_k = self.intercepts[k]
            x_k_test = X_ocean_test[:, k]
            X_k_seq = np.zeros((n_pred_seq, self.seq_len), dtype=np.float32)
            for i in range(n_pred_seq):
                X_k_seq[i] = x_k_test[i:i + self.seq_len]
            y_pred_ocean[:, k, :] = X_k_seq @ coef_k + intercept_k[np.newaxis, :]

        print("   重建空间场...")
        if self.pred_len > 1:
            y_pred_flat = y_pred_ocean.transpose(0, 2, 1).reshape(-1, n_valid_points)
        else:
            y_pred_flat = y_pred_ocean.reshape(n_pred_seq, n_valid_points)

        n_total_preds = y_pred_flat.shape[0]
        predictions = np.full((n_total_preds, self.height * self.width), np.nan, dtype=np.float32)
        mask_flat = self.mask.flatten()
        predictions[:, mask_flat] = y_pred_flat
        predictions = predictions.reshape(n_total_preds, self.height, self.width)
        predictions = self.denormalize(predictions)
        return predictions


# ====================================================================
# 2. 评估函数 (PCC并行加速版)
# ====================================================================

def _calculate_pcc_for_day(t, valid_pred_ts, valid_true_ts):
    """辅助函数: 为 joblib 并行计算单日的PCC"""
    p_t = valid_pred_ts[t]
    t_t = valid_true_ts[t]
    valid_day_mask = np.isfinite(p_t) & np.isfinite(t_t)
    if np.sum(valid_day_mask) > 1:
        pcc, _ = pearsonr(p_t[valid_day_mask], t_t[valid_day_mask])
        if np.isfinite(pcc):
            return pcc
    return np.nan


def metric(mask, pred, true):
    """评估指标计算 (PCC并行加速版)"""
    assert pred.shape == true.shape, f"预测和真值形状不匹配: {pred.shape} vs {true.shape}"
    assert mask.shape == (pred.shape[-2], pred.shape[-1])
    if len(pred.shape) == 4:
        pred = pred.reshape(-1, *pred.shape[-2:])
        true = true.reshape(-1, *true.shape[-2:])

    mask_bool = mask.astype(bool)
    valid_pred = pred[:, mask_bool]
    valid_true = true[:, mask_bool]

    valid_mask = np.isfinite(valid_pred) & np.isfinite(valid_true)
    if not np.all(valid_mask):
        valid_pred = valid_pred[valid_mask]
        valid_true = valid_true[valid_mask]

    # --- RMSE, MAE, MSE, NSE (这些都很快) ---
    error = valid_pred - valid_true
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    y_mean = np.mean(valid_true)
    nse = 1 - np.sum((valid_true - valid_pred) ** 2) / (np.sum((valid_true - y_mean) ** 2) + 1e-10)

    # --- PCC (并行加速版) ---
    valid_pred_ts = pred[:, mask_bool]
    valid_true_ts = true[:, mask_bool]
    time_steps = pred.shape[0]

    pcc_scores_raw = Parallel(n_jobs=-1, backend='threading')(
        delayed(_calculate_pcc_for_day)(t, valid_pred_ts, valid_true_ts)
        for t in range(time_steps)
    )
    pcc_scores = [p for p in pcc_scores_raw if np.isfinite(p)]
    mean_pcc = np.mean(pcc_scores) if pcc_scores else np.nan
    mean_ssim = mean_pcc

    return rmse, mae, mse, nse, mean_pcc, mean_ssim


# ====================================================================
# 3. 批量处理配置
# ====================================================================

# ----- 实验参数 (在这里修改) -----
SEQ_LEN = 3
PRED_LEN = 1
SSP_SCENARIO = 'ssp245'
# -----------------------------------

# 定义所有要处理的模型
MODELS_TO_PROCESS = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'EC-Earth3', 'EC-Earth3-CC',
    'EC-Earth3-veg', 'EC-Earth3-Veg-LR', 'IPSL-CM6A-LR', 'MIROC6',
    'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3',
    'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5',
    'CMCC-ESM2', 'NorESM2-LM', 'NorESM2-MM', 'GFDL-CM4', 'GFDL-ESM4'
]

# 文件路径 (使用相对路径)
OBS_BASE_DIR = '../../Preprocessing/observation'
CMIP_BASE_DIR = '../../Preprocessing/dataset'
OUTPUT_DATA_DIR = f'./lr_results_data_s{SEQ_LEN}_p{PRED_LEN}'

MASK_PATH = os.path.join(OBS_BASE_DIR, 'obs/mask.npy')
OBS_RAW_PATH = os.path.join(OBS_BASE_DIR, 'obs/sst_daily_not_to_be_normalized.npz')
OBS_NORMALIZED_DIR = os.path.join(OBS_BASE_DIR, 'obs_normalized')

# 输出文件 (根据S/P参数命名)
OUTPUT_TXT = f'./lr_metrics_s{SEQ_LEN}_p{PRED_LEN}_results.txt'
OUTPUT_NPY = f'./lr_metrics_s{SEQ_LEN}_p{PRED_LEN}_all.npy'

# ====================================================================
# 4. 批量处理主程序
# ====================================================================

print("=" * 60)
print(f"批量计算 LinearRegression (S{SEQ_LEN}/P{PRED_LEN}) 修正后指标 (已加速)")
print("=" * 60)

print("\n加载全局数据 (Mask, 观测Test)...")
mask = np.load(MASK_PATH)
# 原始观测(Test集), 取最后1827天
y_obs_test_raw = np.load(OBS_RAW_PATH)['sst'][-1827:]
print(f"  ✓ Mask形状: {mask.shape}")
print(f"  ✓ 观测Test(Raw)形状: {y_obs_test_raw.shape}")

all_results = {}
models_failed = []
models_succeeded = []

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write(f"LinearRegression (S{SEQ_LEN}/P{PRED_LEN}) 修正后 Metrics 计算结果\n")
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
            # ----- 1. 定义路径 -----
            model_norm_dir = os.path.join(CMIP_BASE_DIR, f"{model_name}_normalized")
            norm_params_path = os.path.join(model_norm_dir, "normalization_params.npz")
            obs_normalized_path = os.path.join(OBS_NORMALIZED_DIR, f'sst_daily_{model_name}.npz')

            train_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_train.npz")
            val_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_val.npz")
            test_file = os.path.join(model_norm_dir, f"{SSP_SCENARIO}_test.npz")

            # ----- 2. 加载数据 -----
            X_train = np.load(train_file)['sst']
            X_val = np.load(val_file)['sst']
            X_cmip_train = np.concatenate([X_train, X_val], axis=0)
            X_cmip_test = np.load(test_file)['sst']
            y_obs_train_norm = np.load(obs_normalized_path)['sst'][:X_cmip_train.shape[0]]

            # ----- 3. 训练 (Fit) (并行) -----
            model = LinearRegressionCorrector(mask)
            model.fit_vectorized(X_cmip_train, y_obs_train_norm,
                                 seq_len=SEQ_LEN,
                                 pred_len=PRED_LEN,
                                 norm_params_path=norm_params_path)

            # ----- 4. 预测 (Predict) -----
            y_pred_raw = model.predict(X_cmip_test)

            # ----- 5. 对齐与评估 (已修正对齐逻辑) -----
            # y_pred_raw[0] 对应 y_obs[SEQ_LEN]
            # y_pred_raw (shape n_total_preds) 对应 y_obs[SEQ_LEN : SEQ_LEN + n_total_preds]

            n_total_preds = y_pred_raw.shape[0]
            start_idx = SEQ_LEN
            # 确保我们有足够的观测数据来进行比较
            n_obs_available = y_obs_test_raw.shape[0] - start_idx  # 可用于对齐的观测

            # 取两者中较短的
            min_len = min(n_total_preds, n_obs_available)

            y_pred_aligned = y_pred_raw[:min_len]
            y_obs_aligned = y_obs_test_raw[start_idx: start_idx + min_len]
            # 5b. 保存校正后的数据和真实观测数据
            print(f"   保存数据...")
            model_output_dir = os.path.join(OUTPUT_DATA_DIR, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            save_path_preds = os.path.join(model_output_dir, 'test_corrections.npy')
            save_path_trues = os.path.join(model_output_dir, 'test_trues.npy')

            np.save(save_path_preds, y_pred_aligned)
            np.save(save_path_trues, y_obs_aligned)

            print(f"   对齐: 预测 {y_pred_aligned.shape}, 观测 {y_obs_aligned.shape}")

            print(f"   计算指标 (PCC已加速)...")
            rmse, mae, mse, nse, mean_pcc, mean_ssim = metric(mask, y_pred_aligned, y_obs_aligned)

            # ----- 6. 保存结果 -----
            results = {
                "rmse": np.round(rmse, 4),
                "mae": np.round(mae, 4),
                "mse": np.round(mse, 4),
                "nse": np.round(nse, 4),
                "mean_pcc": np.round(mean_pcc, 4),
                "mean_ssim": np.round(mean_ssim, 4),
                "seq_len": SEQ_LEN,
                "pred_len": PRED_LEN
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

# ========== 保存完整结果 ==========
np.save(OUTPUT_NPY, all_results)
print(f"\n✅ 完整结果已保存到: {OUTPUT_NPY}")
print(f"✅ 文本结果已保存到: {OUTPUT_TXT}")

# ========== 最终总结 ==========
print("\n" + "=" * 60)
print("🎉 批量计算完成！(LinearRegression)")
print("=" * 60)
print(f"  ✅ 成功: {len(models_succeeded)}")
print(f"  ❌ 失败: {len(models_failed)}")
if models_failed:
    print("\n失败的模型列表:")
    for model in models_failed:
        print(f"  - {model}")
print("=" * 60)
print(f"\n📊 请查看结果文件: {OUTPUT_TXT}")