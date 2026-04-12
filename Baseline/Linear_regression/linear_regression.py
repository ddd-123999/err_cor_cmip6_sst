import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

# --- 把辅助函数粘贴到这里 ---
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

        # 存储每个格点的回归系数
        self.n_valid_points = np.sum(self.mask)
        self.coefs = None
        self.intercepts = None

    def load_normalization_params(self, norm_params_path):
        """加载模式数据的归一化参数"""
        if os.path.exists(norm_params_path):
            norm_params = np.load(norm_params_path)
            self.data_min = norm_params['min']
            self.data_max = norm_params['max']
            print(f"✅ 加载归一化参数 - Min: {self.data_min:.4f}, Max: {self.data_max:.4f}")
        else:
            print(f"⚠️ 未找到归一化参数文件: {norm_params_path}")

    def denormalize(self, normalized_data):
        """将模式数据反归一化到物理量"""
        if self.data_min is not None and self.data_max is not None:
            return normalized_data * (self.data_max - self.data_min) + self.data_min
        else:
            return normalized_data

    def prepare_sequences(self, X_data, y_data, seq_len, pred_len):
        """
        准备时序数据

        Parameters:
        - X_data: 模式数据 [n_samples, height, width]
        - y_data: 观测数据 [n_samples, height, width]
        - seq_len: 输入序列长度
        - pred_len: 输出序列长度

        Returns:
        - X_sequences: [n_sequences, seq_len, height, width]
        - y_sequences: [n_sequences, pred_len, height, width]
        """
        n_samples = X_data.shape[0]
        n_sequences = n_samples - seq_len - pred_len + 1

        if n_sequences <= 0:
            raise ValueError(f"数据长度不足! 需要至少 {seq_len + pred_len} 个样本")

        height, width = X_data.shape[1], X_data.shape[2]

        X_sequences = np.zeros((n_sequences, seq_len, height, width))
        y_sequences = np.zeros((n_sequences, pred_len, height, width))

        for i in range(n_sequences):
            X_sequences[i] = X_data[i:i + seq_len]
            y_sequences[i] = y_data[i + seq_len:i + seq_len + pred_len]

        return X_sequences, y_sequences

    def fit_vectorized(self, X_cmip_train, y_obs_train, seq_len=1, pred_len=1, norm_params_path=None):
        """
        逐格点训练线性回归模型 (低内存 + 并行加速)
        """
        print(f"🚀 逐格点训练 (低内存 + 并行加速)")
        print(f"   输入序列长度: {seq_len}, 输出序列长度: {pred_len}")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.height = X_cmip_train.shape[1]
        self.width = X_cmip_train.shape[2]

        X_ocean = X_cmip_train[:, self.mask].astype(np.float32)
        y_ocean = y_obs_train[:, self.mask].astype(np.float32)

        n_samples, n_valid_points = X_ocean.shape
        print(f"   有效格点数: {n_valid_points}, 时间步: {n_samples}")

        n_seq = n_samples - seq_len - pred_len + 1
        if n_seq <= 0:
            raise ValueError(f"数据长度不足! 需要至少 {seq_len + pred_len} 个样本")

        print(f"   将生成 {n_seq} 个训练序列...")

        self.coefs = np.zeros((n_valid_points, seq_len, pred_len), dtype=np.float32)
        self.intercepts = np.zeros((n_valid_points, pred_len), dtype=np.float32)
        ridge_alpha = 1e-6

        # 使用 joblib 并行处理循环
        # n_jobs=-1 表示使用所有 CPU 核心
        print("启动并行拟合 (使用所有CPU核心)...")
        results = Parallel(n_jobs=-1)(
            delayed(fit_single_point)(k, X_ocean, y_ocean, seq_len, pred_len, n_seq, ridge_alpha)
            for k in tqdm(range(n_valid_points), desc="分派任务")
        )

        # 收集结果
        print("收集并行计算结果...")
        for k, coef, intercept in results:
            self.coefs[k] = coef
            self.intercepts[k] = intercept

        self.is_fitted = True
        if norm_params_path:
            self.load_normalization_params(norm_params_path)

        print(f"✅ 训练完成!")

    def predict(self, X_cmip_test):
        """使用训练好的模型校正模式数据 (低内存版本)"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        print(f"🔮 逐格点预测...")

        # 1. 提取海洋点
        X_ocean_test = X_cmip_test[:, self.mask].astype(np.float32)
        n_samples, n_valid_points = X_ocean_test.shape

        # 预测结果 (仅海洋点)
        # y_pred_ocean 形状 [n_pred_seq, n_valid_points, pred_len]

        n_pred_seq = n_samples - self.seq_len + 1
        y_pred_ocean = np.zeros((n_pred_seq, n_valid_points, self.pred_len), dtype=np.float32)

        # 2. 逐格点应用模型
        # 这一步也可以并行, 但预测通常很快, 先用循环
        for k in tqdm(range(n_valid_points), desc="逐格点预测"):
            # 提取当前点的系数
            coef_k = self.coefs[k]  # [seq_len, pred_len]
            intercept_k = self.intercepts[k]  # [pred_len]

            # 准备当前点的测试序列
            x_k_test = X_ocean_test[:, k]
            X_k_seq = np.zeros((n_pred_seq, self.seq_len), dtype=np.float32)

            for i in range(n_pred_seq):
                X_k_seq[i] = x_k_test[i:i + self.seq_len]

            # 3. 预测
            # (X @ coef) + intercept
            y_pred_ocean[:, k, :] = X_k_seq @ coef_k + intercept_k[np.newaxis, :]

        # 4. 重建完整空间场
        print("重建空间场...")
        if self.pred_len > 1:
            # y_pred_ocean [n_pred_seq, n_valid_points, pred_len] ->
            # y_pred_flat [n_pred_seq * pred_len, n_valid_points]
            y_pred_flat = y_pred_ocean.transpose(0, 2, 1).reshape(-1, n_valid_points)
        else:
            # y_pred_ocean [n_pred_seq, n_valid_points, 1] ->
            # y_pred_flat [n_pred_seq, n_valid_points]
            y_pred_flat = y_pred_ocean.reshape(n_pred_seq, n_valid_points)

        n_total_preds = y_pred_flat.shape[0]

        # 创建一个大的 NaN 数组, 然后填充海洋点
        predictions = np.full((n_total_preds, self.height * self.width), np.nan, dtype=np.float32)
        mask_flat = self.mask.flatten()

        predictions[:, mask_flat] = y_pred_flat

        # 恢复时空形状
        predictions = predictions.reshape(n_total_preds, self.height, self.width)

        # 5. 反归一化
        predictions = self.denormalize(predictions)

        return predictions


def metric(mask, pred, true):
    """评估指标计算"""
    from scipy.stats import pearsonr

    assert pred.shape == true.shape
    assert mask.shape == (pred.shape[-2], pred.shape[-1])

    if len(pred.shape) == 4:
        pred = pred.reshape(-1, *pred.shape[-2:])
        true = true.reshape(-1, *true.shape[-2:])

    mask_bool = mask.astype(bool)
    valid_pred = pred[:, mask_bool]
    valid_true = true[:, mask_bool]

    # RMSE & MAE & MSE
    error = valid_pred - valid_true
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)

    # NSE
    y_mean = np.mean(valid_true)
    nse = 1 - np.sum((valid_true - valid_pred) ** 2) / np.sum((valid_true - y_mean) ** 2)

    # PCC
    time_steps = pred.shape[0]
    pcc_scores = []
    for t in range(time_steps):
        pcc, _ = pearsonr(valid_pred[t], valid_true[t])
        pcc_scores.append(pcc)
    mean_pcc = np.mean(pcc_scores)

    # SSIM(用PCC近似)
    mean_ssim = mean_pcc

    return rmse, mae, mse, nse, mean_pcc, mean_ssim


def run_linear_regression(config):
    """
    运行线性回归实验 (归一化数据)

    Parameters:
    - config: 配置字典,包含所有路径和参数
    """
    print("=" * 70)
    print(f"线性回归 Baseline - {config['cmip_name']} - {config['scenario']}")
    print(f"输入序列: {config['seq_len']}天, 输出序列: {config['pred_len']}天")
    print(f"数据类型: 归一化")
    print("=" * 70)

    # 加载掩码
    mask = np.load(config['mask_path'])
    print(f"✅ 加载掩码: {mask.shape}, 有效点数: {np.sum(mask)}")

    # 确定数据路径
    model_data_dir = os.path.join(config['cmip_base_dir'], f"{config['cmip_name']}_normalized")
    obs_data_path = config['obs_normalized_path']
    norm_params_path = os.path.join(model_data_dir, "normalization_params.npz")

    print(f"\n📂 数据路径:")
    print(f"   CMIP6模式(归一化): {model_data_dir}")
    print(f"   观测数据(归一化): {obs_data_path}")
    print(f"   归一化参数: {norm_params_path}")

    # 加载数据
    print(f"\n📥 加载数据...")

    # 合并训练集和验证集
    train_file = os.path.join(model_data_dir, f"{config['scenario']}_train.npz")
    val_file = os.path.join(model_data_dir, f"{config['scenario']}_val.npz")
    test_file = os.path.join(model_data_dir, f"{config['scenario']}_test.npz")

    X_train = np.load(train_file)['sst']
    X_val = np.load(val_file)['sst']
    X_cmip_train = np.concatenate([X_train, X_val], axis=0)
    X_cmip_test = np.load(test_file)['sst']

    # 观测数据(训练用归一化,测试用原始物理量评估)
    y_obs_train = np.load(obs_data_path)['sst'][:X_cmip_train.shape[0]]
    y_obs_test = np.load(config['obs_raw_path'])['sst'][-X_cmip_test.shape[0]:]

    print(f"   模式训练(归一化): {X_cmip_train.shape} [train+val: 12053+1826=13879]")
    print(f"   观测训练(归一化): {y_obs_train.shape}")
    print(f"   模式测试(归一化): {X_cmip_test.shape}")
    print(f"   观测测试(原始物理量): {y_obs_test.shape}")

    # 训练模型
    print(f"\n🚀 开始训练...")
    model = LinearRegressionCorrector(mask)
    model.fit_vectorized(
        X_cmip_train, y_obs_train,
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        norm_params_path=norm_params_path
    )

    # 预测 (自动反归一化到物理量)
    print(f"\n🔮 预测测试集...")
    y_pred = model.predict(X_cmip_test)

    # 匹配预测和真值的长度
    min_len = min(y_pred.shape[0], y_obs_test.shape[0])
    y_pred = y_pred[:min_len]
    y_obs_test = y_obs_test[:min_len]

    print(f"   预测结果(反归一化): {y_pred.shape}")
    print(f"   真值数据(物理量): {y_obs_test.shape}")

    # 计算指标(在物理量尺度)
    print(f"\n📊 计算评估指标(物理量尺度)...")
    rmse, mae, mse, nse, pcc, ssim = metric(mask, y_pred, y_obs_test)

    # 结果
    results = {
        "cmip_name": config['cmip_name'],
        "scenario": config['scenario'],
        "seq_len": config['seq_len'],
        "pred_len": config['pred_len'],
        "data_type": "normalized",
        "rmse": float(np.round(rmse, 4)),
        "mae": float(np.round(mae, 4)),
        "mse": float(np.round(mse, 4)),
        "nse": float(np.round(nse, 4)),
        "mean_pcc": float(np.round(pcc, 4)),
        "mean_ssim": float(np.round(ssim, 4))
    }

    # 打印结果
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    for key, value in results.items():
        if key not in ['cmip_name', 'scenario', 'seq_len', 'pred_len', 'data_type']:
            print(f"  {key.upper():10s}: {value}")

    # 保存结果
    output_dir = config.get('output_dir', './baseline_results')
    os.makedirs(output_dir, exist_ok=True)

    filename = f"LR_{config['cmip_name']}_{config['scenario']}_seq{config['seq_len']}_pred{config['pred_len']}_norm"
    output_path = os.path.join(output_dir, f"{filename}.npy")

    np.save(output_path, results)
    print(f"\n✅ 结果已保存至: {output_path}")

    return results


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='线性回归 Baseline for CMIP6偏差校正 (归一化数据)')

    # 必需参数
    parser.add_argument('--cmip_name', type=str, default='ACCESS-CM2',
                        help='CMIP6模式名称 (例如: ACCESS-CM2, CNRM-CM6-1-HR)')
    parser.add_argument('--scenario', type=str, default='ssp245',
                        choices=['ssp245', 'ssp585'],
                        help='情景 (ssp245 或 ssp585)')

    # 序列长度参数
    parser.add_argument('--seq_len', type=int, default=1,
                        help='输入序列长度(天) (默认: 1)')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='输出序列长度(天) (默认: 1)')

    # 路径配置
    parser.add_argument('--cmip_base_dir', type=str,
                        default='D:/err_cor_cmip6_sst/Preprocessing/dataset',
                        help='CMIP6数据根目录')
    parser.add_argument('--obs_base_dir', type=str,
                        default='D:/err_cor_cmip6_sst/Preprocessing/observation',
                        help='观测数据根目录')
    parser.add_argument('--mask_path', type=str,
                        default='D:/err_cor_cmip6_sst/Preprocessing/observation/obs/mask.npy',
                        help='掩码文件路径')
    parser.add_argument('--output_dir', type=str,
                        default='./baseline_results',
                        help='结果输出目录')

    args = parser.parse_args()

    # 构建配置
    config = {
        'cmip_name': args.cmip_name,
        'scenario': args.scenario,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'cmip_base_dir': args.cmip_base_dir,
        'obs_raw_path': os.path.join(args.obs_base_dir, 'obs/sst_daily_not_to_be_normalized.npz'),
        'obs_normalized_path': os.path.join(args.obs_base_dir, f'obs_normalized/sst_daily_{args.cmip_name}.npz'),
        'mask_path': args.mask_path,
        'output_dir': args.output_dir
    }

    # 运行实验
    run_linear_regression(config)


if __name__ == "__main__":
    # 方式1: 命令行运行
    # python linear_regression.py --cmip_name ACCESS-CM2 --scenario ssp245 --seq_len 7 --pred_len 1

    # 方式2: 直接在代码中配置运行
    if len(os.sys.argv) == 1:  # 没有命令行参数时使用默认配置
        print("使用默认配置运行...\n")
        config = {
            'cmip_name': 'ACCESS-CM2',  # ← 修改模型名
            'scenario': 'ssp245',  # ← 修改情景
            'seq_len': 1,  # ← 修改输入天数
            'pred_len': 1,  # ← 修改输出天数
            'cmip_base_dir': 'D:/err_cor_cmip6_sst/Preprocessing/dataset',
            'obs_raw_path': 'D:/err_cor_cmip6_sst/Preprocessing/observation/obs/sst_daily_not_to_be_normalized.npz',
            'obs_normalized_path': 'D:/err_cor_cmip6_sst/Preprocessing/observation/obs_normalized/sst_daily_ACCESS-CM2.npz',
            'mask_path': 'D:/err_cor_cmip6_sst/Preprocessing/observation/obs/mask.npy',
            'output_dir': 'D:/err_cor_cmip6_sst/Experiment/EXP7/Linear regression'
        }
        run_linear_regression(config)
    else:
        main()