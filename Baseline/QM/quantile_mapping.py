import numpy as np
import os
from tqdm import tqdm
from scipy import interpolate
from joblib import Parallel, delayed


class QuantileMappingCorrector:
    """
    逐格点分位数映射偏差校正 - 支持 EDCDF 方法

    支持三种QM方法:
    1. Empirical Quantile Mapping (EQM) - 经验分位数映射 (直接映射值)
    2. Parametric Quantile Mapping (PQM) - 参数化分位数映射 (正态分布假设)
    3. Equidistant CDF (EDCDF) - 等距CDF匹配 (基于分位数差值校正，保留趋势)
    """

    def __init__(self, mask, method='empirical', n_quantiles=100):
        """
        Args:
            mask: 海陆掩码 (H, W)
            method: 'empirical', 'parametric', 或 'edcdf'
            n_quantiles: 分位数个数 (默认100)
        """
        self.mask = mask.astype(bool)
        self.is_fitted = False
        self.method = method
        self.n_quantiles = n_quantiles

        # 归一化参数
        self.data_min = None
        self.data_max = None

        # 存储映射函数
        self.quantile_maps = None  # 每个格点的映射函数或参数
        self.height = mask.shape[0]
        self.width = mask.shape[1]
        self.n_valid_points = np.sum(self.mask)

    def load_normalization_params(self, norm_params_path):
        """加载训练集归一化参数"""
        if os.path.exists(norm_params_path):
            norm_params = np.load(norm_params_path)
            self.data_min = norm_params['min']
            self.data_max = norm_params['max']
            print(f"✅ 加载归一化参数 - Min: {self.data_min:.4f}, Max: {self.data_max:.4f}")
        else:
            raise FileNotFoundError(f"⚠️ 未找到归一化参数文件: {norm_params_path}")

    def denormalize(self, normalized_data):
        """反归一化"""
        if self.data_min is not None and self.data_max is not None:
            return normalized_data * (self.data_max - self.data_min) + self.data_min
        else:
            print("⚠️ 归一化参数未加载, 返回原始数据")
            return normalized_data

    def _fit_single_point(self, k, X_ocean, y_ocean):
        """为单个格点k拟合分位数映射"""
        x_k = X_ocean[:, k]  # 模式数据训练集 (Model Train)
        y_k = y_ocean[:, k]  # 观测数据训练集 (Obs Train)

        # 移除NaN值
        valid_mask = np.isfinite(x_k) & np.isfinite(y_k)
        if np.sum(valid_mask) < 10:  # 数据点太少
            return k, None

        x_k_valid = x_k[valid_mask]
        y_k_valid = y_k[valid_mask]

        # 统一的分位数定义
        quantiles = np.linspace(0, 1, self.n_quantiles)

        if self.method == 'empirical':
            # EQM: 直接建立 Model值 -> Obs值 的映射
            model_q = np.quantile(x_k_valid, quantiles)
            obs_q = np.quantile(y_k_valid, quantiles)

            qm_func = interpolate.interp1d(
                model_q, obs_q,
                kind='linear', bounds_error=False,
                fill_value='extrapolate'  # 关键：允许外推
            )
            return k, qm_func

        elif self.method == 'parametric':
            # PQM: 基于正态分布假设
            model_mean, model_std = np.mean(x_k_valid), np.std(x_k_valid)
            obs_mean, obs_std = np.mean(y_k_valid), np.std(y_k_valid)

            def qm_func(x):
                z = (x - model_mean) / (model_std + 1e-8)
                return z * obs_std + obs_mean

            return k, qm_func

        elif self.method == 'edcdf':
            # EDCDF: 需要存储 "概率 -> 历史Obs值" 和 "概率 -> 历史Model值" 两个反函数
            # 计算分位数对应的值
            model_q_vals = np.quantile(x_k_valid, quantiles)
            obs_q_vals = np.quantile(y_k_valid, quantiles)

            # 建立两个插值函数: P(概率) -> Value(值)
            # F_inv_obs_train(p)
            func_p_to_obs = interpolate.interp1d(
                quantiles, obs_q_vals,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            # F_inv_model_train(p)
            func_p_to_mod = interpolate.interp1d(
                quantiles, model_q_vals,
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )

            # 返回两个函数作为元组
            return k, (func_p_to_obs, func_p_to_mod)

        else:
            raise ValueError(f"未知的方法: {self.method}")

    def fit(self, X_cmip_train, y_obs_train, norm_params_path=None):
        """训练分位数映射模型"""
        print(f"🚀 逐格点训练分位数映射 ({self.method.upper()})...")

        X_ocean = X_cmip_train[:, self.mask].astype(np.float32)
        y_ocean = y_obs_train[:, self.mask].astype(np.float32)
        n_samples, n_valid_points = X_ocean.shape

        # 初始化存储
        self.quantile_maps = [None] * n_valid_points

        results = Parallel(n_jobs=-1)(
            delayed(self._fit_single_point)(k, X_ocean, y_ocean)
            for k in tqdm(range(n_valid_points), desc="拟合QM", leave=False)
        )

        for k, qm_data in results:
            self.quantile_maps[k] = qm_data

        self.is_fitted = True
        if norm_params_path:
            self.load_normalization_params(norm_params_path)
        print(f"✅ 训练完成!")

    def predict(self, X_cmip_test):
        """预测/校正数据"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        print(f"🔮 逐格点预测 ({self.method.upper()})...")

        X_ocean_test = X_cmip_test[:, self.mask].astype(np.float32)
        n_samples, n_valid_points = X_ocean_test.shape
        y_pred_ocean = np.zeros_like(X_ocean_test)

        # 逐格点应用
        for k in tqdm(range(n_valid_points), desc="逐格点Predict", leave=False):
            qm_data = self.quantile_maps[k]

            if qm_data is None:
                # 无法校正，保持原值
                y_pred_ocean[:, k] = X_ocean_test[:, k]
                continue

            x_k_test = X_ocean_test[:, k]
            valid_mask = np.isfinite(x_k_test)
            if np.sum(valid_mask) == 0:
                continue

            x_valid = x_k_test[valid_mask]

            # ================== 方法分支 ==================
            if self.method == 'edcdf':
                # Unpack训练好的两个反函数
                func_p_to_obs, func_p_to_mod = qm_data

                # 1. 计算测试数据的CDF (概率 p)
                # 使用 argsort 获取每个数值在当前测试集分布中的排名百分比
                # 这是 F_model_future(x) 的经验估计
                n = len(x_valid)
                ranks = np.argsort(np.argsort(x_valid))  # 两次argsort得到排名(0到n-1)
                # (ranks + 0.5) / n 避免出现0或1，防止无穷大
                p_test = (ranks + 0.5) / n

                # 2. 计算校正项: F_inv_obs(p) - F_inv_mod(p)
                correction_term = func_p_to_obs(p_test) - func_p_to_mod(p_test)

                # 3. 应用公式: x_new = x + correction
                y_pred_ocean[valid_mask, k] = x_valid + correction_term

            else:
                # EQM 或 PQM (直接即为映射函数)
                qm_func = qm_data
                y_pred_ocean[valid_mask, k] = qm_func(x_valid)
            # ============================================

            # 处理原来的NaN
            invalid_mask = ~valid_mask
            if np.sum(invalid_mask) > 0:
                y_pred_ocean[invalid_mask, k] = np.nan

        # 重建空间场
        print("   重建空间场...")
        predictions = np.full((n_samples, self.height * self.width), np.nan, dtype=np.float32)
        predictions[:, self.mask.flatten()] = y_pred_ocean
        predictions = predictions.reshape(n_samples, self.height, self.width)

        return self.denormalize(predictions)


# 辅助函数保持不变 (metric, _calculate_pcc_for_day)
def _calculate_pcc_for_day(t, valid_pred_ts, valid_true_ts):
    from scipy.stats import pearsonr
    p_t = valid_pred_ts[t]
    t_t = valid_true_ts[t]
    valid_day_mask = np.isfinite(p_t) & np.isfinite(t_t)
    if np.sum(valid_day_mask) > 1:
        pcc, _ = pearsonr(p_t[valid_day_mask], t_t[valid_day_mask])
        if np.isfinite(pcc): return pcc
    return np.nan


def metric(mask, pred, true):
    # (此部分与原文件保持一致即可)
    from scipy.stats import pearsonr
    assert pred.shape == true.shape
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
    error = valid_pred - valid_true
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    y_mean = np.mean(valid_true)
    nse = 1 - np.sum((valid_true - valid_pred) ** 2) / (np.sum((valid_true - y_mean) ** 2) + 1e-10)

    valid_pred_ts = pred[:, mask_bool]
    valid_true_ts = true[:, mask_bool]
    time_steps = pred.shape[0]
    pcc_scores_raw = Parallel(n_jobs=-1, backend='threading')(
        delayed(_calculate_pcc_for_day)(t, valid_pred_ts, valid_true_ts) for t in range(time_steps)
    )
    pcc_scores = [p for p in pcc_scores_raw if np.isfinite(p)]
    mean_pcc = np.mean(pcc_scores) if pcc_scores else np.nan
    return rmse, mae, mse, nse, mean_pcc, mean_pcc