import os

import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

def metric(mask, pred, true):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    assert pred.shape == true.shape, "预测值与真实值形状不一致"
    assert mask.shape == (pred.shape[-2], pred.shape[-1]), "掩码形状不匹配"

    # 处理数据形状
    if len(pred.shape) == 4:
        pred = pred.reshape(-1, *pred.shape[-2:])
        true = true.reshape(-1, *true.shape[-2:])

    mask = mask.astype(bool)
    time_steps = pred.shape[0]

    # 展开成一维并用掩码过滤
    valid_pred = pred[:, mask]
    valid_true = true[:, mask]

    # --- RMSE & MAE & MSE---
    error = valid_pred - valid_true
    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)

    # --- NSE ---
    y_mean = np.mean(valid_true)
    numerator = np.sum((valid_true - valid_pred) ** 2)
    denominator = np.sum((valid_true - y_mean) ** 2)
    nse = 1 - numerator / denominator if denominator != 0 else np.nan

    # PCC (空间模式相关系数)
    pcc_scores = []
    for t in range(time_steps):
        pcc, _ = pearsonr(valid_pred[t], valid_true[t])
        pcc_scores.append(pcc)
    mean_pcc = np.mean(pcc_scores)

    # SSIM (结构相似性指数)
    data_range = np.max(valid_true) - np.min(valid_true)
    ssim_scores = []

    for t in range(time_steps):
        pred_frame = pred[t]
        true_frame = true[t]

        # 应用掩码
        pred_masked = np.where(mask, pred_frame, np.nan)
        true_masked = np.where(mask, true_frame, np.nan)

        # 获取有效像素
        valid_mask = ~(np.isnan(pred_masked) | np.isnan(true_masked))
        if np.sum(valid_mask) > 0:
            pred_valid = pred_masked[valid_mask]
            true_valid = true_masked[valid_mask]

            # 重塑为近似正方形计算SSIM
            n_pixels = len(pred_valid)
            side_len = int(np.sqrt(n_pixels))
            if side_len > 1:
                size = side_len * side_len
                pred_reshaped = pred_valid[:size].reshape(side_len, side_len)
                true_reshaped = true_valid[:size].reshape(side_len, side_len)
                try:
                    ssim_val = ssim(pred_reshaped, true_reshaped, win_size=11,data_range=data_range)
                    ssim_scores.append(ssim_val)
                    continue
                except:
                    pass

            # 如果SSIM失败，使用PCC作为近似
            pcc_approx, _ = pearsonr(pred_valid, true_valid)
            ssim_scores.append(max(0, pcc_approx))
        else:
            ssim_scores.append(0.0)

    mean_ssim = np.mean(ssim_scores)

    return rmse, mae, mse, nse, mean_pcc, mean_ssim


if __name__ == '__main__':
    pass
    # mask = np.load('../../Preprocessing/observation/data1/mask.npy')
    #
    # np.random.seed(4)
    # p = np.random.rand(20, 448, 304)
    # p = np.random.rand(20, 448, 304)
    #S
    # rmse, mae, mse, nse = metric(mask, p, p)
    # print(rmse, mae, mse, nse)
