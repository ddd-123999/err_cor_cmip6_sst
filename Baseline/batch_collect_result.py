import numpy as np
from pathlib import Path
from datetime import datetime
import re

# ====================================================================
# 1. 配置
# ====================================================================

# --- 请修改此路径 ---
# 将此路径设置为您 Experiment 文件夹的顶层目录
# 示例: "D:/err_cor_cmip6_sst/Experiment/EXP9/UNet_ssp245"
# 你的输出显示你是在 UNet 目录下运行的，所以 'UNet' 可能就是对的
START_PATH = "ConvLSTM"

# --- 输出文件配置 ---
# 你可以根据实验设置(如seq_len)修改文件名
# 从你的输出看, sl=3, cl=1
SEQ_LEN = 3
PRED_LEN = 1
MODEL_TYPE = "ConvLSTM"

OUTPUT_TXT = f'ConvLSTM/first/{MODEL_TYPE.lower()}_metrics_s{SEQ_LEN}_p{PRED_LEN}_results.txt'
OUTPUT_NPY = f'ConvLSTM/first/{MODEL_TYPE.lower()}_metrics_s{SEQ_LEN}_p{PRED_LEN}_all.npy'


def extract_model_name(setting_name):
    """
    从设置文件夹名称中提取 CMIP6 模型名称。
    示例: 'md-UNet_cn-ACCESS-CM2_bs-32...' -> 'ACCESS-CM2'
    """
    try:
        # 使用正则表达式查找 _cn- 之后到下一个 _ 之前的内容
        match = re.search(r'_cn-(.+?)_', setting_name)
        if match:
            return match.group(1)

        # 备用方案 (如果 _ 是最后一个)
        match_end = re.search(r'_cn-(.+?)$', setting_name)
        if match_end:
            return match_end.group(1)

        print(f"  [Warn] 无法从 {setting_name} 自动解析 'cn-'")
        return None
    except Exception as e:
        print(f"  [Error] 解析 {setting_name} 出错: {e}")
        return None


def collect_and_save_results(base_dir):
    """
    遍历指定目录，查找所有 'test_criteria.npz' 文件，
    提取指标，并保存为标准 TXT 和 NPY 报告。
    """
    print(f"🚀 正在扫描以下目录以查找 {MODEL_TYPE} 实验结果: {base_dir}\n")
    base_path = Path(base_dir)

    if not base_path.is_dir():
        print(f"❌ 错误: 目录未找到 -> {base_path}")
        return

    result_files = list(base_path.rglob('test_criteria.npz'))

    if not result_files:
        print(f"ℹ️ 在 {base_path} 中未找到任何 'test_criteria.npz' 文件。")
        return

    print(f"✅ 找到了 {len(result_files)} 个实验结果。\n")

    all_results = {}
    models_succeeded = []
    models_failed = []

    # 打开 TXT 文件准备写入
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{MODEL_TYPE} (S{SEQ_LEN}/P{PRED_LEN}) 修正后 Metrics 计算结果\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据源: {base_path}\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"{'模型名称':<25} {'RMSE':<10} {'MAE':<10} {'MSE':<10} {'NSE':<10} {'PCC':<10} {'SSIM':<10} {'状态':<10}\n")
        f.write("-" * 115 + "\n")

        for file_path in sorted(result_files):  # 排序以保证顺序

            setting_name = file_path.parent.name
            model_name = extract_model_name(setting_name)

            if model_name is None:
                model_name = setting_name  # 如果解析失败, 使用文件夹名
                models_failed.append(f"{model_name} (无法解析)")

            try:
                data = np.load(file_path, allow_pickle=True)

                if 'results' not in data:
                    print(f"❌ {model_name:<25} - 失败: 'results' 键未在 {file_path} 中找到")
                    models_failed.append(f"{model_name} (键缺失)")
                    f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                            f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 键缺失':<10}\n")
                    f.flush()
                    continue

                metrics = data['results'].item()

                # 存储结果 (使用 .get 确保安全)
                # 注意: 你的文件保存的是 'pcc' 和 'ssim', 不是 'mean_pcc'
                result_data = {
                    "rmse": metrics.get('rmse', np.nan),
                    "mae": metrics.get('mae', np.nan),
                    "mse": metrics.get('mse', np.nan),
                    "nse": metrics.get('nse', np.nan),
                    "mean_pcc": metrics.get('pcc', np.nan),  # 键名统一
                    "mean_ssim": metrics.get('ssim', np.nan)  # 键名统一
                }
                all_results[model_name] = result_data
                models_succeeded.append(model_name)

                # 写入 TXT 文件
                f.write(f"{model_name:<25} "
                        f"{result_data['rmse']:<10.4f} "
                        f"{result_data['mae']:<10.4f} "
                        f"{result_data['mse']:<10.4f} "
                        f"{result_data['nse']:<10.4f} "
                        f"{result_data['mean_pcc']:<10.4f} "
                        f"{result_data['mean_ssim']:<10.4f} "
                        f"{'✓':<10}\n")
                f.flush()
                print(f"✓ {model_name:<25} - RMSE: {result_data['rmse']:.4f}")

            except Exception as e:
                print(f"❌ {model_name:<25} - 失败: {e}")
                models_failed.append(f"{model_name} (读取错误)")
                f.write(f"{model_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} "
                        f"{'N/A':<10} {'N/A':<10} {'N/A':<10} {'✗ 读取错误':<10}\n")
                f.flush()

        # 写入统计摘要
        f.write("\n" + "=" * 80 + "\n")
        f.write("统计摘要\n")
        f.write("=" * 80 + "\n")
        f.write(f"总模型数: {len(result_files)}\n")
        f.write(f"成功: {len(models_succeeded)}\n")
        f.write(f"失败: {len(models_failed)}\n")
        if models_failed:
            f.write("\n失败的条目:\n")
            for model in models_failed:
                f.write(f"  - {model}\n")

    # ========== 保存完整结果 ==========
    np.save(OUTPUT_NPY, all_results)
    print(f"\n✅ 完整结果已保存到: {OUTPUT_NPY}")
    print(f"✅ 文本结果已保存到: {OUTPUT_TXT}")


if __name__ == "__main__":
    collect_and_save_results(START_PATH)