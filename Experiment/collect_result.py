import numpy as np
from pathlib import Path


def load_and_print_results(base_dir):
    """
    遍历指定目录，查找所有 'test_criteria.npz' 文件，
    并打印出其中存储的指标。
    """
    print(f"🚀 正在扫描以下目录以查找实验结果: {base_dir}\n")

    # 将输入的字符串路径转换为 Pathlib 对象
    base_path = Path(base_dir)

    # 检查路径是否存在
    if not base_path.is_dir():
        print(f"❌ 错误: 目录未找到 -> {base_path}")
        print("请检查 'START_PATH' 变量是否设置正确。")
        return

    # 1. 递归搜索 (rglob) 目录下的所有 'test_criteria.npz' 文件
    result_files = list(base_path.rglob('test_criteria.npz'))

    if not result_files:
        print(f"ℹ️ 在 {base_path} 中未找到任何 'test_criteria.npz' 文件。")
        return

    print(f"✅ 找到了 {len(result_files)} 个实验结果。\n")

    for file_path in result_files:
        try:
            # 2. 加载 .npz 文件
            # allow_pickle=True 是必需的，因为字典被保存为对象
            data = np.load(file_path, allow_pickle=True)

            # 3. 提取指标字典
            # 根据 exp.py (line 450-457)，
            # 指标被保存在 'results' 键下
            if 'results' in data:
                # .item() 将 0 维数组转换回 Python 字典
                metrics = data['results'].item()

                # 4. 从路径中提取实验信息
                # file_path.parent 是 'md-...' 文件夹
                setting_folder = file_path.parent
                setting_name = setting_folder.name
                lr_folder = setting_folder.parent.name
                model_folder = setting_folder.parent.parent.name

                # 5. 打印结果
                print("=" * 80)
                print(f"📂 模型:    {model_folder}")
                print(f"📈 学习率:  {lr_folder}")
                print(f"⚙️ 完整设置: {setting_name}")
                print("-" * 80)

                # 使用 .get() 安全地访问键，如果键不存在则返回 'N/A'
                print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}")
                print(f"  MSE:  {metrics.get('mse', 'N/A'):.4f}")
                print(f"  NSE:  {metrics.get('nse', 'N/A'):.4f}")
                print(f"  PCC:  {metrics.get('pcc', 'N/A'):.4f}")
                print(f"  SSIM: {metrics.get('ssim', 'N/A'):.4f}")
                print("=" * 80 + "\n")

            else:
                print(f"⚠️ 在 {file_path} 中未找到 'results' 键。")

        except Exception as e:
            print(f"❌ 读取 {file_path} 时出错: {e}")


if __name__ == "__main__":
    # --- 请修改此路径 ---
    # 将此路径设置为您 Experiment 文件夹的顶层目录
    # 根据您的截图，'../Experiment/EXP9' 似乎是正确的
    START_PATH = "EXP9/ConvLSTM/4GPU/lr=0.001"

    load_and_print_results(START_PATH)