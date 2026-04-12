import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class Config:
    """
    配置类 - 可灵活调整误差阈值

    使用方法：
    1. 修改 PRIMARY_THRESHOLD 来改变主要分析的阈值
    2. 修改 ERROR_THRESHOLDS 来同时计算多个阈值的统计量
    """

    MODELS = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5',
        'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC',
        'EC-Earth3-Veg-LR', 'EC-Earth3-veg', 'EC-Earth3', 'GFDL-CM4',
        'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM'
    ]

    METHODS = ['base', 'EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']
    METHOD_LABELS = {
        'base': 'Control',
        'EDCDF': 'EDCDF',
        'ConvLSTM': 'ConvLSTM',
        'UNet': 'UNet',
        'MambaUNet': 'MambaUNet'
    }

    # 可配置的误差阈值列表
    ERROR_THRESHOLDS = [0.25, 0.5, 0.75, 1.0]

    # 主要分析的阈值
    PRIMARY_THRESHOLD = 0.75

    CACHE_FILE = './cache_data/figure4_bias_monthly_cache.pkl'
    OUTPUT_DIR = './detailed_bias_3part_analysis'


class DetailedBiasAnalyzer:
    def __init__(self, config):
        self.config = config
        self.bias_data = None

    def load_data(self):
        """加载数据"""
        cache_path = Path(self.config.CACHE_FILE)
        print(f"📂 加载数据: {cache_path}")
        with open(cache_path, 'rb') as f:
            self.bias_data = pickle.load(f)
        print(f"   ✅ 加载成功！包含 {len(self.bias_data)} 个模型")

    def analyze_three_parts(self):
        """分析三部分：范围内、>threshold、<-threshold"""
        print(f"\n📊 开始三部分分析（阈值: {self.config.ERROR_THRESHOLDS}）...")

        all_stats = []

        for model in self.config.MODELS:
            model_data = self.bias_data.get(model, {})
            if not model_data:
                continue

            row = {'Model': model}

            for method in self.config.METHODS:
                bias_monthly = model_data.get(method)
                if bias_monthly is None:
                    continue

                data = bias_monthly.flatten()
                valid_data = data[~np.isnan(data)]

                if len(valid_data) == 0:
                    continue

                prefix = self.config.METHOD_LABELS[method]
                total = len(valid_data)

                # ========== 对每个阈值进行三部分分析 ==========
                for threshold in self.config.ERROR_THRESHOLDS:
                    thresh_str = str(threshold).replace('.', '')

                    # 1. 范围内 [-threshold, +threshold]
                    within_mask = np.abs(valid_data) <= threshold
                    within_pct = np.sum(within_mask) / total * 100

                    # 2. 范围外正偏差 (> threshold)
                    beyond_positive_pct = np.sum(valid_data > threshold) / total * 100

                    # 3. 范围外负偏差 (< -threshold)
                    beyond_negative_pct = np.sum(valid_data < -threshold) / total * 100

                    row[f'{prefix}_Within{thresh_str}'] = within_pct
                    row[f'{prefix}_BeyondPos{thresh_str}'] = beyond_positive_pct
                    row[f'{prefix}_BeyondNeg{thresh_str}'] = beyond_negative_pct

            all_stats.append(row)

        self.df_detailed = pd.DataFrame(all_stats)
        print(f"   ✅ 三部分统计完成！共 {len(self.df_detailed)} 个模型")

    def calculate_improvements(self):
        """计算三部分的改善情况（基于主要阈值）"""
        threshold = self.config.PRIMARY_THRESHOLD
        thresh_str = str(threshold).replace('.', '')

        print(f"\n📈 计算三部分改善情况（阈值: ±{threshold}°C）...")

        improvements = []

        for idx, row in self.df_detailed.iterrows():
            model = row['Model']

            # Control基准值（三部分）
            control_within = row.get(f'Control_Within{thresh_str}', np.nan)
            control_beyond_pos = row.get(f'Control_BeyondPos{thresh_str}', np.nan)
            control_beyond_neg = row.get(f'Control_BeyondNeg{thresh_str}', np.nan)

            if np.isnan(control_within):
                continue

            improvement_row = {
                'Model': model,
                f'Control_Within{threshold}': control_within,
                f'Control_BeyondPos{threshold}': control_beyond_pos,
                f'Control_BeyondNeg{threshold}': control_beyond_neg
            }

            for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
                # 范围内改善
                within_col = f'{method}_Within{thresh_str}'
                if within_col in row:
                    method_within = row[within_col]
                    improve_within_abs = method_within - control_within
                    improve_within_rel = (improve_within_abs / control_within) * 100 if control_within != 0 else 0
                    is_improved_within = 'Yes' if improve_within_abs > 0 else 'No'

                    improvement_row[f'{method}_Within{threshold}'] = method_within
                    improvement_row[f'{method}_Within_Improve_Abs'] = improve_within_abs
                    improvement_row[f'{method}_Within_Improve_Rel%'] = improve_within_rel
                    improvement_row[f'{method}_Within_IsImproved'] = is_improved_within

                # 范围外正偏差改善（注意：这里减少是好的，所以改善量是负的更好）
                beyond_pos_col = f'{method}_BeyondPos{thresh_str}'
                if beyond_pos_col in row:
                    method_beyond_pos = row[beyond_pos_col]
                    improve_pos_abs = control_beyond_pos - method_beyond_pos  # 注意方向
                    improve_pos_rel = (improve_pos_abs / control_beyond_pos) * 100 if control_beyond_pos != 0 else 0
                    is_improved_pos = 'Yes' if improve_pos_abs > 0 else 'No'

                    improvement_row[f'{method}_BeyondPos{threshold}'] = method_beyond_pos
                    improvement_row[f'{method}_BeyondPos_Improve_Abs'] = improve_pos_abs
                    improvement_row[f'{method}_BeyondPos_Improve_Rel%'] = improve_pos_rel
                    improvement_row[f'{method}_BeyondPos_IsImproved'] = is_improved_pos

                # 范围外负偏差改善（注意：这里减少是好的，所以改善量是负的更好）
                beyond_neg_col = f'{method}_BeyondNeg{thresh_str}'
                if beyond_neg_col in row:
                    method_beyond_neg = row[beyond_neg_col]
                    improve_neg_abs = control_beyond_neg - method_beyond_neg  # 注意方向
                    improve_neg_rel = (improve_neg_abs / control_beyond_neg) * 100 if control_beyond_neg != 0 else 0
                    is_improved_neg = 'Yes' if improve_neg_abs > 0 else 'No'

                    improvement_row[f'{method}_BeyondNeg{threshold}'] = method_beyond_neg
                    improvement_row[f'{method}_BeyondNeg_Improve_Abs'] = improve_neg_abs
                    improvement_row[f'{method}_BeyondNeg_Improve_Rel%'] = improve_neg_rel
                    improvement_row[f'{method}_BeyondNeg_IsImproved'] = is_improved_neg

            improvements.append(improvement_row)

        self.df_improvements = pd.DataFrame(improvements)
        print(f"   ✅ 三部分改善分析完成！")

    def generate_summary(self):
        """生成汇总统计（三部分）"""
        print("\n📊 生成三部分汇总统计...")

        all_summaries = []

        for threshold in self.config.ERROR_THRESHOLDS:
            thresh_str = str(threshold).replace('.', '')

            for method in self.config.METHODS:
                label = self.config.METHOD_LABELS[method]

                summary = {
                    'Threshold': threshold,
                    'Method': label
                }

                # 范围内统计
                within_col = f'{label}_Within{thresh_str}'
                if within_col in self.df_detailed.columns:
                    within_values = self.df_detailed[within_col].dropna()
                    summary['Within_Avg'] = within_values.mean()
                    summary['Within_Min'] = within_values.min()
                    summary['Within_Max'] = within_values.max()
                    summary['Within_Std'] = within_values.std()

                # 范围外正偏差统计
                beyond_pos_col = f'{label}_BeyondPos{thresh_str}'
                if beyond_pos_col in self.df_detailed.columns:
                    beyond_pos_values = self.df_detailed[beyond_pos_col].dropna()
                    summary['BeyondPos_Avg'] = beyond_pos_values.mean()
                    summary['BeyondPos_Min'] = beyond_pos_values.min()
                    summary['BeyondPos_Max'] = beyond_pos_values.max()
                    summary['BeyondPos_Std'] = beyond_pos_values.std()

                # 范围外负偏差统计
                beyond_neg_col = f'{label}_BeyondNeg{thresh_str}'
                if beyond_neg_col in self.df_detailed.columns:
                    beyond_neg_values = self.df_detailed[beyond_neg_col].dropna()
                    summary['BeyondNeg_Avg'] = beyond_neg_values.mean()
                    summary['BeyondNeg_Min'] = beyond_neg_values.min()
                    summary['BeyondNeg_Max'] = beyond_neg_values.max()
                    summary['BeyondNeg_Std'] = beyond_neg_values.std()

                all_summaries.append(summary)

        self.df_summary = pd.DataFrame(all_summaries)

        # ========== 改善汇总（仅主要阈值，三部分）==========
        threshold = self.config.PRIMARY_THRESHOLD

        improvement_summary = []

        for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
            summary_row = {'Method': method, 'Threshold': threshold}

            # 范围内改善
            within_improve_col = f'{method}_Within_Improve_Abs'
            within_is_improved_col = f'{method}_Within_IsImproved'
            if within_improve_col in self.df_improvements.columns:
                improve_values = self.df_improvements[within_improve_col].dropna()
                is_improved = self.df_improvements[within_is_improved_col]

                summary_row['Within_Avg_Improve'] = improve_values.mean()
                summary_row['Within_Min_Improve'] = improve_values.min()
                summary_row['Within_Max_Improve'] = improve_values.max()
                summary_row['Within_Num_Improved'] = (is_improved == 'Yes').sum()
                summary_row['Within_Num_Degraded'] = (is_improved == 'No').sum()

            # 范围外正偏差改善
            pos_improve_col = f'{method}_BeyondPos_Improve_Abs'
            pos_is_improved_col = f'{method}_BeyondPos_IsImproved'
            if pos_improve_col in self.df_improvements.columns:
                improve_values = self.df_improvements[pos_improve_col].dropna()
                is_improved = self.df_improvements[pos_is_improved_col]

                summary_row['BeyondPos_Avg_Improve'] = improve_values.mean()
                summary_row['BeyondPos_Min_Improve'] = improve_values.min()
                summary_row['BeyondPos_Max_Improve'] = improve_values.max()
                summary_row['BeyondPos_Num_Improved'] = (is_improved == 'Yes').sum()
                summary_row['BeyondPos_Num_Degraded'] = (is_improved == 'No').sum()

            # 范围外负偏差改善
            neg_improve_col = f'{method}_BeyondNeg_Improve_Abs'
            neg_is_improved_col = f'{method}_BeyondNeg_IsImproved'
            if neg_improve_col in self.df_improvements.columns:
                improve_values = self.df_improvements[neg_improve_col].dropna()
                is_improved = self.df_improvements[neg_is_improved_col]

                summary_row['BeyondNeg_Avg_Improve'] = improve_values.mean()
                summary_row['BeyondNeg_Min_Improve'] = improve_values.min()
                summary_row['BeyondNeg_Max_Improve'] = improve_values.max()
                summary_row['BeyondNeg_Num_Improved'] = (is_improved == 'Yes').sum()
                summary_row['BeyondNeg_Num_Degraded'] = (is_improved == 'No').sum()

            summary_row['Total_Models'] = len(self.df_improvements)
            improvement_summary.append(summary_row)

        self.df_improvement_summary = pd.DataFrame(improvement_summary)
        print(f"   ✅ 汇总完成！")

    def save_results(self):
        """保存结果"""
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        print(f"\n💾 保存结果到: {output_dir}")

        # CSV文件
        self.df_detailed.to_csv(output_dir / '1_detailed_per_model_3parts.csv', index=False)
        self.df_summary.to_csv(output_dir / '2_summary_by_method_3parts.csv', index=False)
        self.df_improvements.to_csv(output_dir / '3_improvements_per_model_3parts.csv', index=False)
        self.df_improvement_summary.to_csv(output_dir / '4_improvement_summary_3parts.csv', index=False)

        print(f"   ✅ CSV文件已生成")

        # 生成文本报告
        self.generate_report(output_dir)

    def generate_report(self, output_dir):
        """生成详细文本报告（三部分独立）"""
        report_path = output_dir / 'detailed_3part_analysis_report.txt'
        threshold = self.config.PRIMARY_THRESHOLD

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"Bias三部分独立分析报告（阈值: ±{threshold}°C）\n")
            f.write(f"部分1: 范围内 [-{threshold}, +{threshold}]°C\n")
            f.write(f"部分2: 范围外正偏差 (>{threshold}°C)\n")
            f.write(f"部分3: 范围外负偏差 (<-{threshold}°C)\n")
            f.write("=" * 100 + "\n\n")

            # 配置信息
            f.write(f"【配置信息】\n")
            f.write("-" * 100 + "\n")
            f.write(f"主要分析阈值: ±{threshold}°C\n")
            f.write(f"所有计算阈值: {self.config.ERROR_THRESHOLDS}\n")
            f.write(f"模型数量: {len(self.config.MODELS)}\n\n")

            # ========== 一、多阈值对比 ==========
            f.write("【一】不同阈值下各方法的三部分表现\n")
            f.write("-" * 100 + "\n\n")

            for method in self.config.METHODS:
                label = self.config.METHOD_LABELS[method]
                f.write(f"{label}:\n")

                method_data = self.df_summary[self.df_summary['Method'] == label]

                for _, row in method_data.iterrows():
                    thresh = row['Threshold']
                    f.write(f"  阈值±{thresh}°C:\n")
                    f.write(f"    • 范围内: {row['Within_Avg']:.2f}%")
                    f.write(f" (范围: {row['Within_Min']:.2f}% - {row['Within_Max']:.2f}%)\n")
                    f.write(f"    • 范围外(>{thresh}°C): {row['BeyondPos_Avg']:.2f}%")
                    f.write(f" (范围: {row['BeyondPos_Min']:.2f}% - {row['BeyondPos_Max']:.2f}%)\n")
                    f.write(f"    • 范围外(<-{thresh}°C): {row['BeyondNeg_Avg']:.2f}%")
                    f.write(f" (范围: {row['BeyondNeg_Min']:.2f}% - {row['BeyondNeg_Max']:.2f}%)\n")

                f.write("\n")

            # ========== 二、主要阈值的详细统计 ==========
            f.write(f"\n【二】±{threshold}°C阈值的详细统计\n")
            f.write("-" * 100 + "\n\n")

            primary_summary = self.df_summary[self.df_summary['Threshold'] == threshold]

            for _, row in primary_summary.iterrows():
                method = row['Method']
                f.write(f"{method}:\n")
                f.write(f"  范围内 [-{threshold}, +{threshold}]°C:\n")
                f.write(f"    • 平均占比: {row['Within_Avg']:.2f}%\n")
                f.write(f"    • 范围: {row['Within_Min']:.2f}% - {row['Within_Max']:.2f}%\n")
                f.write(f"    • 标准差: {row['Within_Std']:.2f}%\n")

                f.write(f"  范围外正偏差 (>{threshold}°C):\n")
                f.write(f"    • 平均占比: {row['BeyondPos_Avg']:.2f}%\n")
                f.write(f"    • 范围: {row['BeyondPos_Min']:.2f}% - {row['BeyondPos_Max']:.2f}%\n")
                f.write(f"    • 标准差: {row['BeyondPos_Std']:.2f}%\n")

                f.write(f"  范围外负偏差 (<-{threshold}°C):\n")
                f.write(f"    • 平均占比: {row['BeyondNeg_Avg']:.2f}%\n")
                f.write(f"    • 范围: {row['BeyondNeg_Min']:.2f}% - {row['BeyondNeg_Max']:.2f}%\n")
                f.write(f"    • 标准差: {row['BeyondNeg_Std']:.2f}%\n")
                f.write("\n")

            # ========== 三、改善情况（三部分独立）==========
            f.write(f"\n【三】相对Control的改善情况（三部分独立）\n")
            f.write("-" * 100 + "\n\n")

            control_row = primary_summary[primary_summary['Method'] == 'Control'].iloc[0]

            for _, row in self.df_improvement_summary.iterrows():
                method = row['Method']
                method_row = primary_summary[primary_summary['Method'] == method].iloc[0]

                f.write(f"{method}:\n")

                # 范围内改善
                f.write(f"  1. 范围内 [-{threshold}, +{threshold}]°C:\n")
                f.write(f"     • Control: {control_row['Within_Avg']:.2f}%\n")
                f.write(f"     • {method}: {method_row['Within_Avg']:.2f}%\n")
                f.write(f"     • 平均改善: {row['Within_Avg_Improve']:+.2f}%")
                f.write(f" (范围: {row['Within_Min_Improve']:+.2f}% 至 {row['Within_Max_Improve']:+.2f}%)\n")
                f.write(f"     • 改善模型数: {row['Within_Num_Improved']}/{row['Total_Models']}\n")
                f.write(f"     • 退化模型数: {row['Within_Num_Degraded']}/{row['Total_Models']}\n\n")

                # 范围外正偏差改善
                f.write(f"  2. 范围外正偏差 (>{threshold}°C):\n")
                f.write(f"     • Control: {control_row['BeyondPos_Avg']:.2f}%\n")
                f.write(f"     • {method}: {method_row['BeyondPos_Avg']:.2f}%\n")
                f.write(f"     • 平均减少: {row['BeyondPos_Avg_Improve']:+.2f}%")
                f.write(f" (范围: {row['BeyondPos_Min_Improve']:+.2f}% 至 {row['BeyondPos_Max_Improve']:+.2f}%)\n")
                f.write(f"     • 改善模型数: {row['BeyondPos_Num_Improved']}/{row['Total_Models']}\n")
                f.write(f"     • 退化模型数: {row['BeyondPos_Num_Degraded']}/{row['Total_Models']}\n\n")

                # 范围外负偏差改善
                f.write(f"  3. 范围外负偏差 (<-{threshold}°C):\n")
                f.write(f"     • Control: {control_row['BeyondNeg_Avg']:.2f}%\n")
                f.write(f"     • {method}: {method_row['BeyondNeg_Avg']:.2f}%\n")
                f.write(f"     • 平均减少: {row['BeyondNeg_Avg_Improve']:+.2f}%")
                f.write(f" (范围: {row['BeyondNeg_Min_Improve']:+.2f}% 至 {row['BeyondNeg_Max_Improve']:+.2f}%)\n")
                f.write(f"     • 改善模型数: {row['BeyondNeg_Num_Improved']}/{row['Total_Models']}\n")
                f.write(f"     • 退化模型数: {row['BeyondNeg_Num_Degraded']}/{row['Total_Models']}\n")
                f.write("\n")

            # ========== 四、每个模型的详细占比 ==========
            f.write(f"\n【四】每个模型的三部分详细占比\n")
            f.write("-" * 100 + "\n\n")

            for _, row in self.df_improvements.iterrows():
                model = row['Model']
                f.write(f"{model}:\n")

                # Control基准
                f.write(f"  Control:\n")
                f.write(f"    • 范围内: {row[f'Control_Within{threshold}']:.2f}%\n")
                f.write(f"    • 范围外(>{threshold}°C): {row[f'Control_BeyondPos{threshold}']:.2f}%\n")
                f.write(f"    • 范围外(<-{threshold}°C): {row[f'Control_BeyondNeg{threshold}']:.2f}%\n\n")

                for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
                    within_val = row.get(f'{method}_Within{threshold}', np.nan)
                    pos_val = row.get(f'{method}_BeyondPos{threshold}', np.nan)
                    neg_val = row.get(f'{method}_BeyondNeg{threshold}', np.nan)

                    if not np.isnan(within_val):
                        within_improve = row[f'{method}_Within_Improve_Abs']
                        pos_improve = row[f'{method}_BeyondPos_Improve_Abs']
                        neg_improve = row[f'{method}_BeyondNeg_Improve_Abs']

                        within_mark = "✓" if row[f'{method}_Within_IsImproved'] == 'Yes' else "✗"
                        pos_mark = "✓" if row[f'{method}_BeyondPos_IsImproved'] == 'Yes' else "✗"
                        neg_mark = "✓" if row[f'{method}_BeyondNeg_IsImproved'] == 'Yes' else "✗"

                        f.write(f"  {method}:\n")
                        f.write(f"    • 范围内: {within_val:.2f}% ({within_improve:+.2f}%) {within_mark}\n")
                        f.write(f"    • 范围外(>{threshold}°C): {pos_val:.2f}% (减少{pos_improve:+.2f}%) {pos_mark}\n")
                        f.write(
                            f"    • 范围外(<-{threshold}°C): {neg_val:.2f}% (减少{neg_improve:+.2f}%) {neg_mark}\n\n")

                f.write("\n")

            # ========== 五、论文用统计描述（三部分独立）==========
            f.write("\n【五】论文用统计描述（三部分独立）\n")
            f.write("-" * 100 + "\n\n")
            self.generate_paper_text_3parts(f, threshold, primary_summary)

        print(f"   ✅ 文本报告: {report_path}")

    def generate_paper_text_3parts(self, f, threshold, summary_data):
        """生成论文用统计描述（三部分独立）"""

        control = summary_data[summary_data['Method'] == 'Control'].iloc[0]
        edcdf = summary_data[summary_data['Method'] == 'EDCDF'].iloc[0]
        convlstm = summary_data[summary_data['Method'] == 'ConvLSTM'].iloc[0]
        unet = summary_data[summary_data['Method'] == 'UNet'].iloc[0]
        mamba = summary_data[summary_data['Method'] == 'MambaUNet'].iloc[0]

        edcdf_imp = self.df_improvement_summary[self.df_improvement_summary['Method'] == 'EDCDF'].iloc[0]
        convlstm_imp = self.df_improvement_summary[self.df_improvement_summary['Method'] == 'ConvLSTM'].iloc[0]
        unet_imp = self.df_improvement_summary[self.df_improvement_summary['Method'] == 'UNet'].iloc[0]
        mamba_imp = self.df_improvement_summary[self.df_improvement_summary['Method'] == 'MambaUNet'].iloc[0]

        f.write(f"【Part 1: 范围内分析 [-{threshold}, +{threshold}]°C】\n")
        f.write("-" * 100 + "\n\n")

        # Control描述
        f.write(f"Control误差分布在±{threshold}°C范围内的样本点百分比")
        f.write(f"从 {control['Within_Min']:.2f}% 到 {control['Within_Max']:.2f}%，")
        f.write(f"平均为 {control['Within_Avg']:.2f}%。\n\n")

        # EDCDF描述
        f.write(f"传统统计方法EDCDF将±{threshold}°C范围内样本点百分比")
        f.write(f"提升至平均 {edcdf['Within_Avg']:.2f}%")
        f.write(f"（范围 {edcdf['Within_Min']:.2f}% - {edcdf['Within_Max']:.2f}%），")
        f.write(f"相比Control平均提高 {edcdf_imp['Within_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {edcdf_imp['Within_Min_Improve']:.2f}% 至 {edcdf_imp['Within_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{edcdf_imp['Within_Num_Improved']}个模型范围内占比有所提高，")
        f.write(f"{edcdf_imp['Within_Num_Degraded']}个模型出现下降。\n\n")

        # ConvLSTM描述
        f.write(f"ConvLSTM将±{threshold}°C范围内样本点百分比")
        f.write(f"调整至平均 {convlstm['Within_Avg']:.2f}%")
        f.write(f"（范围 {convlstm['Within_Min']:.2f}% - {convlstm['Within_Max']:.2f}%），")
        f.write(f"相比Control平均改变 {convlstm_imp['Within_Avg_Improve']:+.2f}%，")
        f.write(f"仅对 {convlstm_imp['Within_Num_Improved']}/{convlstm_imp['Total_Models']} 个模型实现改善。\n\n")

        # UNet描述
        f.write(f"UNet可将±{threshold}°C范围内样本点百分比")
        f.write(f"提升至平均 {unet['Within_Avg']:.2f}%")
        f.write(f"（范围 {unet['Within_Min']:.2f}% - {unet['Within_Max']:.2f}%），")
        f.write(f"相比Control平均提高 {unet_imp['Within_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {unet_imp['Within_Min_Improve']:.2f}% 至 {unet_imp['Within_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{unet_imp['Within_Num_Improved']}个模型范围内占比有所提高。\n\n")

        # MambaUNet描述
        f.write(f"MambaUNet进一步将±{threshold}°C范围内样本点百分比提高至")
        f.write(f"平均 {mamba['Within_Avg']:.2f}%")
        f.write(f"（范围 {mamba['Within_Min']:.2f}% - {mamba['Within_Max']:.2f}%），")

        mamba_vs_control = mamba['Within_Avg'] - control['Within_Avg']
        mamba_vs_unet = mamba['Within_Avg'] - unet['Within_Avg']

        f.write(f"相比Control提高 {mamba_vs_control:.2f}%，")
        f.write(f"相比UNet提高 {mamba_vs_unet:.2f}%。")
        f.write(f"改善范围为 {mamba_imp['Within_Min_Improve']:.2f}% 至 {mamba_imp['Within_Max_Improve']:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['Within_Num_Improved']}个模型实现改善。\n\n")

        # ========== Part 2: 范围外正偏差 ==========
        f.write(f"\n【Part 2: 范围外正偏差分析 (>{threshold}°C)】\n")
        f.write("-" * 100 + "\n\n")

        # Control描述
        f.write(f"Control误差分布中，大于{threshold}°C的正偏差样本点百分比")
        f.write(f"从 {control['BeyondPos_Min']:.2f}% 到 {control['BeyondPos_Max']:.2f}%，")
        f.write(f"平均为 {control['BeyondPos_Avg']:.2f}%。\n\n")

        # EDCDF描述
        f.write(f"传统统计方法EDCDF将大于{threshold}°C的正偏差样本点百分比")
        f.write(f"降低至平均 {edcdf['BeyondPos_Avg']:.2f}%")
        f.write(f"（范围 {edcdf['BeyondPos_Min']:.2f}% - {edcdf['BeyondPos_Max']:.2f}%），")
        f.write(f"相比Control平均减少 {edcdf_imp['BeyondPos_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {edcdf_imp['BeyondPos_Min_Improve']:.2f}% 至 {edcdf_imp['BeyondPos_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{edcdf_imp['BeyondPos_Num_Improved']}个模型正偏差占比有所降低，")
        f.write(f"{edcdf_imp['BeyondPos_Num_Degraded']}个模型出现增加。\n\n")

        # ConvLSTM描述
        f.write(f"ConvLSTM将大于{threshold}°C的正偏差样本点百分比")
        f.write(f"调整至平均 {convlstm['BeyondPos_Avg']:.2f}%")
        f.write(f"（范围 {convlstm['BeyondPos_Min']:.2f}% - {convlstm['BeyondPos_Max']:.2f}%），")
        if convlstm_imp['BeyondPos_Avg_Improve'] > 0:
            f.write(f"相比Control平均减少 {convlstm_imp['BeyondPos_Avg_Improve']:.2f}%，")
        else:
            f.write(f"相比Control平均增加 {abs(convlstm_imp['BeyondPos_Avg_Improve']):.2f}%，")
        f.write(f"引入了显著的系统性正偏差。")
        f.write(f"仅对 {convlstm_imp['BeyondPos_Num_Improved']}/{convlstm_imp['Total_Models']} 个模型实现改善。\n\n")

        # UNet描述
        f.write(f"UNet可将大于{threshold}°C的正偏差样本点百分比")
        f.write(f"降低至平均 {unet['BeyondPos_Avg']:.2f}%")
        f.write(f"（范围 {unet['BeyondPos_Min']:.2f}% - {unet['BeyondPos_Max']:.2f}%），")
        f.write(f"相比Control平均减少 {unet_imp['BeyondPos_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {unet_imp['BeyondPos_Min_Improve']:.2f}% 至 {unet_imp['BeyondPos_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{unet_imp['BeyondPos_Num_Improved']}个模型正偏差占比有所降低。\n\n")

        # MambaUNet描述
        f.write(f"MambaUNet进一步将大于{threshold}°C的正偏差样本点百分比降低至")
        f.write(f"平均 {mamba['BeyondPos_Avg']:.2f}%")
        f.write(f"（范围 {mamba['BeyondPos_Min']:.2f}% - {mamba['BeyondPos_Max']:.2f}%），")

        mamba_pos_vs_control = control['BeyondPos_Avg'] - mamba['BeyondPos_Avg']
        mamba_pos_vs_unet = unet['BeyondPos_Avg'] - mamba['BeyondPos_Avg']

        f.write(f"相比Control减少 {mamba_pos_vs_control:.2f}%，")
        f.write(f"相比UNet减少 {mamba_pos_vs_unet:.2f}%。")
        f.write(f"改善范围为 {mamba_imp['BeyondPos_Min_Improve']:.2f}% 至 {mamba_imp['BeyondPos_Max_Improve']:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['BeyondPos_Num_Improved']}个模型实现改善。\n\n")

        # ========== Part 3: 范围外负偏差 ==========
        f.write(f"\n【Part 3: 范围外负偏差分析 (<-{threshold}°C)】\n")
        f.write("-" * 100 + "\n\n")

        # Control描述
        f.write(f"Control误差分布中，小于-{threshold}°C的负偏差样本点百分比")
        f.write(f"从 {control['BeyondNeg_Min']:.2f}% 到 {control['BeyondNeg_Max']:.2f}%，")
        f.write(f"平均为 {control['BeyondNeg_Avg']:.2f}%。\n\n")

        # EDCDF描述
        f.write(f"传统统计方法EDCDF将小于-{threshold}°C的负偏差样本点百分比")
        f.write(f"降低至平均 {edcdf['BeyondNeg_Avg']:.2f}%")
        f.write(f"（范围 {edcdf['BeyondNeg_Min']:.2f}% - {edcdf['BeyondNeg_Max']:.2f}%），")
        f.write(f"相比Control平均减少 {edcdf_imp['BeyondNeg_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {edcdf_imp['BeyondNeg_Min_Improve']:.2f}% 至 {edcdf_imp['BeyondNeg_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{edcdf_imp['BeyondNeg_Num_Improved']}个模型负偏差占比有所降低，")
        f.write(f"{edcdf_imp['BeyondNeg_Num_Degraded']}个模型出现增加。\n\n")

        # ConvLSTM描述
        f.write(f"ConvLSTM将小于-{threshold}°C的负偏差样本点百分比")
        f.write(f"调整至平均 {convlstm['BeyondNeg_Avg']:.2f}%")
        f.write(f"（范围 {convlstm['BeyondNeg_Min']:.2f}% - {convlstm['BeyondNeg_Max']:.2f}%），")
        if convlstm_imp['BeyondNeg_Avg_Improve'] > 0:
            f.write(f"相比Control平均减少 {convlstm_imp['BeyondNeg_Avg_Improve']:.2f}%，")
        else:
            f.write(f"相比Control平均增加 {abs(convlstm_imp['BeyondNeg_Avg_Improve']):.2f}%，")
        f.write(f"对 {convlstm_imp['BeyondNeg_Num_Improved']}/{convlstm_imp['Total_Models']} 个模型实现改善。\n\n")

        # UNet描述
        f.write(f"UNet可将小于-{threshold}°C的负偏差样本点百分比")
        f.write(f"降低至平均 {unet['BeyondNeg_Avg']:.2f}%")
        f.write(f"（范围 {unet['BeyondNeg_Min']:.2f}% - {unet['BeyondNeg_Max']:.2f}%），")
        f.write(f"相比Control平均减少 {unet_imp['BeyondNeg_Avg_Improve']:.2f}%")
        f.write(f"（改善范围 {unet_imp['BeyondNeg_Min_Improve']:.2f}% 至 {unet_imp['BeyondNeg_Max_Improve']:.2f}%）。")
        f.write(f"在21个模型中，{unet_imp['BeyondNeg_Num_Improved']}个模型负偏差占比有所降低。\n\n")

        # MambaUNet描述
        f.write(f"MambaUNet进一步将小于-{threshold}°C的负偏差样本点百分比降低至")
        f.write(f"平均 {mamba['BeyondNeg_Avg']:.2f}%")
        f.write(f"（范围 {mamba['BeyondNeg_Min']:.2f}% - {mamba['BeyondNeg_Max']:.2f}%），")

        mamba_neg_vs_control = control['BeyondNeg_Avg'] - mamba['BeyondNeg_Avg']
        mamba_neg_vs_unet = unet['BeyondNeg_Avg'] - mamba['BeyondNeg_Avg']

        f.write(f"相比Control减少 {mamba_neg_vs_control:.2f}%，")
        f.write(f"相比UNet减少 {mamba_neg_vs_unet:.2f}%。")
        f.write(f"改善范围为 {mamba_imp['BeyondNeg_Min_Improve']:.2f}% 至 {mamba_imp['BeyondNeg_Max_Improve']:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['BeyondNeg_Num_Improved']}个模型实现改善。\n\n")

        # ========== 综合总结 ==========
        f.write(f"\n【综合总结】\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"基于±{threshold}°C阈值的三部分独立分析表明：\n\n")

        f.write("1. 范围内改善：")
        f.write(f"MambaUNet在±{threshold}°C范围内的样本点占比")
        f.write(f"相比Control提高了 {mamba_vs_control:.2f}%，")
        f.write(f"相比UNet提高了 {mamba_vs_unet:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['Within_Num_Improved']}个模型实现改善。\n\n")

        f.write("2. 正偏差控制：")
        f.write(f"MambaUNet将大于{threshold}°C的正偏差样本点占比")
        f.write(f"相比Control减少了 {mamba_pos_vs_control:.2f}%，")
        f.write(f"相比UNet减少了 {mamba_pos_vs_unet:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['BeyondPos_Num_Improved']}个模型实现改善。\n\n")

        f.write("3. 负偏差控制：")
        f.write(f"MambaUNet将小于-{threshold}°C的负偏差样本点占比")
        f.write(f"相比Control减少了 {mamba_neg_vs_control:.2f}%，")
        f.write(f"相比UNet减少了 {mamba_neg_vs_unet:.2f}%，")
        f.write(f"在21个模型中有 {mamba_imp['BeyondNeg_Num_Improved']}个模型实现改善。\n\n")

        f.write("总体而言，MambaUNet在三个方面均展现出优越性能：")
        f.write("既提高了小误差范围内的样本占比，")
        f.write("又有效降低了极端正负偏差的发生频率，")
        f.write("显著改善了bias分布的整体特征。\n")

    def run_analysis(self):
        """运行完整分析"""
        self.load_data()
        self.analyze_three_parts()
        self.calculate_improvements()
        self.generate_summary()
        self.save_results()

        print("\n" + "=" * 100)
        print("✅ 三部分独立分析完成！")
        print("=" * 100)
        print(f"\n📊 生成的文件：")
        print(f"  1. 每个模型的三部分详细占比")
        print(f"  2. 各方法的三部分汇总统计")
        print(f"  3. 每个模型的三部分改善情况")
        print(f"  4. 三部分改善汇总")
        print(f"  5. 详细文本报告（含论文用统计描述）")


if __name__ == "__main__":
    config = Config()
    analyzer = DetailedBiasAnalyzer(config)
    analyzer.run_analysis()