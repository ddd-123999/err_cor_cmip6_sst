import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class BiasAnalysisReport:
    """读取缓存数据并生成bias统计报告"""

    def __init__(self):
        self.config = self._get_config()
        self.mean_bias_data = None
        self.detailed_bias_data = None

    def _get_config(self):
        """配置类"""

        class Config:
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

            # 缓存路径
            MEAN_BIAS_CACHE = './cache_data/mean_bias_all_models.pkl'
            BIAS_CACHE_DIR = './cache_data'

            # 方法转换字典（大小写转换）
            METHOD_KEY_MAP = {
                'EDCDF': 'EDCDF',
                'ConvLSTM': 'ConvLSTM',
                'UNet': 'UNet',
                'MambaUNet': 'MambaUNet'
            }

        return Config()

    def load_mean_bias_data(self):
        """加载平均bias缓存数据"""
        cache_path = Path(self.config.MEAN_BIAS_CACHE)

        if not cache_path.exists():
            print(f"❌ 缓存文件不存在: {cache_path}")
            print("请先运行 figure4_day.py 生成缓存数据")
            return False

        try:
            with open(cache_path, 'rb') as f:
                self.mean_bias_data = pickle.load(f)
            print(f"✅ 成功加载平均bias数据: {len(self.mean_bias_data)} 个模型")
            return True
        except Exception as e:
            print(f"❌ 加载缓存失败: {e}")
            return False

    def analyze_bias_statistics(self):
        """分析bias统计信息"""
        print("\n📊 开始分析bias统计信息...")

        if not self.mean_bias_data:
            print("❌ 没有可用的数据")
            return None

        # 初始化统计字典
        stats = {}
        for method in self.config.METHODS:
            method_label = self.config.METHOD_LABELS[method]
            stats[method_label] = {
                'values': [],
                'models': []
            }

        # 收集每个方法的所有bias值
        for model, method_dict in self.mean_bias_data.items():
            for method_key, bias_value in method_dict.items():
                method_label = self.config.METHOD_LABELS.get(method_key)
                if method_label and method_label in stats:
                    stats[method_label]['values'].append(bias_value)
                    stats[method_label]['models'].append(model)

        # 计算统计量
        results = {}
        for method_label, data in stats.items():
            if data['values']:
                values = np.array(data['values'])
                results[method_label] = {
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'num_positive': np.sum(values > 0),
                    'num_negative': np.sum(values < 0),
                    'num_total': len(values),
                    'values': values,
                    'models': data['models']
                }

        return results

    def analyze_improvements(self):
        """分析改善情况"""
        print("\n📈 分析bias改善情况...")

        if not self.mean_bias_data:
            return None

        improvements = {}

        for model in self.config.MODELS:
            if model not in self.mean_bias_data:
                continue

            model_data = self.mean_bias_data[model]

            if 'base' not in model_data:
                continue

            control_bias = model_data['base']

            # 检查每种方法
            for method_key in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
                # 查找对应的键（可能大小写不一致）
                found_key = None
                for key in model_data.keys():
                    if key.lower() == method_key.lower():
                        found_key = key
                        break

                if found_key is not None:
                    method_bias = model_data[found_key]
                    improvement = control_bias - method_bias  # 正值表示改善
                    abs_bias = abs(method_bias)

                    if model not in improvements:
                        improvements[model] = {}

                    improvements[model][method_key] = {
                        'control_bias': control_bias,
                        'method_bias': method_bias,
                        'improvement': improvement,
                        'abs_bias': abs_bias,
                        'is_positive_control': control_bias > 0,
                        'is_positive_method': method_bias > 0,
                        'control_sign': 'positive' if control_bias > 0 else 'negative',
                        'method_sign': 'positive' if method_bias > 0 else 'negative'
                    }

        return improvements

    def find_best_improvements(self, improvements):
        """找出最佳改善"""
        best_improvements = {}

        for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
            # 找出改善最大的模型（改善值最大的）
            max_improvement = -float('inf')
            best_model_improve = None

            # 找出bias绝对值最小的模型
            min_abs_bias = float('inf')
            best_model_abs = None

            for model, data in improvements.items():
                if method in data:
                    # 改善最大
                    if data[method]['improvement'] > max_improvement:
                        max_improvement = data[method]['improvement']
                        best_model_improve = model

                    # bias绝对值最小
                    if data[method]['abs_bias'] < min_abs_bias:
                        min_abs_bias = data[method]['abs_bias']
                        best_model_abs = model

            best_improvements[method] = {
                'best_improvement_model': best_model_improve,
                'best_improvement_value': max_improvement if best_model_improve else None,
                'lowest_bias_model': best_model_abs,
                'lowest_bias_value': min_abs_bias if best_model_abs else None
            }

        return best_improvements

    def analyze_bias_sign_changes(self, improvements):
        """分析bias符号变化 - 重新实现"""
        print("\n🔄 分析bias符号变化...")

        if not improvements:
            print("❌ 没有improvements数据")
            return None

        sign_changes = {}

        # 初始化统计
        for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
            sign_changes[method] = {
                'positive_to_positive': 0,
                'positive_to_negative': 0,
                'negative_to_positive': 0,
                'negative_to_negative': 0,
                'total': 0,
                'sign_changed': 0,
                'sign_unchanged': 0
            }

        # 统计每个模型的符号变化
        model_sign_details = {}

        for model in self.config.MODELS:
            if model not in improvements:
                continue

            model_details = {'model': model}

            for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
                if method in improvements[model]:
                    data = improvements[model][method]
                    control_sign = data['control_sign']
                    method_sign = data['method_sign']

                    # 更新统计
                    key = f"{control_sign}_to_{method_sign}"
                    if key == 'positive_to_positive':
                        sign_changes[method]['positive_to_positive'] += 1
                    elif key == 'positive_to_negative':
                        sign_changes[method]['positive_to_negative'] += 1
                    elif key == 'negative_to_positive':
                        sign_changes[method]['negative_to_positive'] += 1
                    elif key == 'negative_to_negative':
                        sign_changes[method]['negative_to_negative'] += 1

                    sign_changes[method]['total'] += 1

                    # 记录模型详情
                    model_details[f'{method}_control_sign'] = control_sign
                    model_details[f'{method}_method_sign'] = method_sign
                    model_details[f'{method}_changed'] = (control_sign != method_sign)

            model_sign_details[model] = model_details

        # 计算符号变化总结
        for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
            sc = sign_changes[method]
            sc['sign_changed'] = sc['positive_to_negative'] + sc['negative_to_positive']
            sc['sign_unchanged'] = sc['positive_to_positive'] + sc['negative_to_negative']

            if sc['total'] > 0:
                sc['changed_percent'] = sc['sign_changed'] / sc['total'] * 100
                sc['unchanged_percent'] = sc['sign_unchanged'] / sc['total'] * 100
            else:
                sc['changed_percent'] = 0
                sc['unchanged_percent'] = 0

        return sign_changes, model_sign_details

    def generate_detailed_bias_ranges(self):
        """生成详细的bias范围报告"""
        print("\n📏 生成详细的bias范围报告...")

        if not self.mean_bias_data:
            return None

        # 收集所有bias值（包括每个模型的每个方法）
        bias_ranges = {}

        for model in self.config.MODELS:
            if model not in self.mean_bias_data:
                continue

            model_data = self.mean_bias_data[model]
            bias_ranges[model] = {}

            for method_key, method_label in self.config.METHOD_LABELS.items():
                if method_key in model_data:
                    bias_ranges[model][method_label] = model_data[method_key]

        return bias_ranges

    def generate_report(self):
        """生成完整报告"""
        if not self.load_mean_bias_data():
            return

        print("\n" + "=" * 80)
        print("           BIAS 统计报告")
        print("=" * 80)

        # 1. 基本统计
        stats = self.analyze_bias_statistics()
        improvements = self.analyze_improvements()
        best_improvements = self.find_best_improvements(improvements)

        # 2. 符号变化分析（重新实现）
        sign_changes, model_sign_details = self.analyze_bias_sign_changes(improvements)

        # 3. 生成bias范围报告
        bias_ranges = self.generate_detailed_bias_ranges()

        # 生成文本报告
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("BIAS 统计报告")
        report_lines.append("=" * 80)
        report_lines.append("")

        # 一、总体统计
        report_lines.append("一、各方法bias总体统计")
        report_lines.append("-" * 80)

        for method_label in ['Control', 'EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
            if method_label in stats:
                s = stats[method_label]
                report_lines.append(f"{method_label}:")
                report_lines.append(f"  • 平均bias: {s['mean']:+.3f}°C")
                report_lines.append(f"  • 范围: {s['min']:+.3f}°C 到 {s['max']:+.3f}°C")
                report_lines.append(f"  • 标准差: {s['std']:.3f}°C")
                report_lines.append(
                    f"  • 正偏差模型数: {s['num_positive']}/{s['num_total']} ({s['num_positive'] / s['num_total'] * 100:.1f}%)")
                report_lines.append(
                    f"  • 负偏差模型数: {s['num_negative']}/{s['num_total']} ({s['num_negative'] / s['num_total'] * 100:.1f}%)")
                report_lines.append("")

        # 二、改善情况
        report_lines.append("\n二、UNet和MambaUNet改善情况")
        report_lines.append("-" * 80)

        # Control平均bias
        control_mean = stats['Control']['mean']

        # UNet改善
        if 'UNet' in stats:
            unet_mean = stats['UNet']['mean']
            unet_improvement = control_mean - unet_mean
            unet_improvement_abs = abs(control_mean) - abs(unet_mean)

            report_lines.append(f"UNet相比Control，将平均bias从 {control_mean:+.3f}°C 降低到 {unet_mean:+.3f}°C")
            report_lines.append(f"平均缩小: {unet_improvement:.3f}°C")

            # UNet的bias范围
            if 'UNet' in stats:
                unet_min = stats['UNet']['min']
                unet_max = stats['UNet']['max']
                report_lines.append(
                    f"UNet将bias范围从 {stats['Control']['min']:+.3f}°C到{stats['Control']['max']:+.3f}°C")
                report_lines.append(f"减小到 {unet_min:+.3f}°C到{unet_max:+.3f}°C")

        # MambaUNet改善
        if 'MambaUNet' in stats:
            mamba_mean = stats['MambaUNet']['mean']
            mamba_improvement = control_mean - mamba_mean
            mamba_improvement_abs = abs(control_mean) - abs(mamba_mean)

            report_lines.append(f"MambaUNet相比Control，将平均bias从 {control_mean:+.3f}°C 降低到 {mamba_mean:+.3f}°C")
            report_lines.append(f"平均缩小: {mamba_improvement:.3f}°C")

            # MambaUNet的bias范围
            if 'MambaUNet' in stats:
                mamba_min = stats['MambaUNet']['min']
                mamba_max = stats['MambaUNet']['max']
                report_lines.append(f"MambaUNet将bias范围减小到 {mamba_min:+.3f}°C到{mamba_max:+.3f}°C")

            # 与UNet比较
            if 'UNet' in stats:
                mamba_vs_unet = abs(unet_mean) - abs(mamba_mean)
                report_lines.append(f"MambaUNet相比UNet，进一步缩小bias绝对值: {mamba_vs_unet:.3f}°C")

        report_lines.append("")

        # 三、最佳改善模型
        report_lines.append("三、最佳改善情况")
        report_lines.append("-" * 80)

        for method in ['UNet', 'MambaUNet']:
            if method in best_improvements:
                best = best_improvements[method]
                report_lines.append(f"{method}:")

                if best['best_improvement_model']:
                    model = best['best_improvement_model']
                    improve_val = best['best_improvement_value']
                    control_bias = improvements[model][method]['control_bias']
                    method_bias = improvements[model][method]['method_bias']
                    report_lines.append(f"  • 改善最大的模型: {model}")
                    report_lines.append(f"    Control: {control_bias:+.3f}°C → {method}: {method_bias:+.3f}°C")
                    report_lines.append(f"    改善幅度: {improve_val:.3f}°C")

                if best['lowest_bias_model']:
                    model = best['lowest_bias_model']
                    bias_val = best['lowest_bias_value']
                    method_bias = improvements[model][method]['method_bias']
                    report_lines.append(f"  • 矫正后bias绝对值最小的模型: {model}")
                    report_lines.append(f"    bias值: {method_bias:+.3f}°C, 绝对值: {bias_val:.3f}°C")

                report_lines.append("")

        # 四、bias符号变化分析
        report_lines.append("\n四、bias符号变化分析")
        report_lines.append("-" * 80)

        if sign_changes:
            # Control符号统计
            control_pos = stats['Control']['num_positive']
            control_neg = stats['Control']['num_negative']
            control_total = stats['Control']['num_total']

            report_lines.append(f"Control中:")
            report_lines.append(
                f"  • 正偏差模型数: {control_pos}/{control_total} ({control_pos / control_total * 100:.1f}%)")
            report_lines.append(
                f"  • 负偏差模型数: {control_neg}/{control_total} ({control_neg / control_total * 100:.1f}%)")
            report_lines.append("")

            for method in ['EDCDF', 'ConvLSTM', 'UNet', 'MambaUNet']:
                if method in sign_changes:
                    sc = sign_changes[method]
                    report_lines.append(f"{method}:")
                    report_lines.append(
                        f"  正→正: {sc['positive_to_positive']}个 ({sc['positive_to_positive'] / sc['total'] * 100:.1f}%)")
                    report_lines.append(
                        f"  正→负: {sc['positive_to_negative']}个 ({sc['positive_to_negative'] / sc['total'] * 100:.1f}%)")
                    report_lines.append(
                        f"  负→正: {sc['negative_to_positive']}个 ({sc['negative_to_positive'] / sc['total'] * 100:.1f}%)")
                    report_lines.append(
                        f"  负→负: {sc['negative_to_negative']}个 ({sc['negative_to_negative'] / sc['total'] * 100:.1f}%)")

                    # 符号变化总结
                    changed_count = sc['sign_changed']
                    unchanged_count = sc['sign_unchanged']

                    if changed_count == 0:
                        report_lines.append(f"  ✅ 所有模型保持原符号，未改变高估低估特征")
                    else:
                        report_lines.append(
                            f"  ⚠️  有{changed_count}个模型改变了符号特征 ({changed_count / sc['total'] * 100:.1f}%)")
                        report_lines.append(
                            f"     其中: 正→负: {sc['positive_to_negative']}个, 负→正: {sc['negative_to_positive']}个")

                    report_lines.append("")

        # 五、ConvLSTM特殊分析
        report_lines.append("\n五、ConvLSTM特殊分析")
        report_lines.append("-" * 80)

        if 'ConvLSTM' in stats:
            convlstm_stats = stats['ConvLSTM']
            convlstm_pos = convlstm_stats['num_positive']
            convlstm_total = convlstm_stats['num_total']

            if convlstm_pos == convlstm_total:
                report_lines.append("⚠️ ConvLSTM矫正使得所有21个模型产生正偏差 (100%)")
                report_lines.append("   这表明ConvLSTM引入了系统性的正偏差")
            elif convlstm_pos == 0:
                report_lines.append("⚠️ ConvLSTM矫正使得所有21个模型产生负偏差 (100%)")
                report_lines.append("   这表明ConvLSTM引入了系统性的负偏差")
            else:
                report_lines.append(
                    f"ConvLSTM: 正偏差模型{convlstm_pos}个 ({convlstm_pos / convlstm_total * 100:.1f}%)，负偏差模型{convlstm_total - convlstm_pos}个 ({(convlstm_total - convlstm_pos) / convlstm_total * 100:.1f}%)")

        # 保存报告
        output_dir = Path('./bias_analysis_reports')
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / 'bias_statistical_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print('\n'.join(report_lines))
        print("\n" + "=" * 80)
        print(f"✅ 报告已保存到: {report_file}")
        print("=" * 80)

        # 生成CSV数据文件
        self.generate_csv_files(stats, improvements, best_improvements, sign_changes, model_sign_details, bias_ranges,
                                output_dir)

        return report_lines

    def generate_csv_files(self, stats, improvements, best_improvements, sign_changes, model_sign_details, bias_ranges,
                           output_dir):
        """生成CSV数据文件"""

        # 1. 各方法统计表
        stats_df = pd.DataFrame([
            {
                'Method': method,
                'Mean_Bias': data['mean'],
                'Min_Bias': data['min'],
                'Max_Bias': data['max'],
                'Std_Bias': data['std'],
                'Num_Positive': data['num_positive'],
                'Num_Negative': data['num_negative'],
                'Num_Total': data['num_total'],
                'Percent_Positive': data['num_positive'] / data['num_total'] * 100,
                'Percent_Negative': data['num_negative'] / data['num_total'] * 100
            }
            for method, data in stats.items()
        ])
        stats_df.to_csv(output_dir / '1_method_statistics.csv', index=False)

        # 2. 每个模型的详细bias数据
        model_data = []
        for model in self.config.MODELS:
            if model in self.mean_bias_data:
                row = {'Model': model}
                for method_key, bias_value in self.mean_bias_data[model].items():
                    method_label = self.config.METHOD_LABELS.get(method_key)
                    if method_label:
                        row[method_label] = bias_value
                model_data.append(row)

        model_df = pd.DataFrame(model_data)
        model_df.to_csv(output_dir / '2_model_bias_details.csv', index=False)

        # 3. 改善数据
        improve_data = []
        for model in self.config.MODELS:
            if model in improvements:
                for method, data in improvements[model].items():
                    improve_data.append({
                        'Model': model,
                        'Method': method,
                        'Control_Bias': data['control_bias'],
                        'Method_Bias': data['method_bias'],
                        'Improvement': data['improvement'],
                        'Abs_Bias': data['abs_bias'],
                        'Control_Sign': 'Positive' if data['is_positive_control'] else 'Negative',
                        'Method_Sign': 'Positive' if data['is_positive_method'] else 'Negative',
                        'Sign_Changed': data['control_sign'] != data['method_sign']
                    })

        improve_df = pd.DataFrame(improve_data)
        improve_df.to_csv(output_dir / '3_improvement_details.csv', index=False)

        # 4. 符号变化统计
        sign_df = pd.DataFrame([
            {
                'Method': method,
                'Positive_to_Positive': sc['positive_to_positive'],
                'Positive_to_Negative': sc['positive_to_negative'],
                'Negative_to_Positive': sc['negative_to_positive'],
                'Negative_to_Negative': sc['negative_to_negative'],
                'Total': sc['total'],
                'Sign_Changed': sc['sign_changed'],
                'Sign_Unchanged': sc['sign_unchanged'],
                'Percent_Changed': sc['changed_percent'],
                'Percent_Unchanged': sc['unchanged_percent']
            }
            for method, sc in sign_changes.items()
        ])
        sign_df.to_csv(output_dir / '4_sign_changes_summary.csv', index=False)

        print(f"💾 已生成4个CSV数据文件到 {output_dir}")


def main():
    """主函数"""
    print("🔍 Bias统计报告生成器")
    print("=" * 50)
    print("注意：本程序需要先运行 figure4_day.py 生成缓存数据")
    print("=" * 50)

    analyzer = BiasAnalysisReport()
    report = analyzer.generate_report()


if __name__ == "__main__":
    main()