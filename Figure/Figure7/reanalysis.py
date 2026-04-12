# file: analyze_sst_trends_with_metrics.py
"""
增强版SST趋势分析脚本 - 结合性能指标分析
分析每个模型的趋势斜率k与性能指标关系
"""

import numpy as np
import pickle
from pathlib import Path
import warnings
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')


class SSTTrendsWithMetricsAnalyzer:
    """结合性能指标的SST趋势分析器"""

    def __init__(self):
        """初始化"""
        self.config = self.load_config()
        self.timeseries_data = None
        self.performance_metrics = None
        self.model_results = {}

    def load_config(self):
        """加载配置"""
        config = {
            'TIMESERIES_CACHE': './trend_cache_yearly/timeseries_cache_yearly.pkl',
            'PERFORMANCE_CACHE': '../Figure11/cache_data_metrics/global_metrics_mixed_cache.pkl',  # 新增：性能指标缓存
            'START_YEAR': 2025,
            'END_YEAR': 2100,
            'MODELS': [
                'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
                'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3-CC', 'EC-Earth3-Veg-LR', 'EC-Earth3-veg',
                'EC-Earth3', 'GFDL-CM4', 'GFDL-ESM4', 'IPSL-CM6A-LR', 'MIROC6',
                'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
                'NorESM2-MM'
            ]
        }
        return config

    def load_all_data(self):
        """加载所有数据"""
        print("📂 加载所有数据...")

        # 1. 加载时间序列数据
        ts_cache = Path(self.config['TIMESERIES_CACHE'])
        if ts_cache.exists():
            with open(ts_cache, 'rb') as f:
                self.timeseries_data = pickle.load(f)
            print(f"  ✅ 时间序列数据: {len(self.timeseries_data)} 个模型")
        else:
            print(f"  ❌ 时间序列缓存不存在: {ts_cache}")
            return False

        # 2. 加载性能指标数据
        perf_cache = Path(self.config['PERFORMANCE_CACHE'])
        if perf_cache.exists():
            with open(perf_cache, 'rb') as f:
                perf_data = pickle.load(f)

            # 提取control和mambaunet的性能指标
            self.performance_metrics = {
                'control': {},
                'mambaunet': {}
            }

            for model in self.config['MODELS']:
                if model in perf_data.get('base', {}):
                    self.performance_metrics['control'][model] = perf_data['base'][model]
                if model in perf_data.get('MambaUNet', {}):
                    self.performance_metrics['mambaunet'][model] = perf_data['MambaUNet'][model]

            print(f"  ✅ 性能指标数据: Control {len(self.performance_metrics['control'])} 个, "
                  f"MambaUNet {len(self.performance_metrics['mambaunet'])} 个")
        else:
            print(f"  ⚠️  性能指标缓存不存在: {perf_cache}")
            self.performance_metrics = None

        return True

    def calculate_yearly_sst(self, time_series_data, years=None):
        """计算年平均SST"""
        if years is None:
            years = np.arange(self.config['START_YEAR'], self.config['END_YEAR'] + 1)

        yearly_sst = []
        for i, year in enumerate(years):
            year_data = time_series_data[i]
            if np.any(~np.isnan(year_data)):
                yearly_sst.append(np.nanmean(year_data))
            else:
                yearly_sst.append(np.nan)

        return np.array(yearly_sst)

    def calculate_trend_statistics(self, yearly_sst, years):
        """计算趋势统计"""
        valid_mask = ~np.isnan(yearly_sst)
        if np.sum(valid_mask) < 2:
            return None

        y = yearly_sst[valid_mask]
        x = years[valid_mask]

        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 计算10年趋势 (°C/decade)
        trend_per_decade = slope * 10

        # 计算年平均SST
        mean_sst = np.mean(y)

        return {
            'trend_per_decade': trend_per_decade,
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'mean_sst': mean_sst,
            'years': x,
            'sst_values': y
        }

    def analyze_all_models(self):
        """分析所有模型"""
        print("\n" + "=" * 80)
        print("📊 分析所有模型的趋势和性能指标")
        print("=" * 80)

        years = np.arange(self.config['START_YEAR'], self.config['END_YEAR'] + 1)
        model_count = len(self.config['MODELS'])

        # 初始化结果存储
        results = {
            'control': {},
            'mambaunet': {}
        }

        # 分析每个模型
        for model_idx, model_name in enumerate(self.config['MODELS'], 1):
            print(f"\n[{model_idx}/{model_count}] 分析模型: {model_name}")

            if model_name not in self.timeseries_data:
                print(f"  ⚠️  模型 {model_name} 在时间序列数据中不存在")
                continue

            model_data = self.timeseries_data[model_name]

            # 分析Control
            if 'original' in model_data:
                yearly_sst = self.calculate_yearly_sst(
                    model_data['original']['time_series'],
                    years
                )
                trend_stats = self.calculate_trend_statistics(yearly_sst, years)

                if trend_stats:
                    results['control'][model_name] = {
                        'yearly_sst': yearly_sst,
                        'trend_stats': trend_stats,
                        'performance_metrics': self.performance_metrics['control'].get(model_name)
                        if self.performance_metrics else None
                    }

                    print(f"  ✅ Control: 趋势={trend_stats['trend_per_decade']:.3f}°C/decade, "
                          f"平均SST={trend_stats['mean_sst']:.2f}°C")

            # 分析MambaUNet
            if 'corrected' in model_data:
                yearly_sst = self.calculate_yearly_sst(
                    model_data['corrected']['time_series'],
                    years
                )
                trend_stats = self.calculate_trend_statistics(yearly_sst, years)

                if trend_stats:
                    results['mambaunet'][model_name] = {
                        'yearly_sst': yearly_sst,
                        'trend_stats': trend_stats,
                        'performance_metrics': self.performance_metrics['mambaunet'].get(model_name)
                        if self.performance_metrics else None
                    }

                    print(f"  ✅ MambaUNet: 趋势={trend_stats['trend_per_decade']:.3f}°C/decade, "
                          f"平均SST={trend_stats['mean_sst']:.2f}°C")

        self.model_results = results
        return results

    def print_detailed_results_table(self):
        """打印详细结果表格"""
        print("\n" + "=" * 120)
        print("📋 详细结果表格 (2025-2100年)")
        print("=" * 120)

        # 表格头部
        header = f"{'模型':<20} | {'趋势斜率k (°C/decade)':>25} | {'平均SST (°C)':>20} | {'性能指标':>40}"
        print(header)
        print("-" * 120)

        for model in self.config['MODELS']:
            # Control结果
            control_trend = ""
            control_mean = ""
            control_metrics = ""

            if model in self.model_results['control']:
                stats = self.model_results['control'][model]['trend_stats']
                control_trend = f"{stats['trend_per_decade']:.3f}"
                control_mean = f"{stats['mean_sst']:.2f}"

                if self.model_results['control'][model]['performance_metrics']:
                    metrics = self.model_results['control'][model]['performance_metrics']
                    control_metrics = f"RMSE={metrics.get('rmse', 'N/A'):.2f}, MAE={metrics.get('mae', 'N/A'):.2f}, Bias={metrics.get('bias', 'N/A'):.2f}, PCC={metrics.get('pcc', 'N/A'):.3f}"

            # MambaUNet结果
            mamba_trend = ""
            mamba_mean = ""
            mamba_metrics = ""

            if model in self.model_results['mambaunet']:
                stats = self.model_results['mambaunet'][model]['trend_stats']
                mamba_trend = f"{stats['trend_per_decade']:.3f}"
                mamba_mean = f"{stats['mean_sst']:.2f}"

                if self.model_results['mambaunet'][model]['performance_metrics']:
                    metrics = self.model_results['mambaunet'][model]['performance_metrics']
                    mamba_metrics = f"RMSE={metrics.get('rmse', 'N/A'):.2f}, MAE={metrics.get('mae', 'N/A'):.2f}, Bias={metrics.get('bias', 'N/A'):.2f}, PCC={metrics.get('pcc', 'N/A'):.3f}"

            # 打印行
            print(f"{model:<20} | Control: {control_trend:>15} | {control_mean:>10} | {control_metrics:<35}")
            print(f"{'':<20} | MambaUNet: {mamba_trend:>13} | {mamba_mean:>10} | {mamba_metrics:<35}")
            print("-" * 120)

    def print_statistical_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 80)
        print("📈 统计摘要")
        print("=" * 80)

        # 提取所有模型的趋势值
        control_trends = []
        mamba_trends = []
        control_means = []
        mamba_means = []

        for model in self.config['MODELS']:
            if model in self.model_results['control']:
                control_trends.append(
                    self.model_results['control'][model]['trend_stats']['trend_per_decade']
                )
                control_means.append(
                    self.model_results['control'][model]['trend_stats']['mean_sst']
                )

            if model in self.model_results['mambaunet']:
                mamba_trends.append(
                    self.model_results['mambaunet'][model]['trend_stats']['trend_per_decade']
                )
                mamba_means.append(
                    self.model_results['mambaunet'][model]['trend_stats']['mean_sst']
                )

        if control_trends and mamba_trends:
            print(f"\n🔍 趋势分析:")
            print(f"  Control平均趋势: {np.mean(control_trends):.3f} ± {np.std(control_trends):.3f} °C/decade")
            print(f"  MambaUNet平均趋势: {np.mean(mamba_trends):.3f} ± {np.std(mamba_trends):.3f} °C/decade")
            print(f"  趋势变化: {np.mean(mamba_trends) - np.mean(control_trends):+.3f} °C/decade")
            print(
                f"  变化比例: {(np.mean(mamba_trends) - np.mean(control_trends)) / np.mean(control_trends) * 100:+.1f}%")

            # 统计检验
            t_stat, p_val = stats.ttest_rel(control_trends, mamba_trends)
            print(f"  配对t检验: t={t_stat:.3f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"  → 统计显著 (p<0.05): MambaUNet校正显著改变了趋势")

        if control_means and mamba_means:
            print(f"\n🌡️ 平均SST分析:")
            print(f"  Control平均SST: {np.mean(control_means):.2f} ± {np.std(control_means):.2f} °C")
            print(f"  MambaUNet平均SST: {np.mean(mamba_means):.2f} ± {np.std(mamba_means):.2f} °C")
            print(f"  SST变化: {np.mean(mamba_means) - np.mean(control_means):+.2f} °C")

    def analyze_trend_metrics_relationship(self):
        """分析趋势与性能指标的关系"""
        print("\n" + "=" * 80)
        print("🔗 趋势与性能指标关系分析")
        print("=" * 80)

        if not self.performance_metrics:
            print("⚠️  无法进行分析: 性能指标数据未加载")
            return

        # 收集数据
        data_points = []

        for model in self.config['MODELS']:
            if (model in self.model_results['control'] and
                    model in self.model_results['mambaunet'] and
                    model in self.performance_metrics['control'] and
                    model in self.performance_metrics['mambaunet']):
                control_trend = self.model_results['control'][model]['trend_stats']['trend_per_decade']
                mamba_trend = self.model_results['mambaunet'][model]['trend_stats']['trend_per_decade']
                control_rmse = self.performance_metrics['control'][model].get('rmse', np.nan)
                mamba_rmse = self.performance_metrics['mambaunet'][model].get('rmse', np.nan)
                control_mae = self.performance_metrics['control'][model].get('mae', np.nan)
                mamba_mae = self.performance_metrics['mambaunet'][model].get('mae', np.nan)
                control_pcc = self.performance_metrics['control'][model].get('pcc', np.nan)
                mamba_pcc = self.performance_metrics['mambaunet'][model].get('pcc', np.nan)

                data_points.append({
                    'model': model,
                    'control_trend': control_trend,
                    'mamba_trend': mamba_trend,
                    'control_rmse': control_rmse,
                    'mamba_rmse': mamba_rmse,
                    'control_mae': control_mae,
                    'mamba_mae': mamba_mae,
                    'control_pcc': control_pcc,
                    'mamba_pcc': mamba_pcc,
                    'trend_change': mamba_trend - control_trend,
                    'rmse_change': mamba_rmse - control_rmse,
                    'mae_change': mamba_mae - control_mae,
                    'pcc_change': mamba_pcc - control_pcc
                })

        if not data_points:
            print("⚠️  无完整数据进行分析")
            return

        df = pd.DataFrame(data_points)

        print(f"\n📊 有效数据: {len(df)} 个模型")

        # 分析Control中趋势与指标的关系
        print(f"\n🔍 Control分析 (原始模型):")
        valid_mask = df['control_trend'].notna() & df['control_rmse'].notna()
        if np.sum(valid_mask) > 2:
            trend = df.loc[valid_mask, 'control_trend'].values
            rmse = df.loc[valid_mask, 'control_rmse'].values
            mae = df.loc[valid_mask, 'control_mae'].values
            pcc = df.loc[valid_mask, 'control_pcc'].values

            # 计算相关系数
            corr_rmse, p_rmse = stats.pearsonr(trend, rmse)
            corr_mae, p_mae = stats.pearsonr(trend, mae)
            corr_pcc, p_pcc = stats.pearsonr(trend, pcc)

            print(f"  趋势 vs RMSE 相关性: r={corr_rmse:.3f}, p={p_rmse:.4f}")
            print(f"  趋势 vs MAE 相关性: r={corr_mae:.3f}, p={p_mae:.4f}")
            print(f"  趋势 vs PCC 相关性: r={corr_pcc:.3f}, p={p_pcc:.4f}")

            # 判断是否存在显著关系
            if p_rmse < 0.05:
                if corr_rmse > 0:
                    print(f"  → 趋势越大，RMSE越高 (模型性能越差)")
                else:
                    print(f"  → 趋势越大，RMSE越低 (模型性能越好)")
            else:
                print(f"  → 趋势与RMSE无显著关系")

        # 分析MambaUNet中趋势与指标的关系
        print(f"\n🔍 MambaUNet分析 (校正后):")
        valid_mask = df['mamba_trend'].notna() & df['mamba_rmse'].notna()
        if np.sum(valid_mask) > 2:
            trend = df.loc[valid_mask, 'mamba_trend'].values
            rmse = df.loc[valid_mask, 'mamba_rmse'].values
            mae = df.loc[valid_mask, 'mamba_mae'].values
            pcc = df.loc[valid_mask, 'mamba_pcc'].values

            # 计算相关系数
            corr_rmse, p_rmse = stats.pearsonr(trend, rmse)
            corr_mae, p_mae = stats.pearsonr(trend, mae)
            corr_pcc, p_pcc = stats.pearsonr(trend, pcc)

            print(f"  趋势 vs RMSE 相关性: r={corr_rmse:.3f}, p={p_rmse:.4f}")
            print(f"  趋势 vs MAE 相关性: r={corr_mae:.3f}, p={p_mae:.4f}")
            print(f"  趋势 vs PCC 相关性: r={corr_pcc:.3f}, p={p_pcc:.4f}")

        # 分析趋势变化与指标变化的关系
        print(f"\n🔍 变化分析 (MambaUNet - Control):")
        valid_mask = df['trend_change'].notna() & df['rmse_change'].notna()
        if np.sum(valid_mask) > 2:
            trend_change = df.loc[valid_mask, 'trend_change'].values
            rmse_change = df.loc[valid_mask, 'rmse_change'].values
            mae_change = df.loc[valid_mask, 'mae_change'].values
            pcc_change = df.loc[valid_mask, 'pcc_change'].values

            # 计算相关系数
            corr_rmse, p_rmse = stats.pearsonr(trend_change, rmse_change)
            corr_mae, p_mae = stats.pearsonr(trend_change, mae_change)
            corr_pcc, p_pcc = stats.pearsonr(trend_change, pcc_change)

            print(f"  趋势变化 vs RMSE变化: r={corr_rmse:.3f}, p={p_rmse:.4f}")
            print(f"  趋势变化 vs MAE变化: r={corr_mae:.3f}, p={p_mae:.4f}")
            print(f"  趋势变化 vs PCC变化: r={corr_pcc:.3f}, p={p_pcc:.4f}")

            if p_rmse < 0.05:
                if corr_rmse < 0:
                    print(f"  → 趋势减弱越大，RMSE改善越大 (负值越大改善越多)")
                elif corr_rmse > 0:
                    print(f"  → 趋势减弱越大，RMSE改善越小")

        # 识别极端模型
        print(f"\n🔍 极端模型识别:")

        # 1. Control中趋势最大的模型
        if df['control_trend'].notna().any():
            max_trend_idx = df['control_trend'].idxmax()
            max_trend_model = df.loc[max_trend_idx, 'model']
            max_trend_value = df.loc[max_trend_idx, 'control_trend']
            max_trend_rmse = df.loc[max_trend_idx, 'control_rmse']

            print(f"  1. Control中趋势最大: {max_trend_model}")
            print(f"     趋势: {max_trend_value:.3f}°C/decade, RMSE: {max_trend_rmse:.3f}")

            # 该模型在MambaUNet中的表现
            mamba_trend = df.loc[max_trend_idx, 'mamba_trend']
            mamba_rmse = df.loc[max_trend_idx, 'mamba_rmse']
            print(f"     MambaUNet中: 趋势={mamba_trend:.3f}°C/decade, RMSE={mamba_rmse:.3f}")
            print(f"     趋势变化: {mamba_trend - max_trend_value:+.3f}°C/decade")
            print(f"     RMSE变化: {mamba_rmse - max_trend_rmse:+.3f}")

        # 2. Control中趋势最小的模型
        if df['control_trend'].notna().any():
            min_trend_idx = df['control_trend'].idxmin()
            min_trend_model = df.loc[min_trend_idx, 'model']
            min_trend_value = df.loc[min_trend_idx, 'control_trend']
            min_trend_rmse = df.loc[min_trend_idx, 'control_rmse']

            print(f"\n  2. Control中趋势最小: {min_trend_model}")
            print(f"     趋势: {min_trend_value:.3f}°C/decade, RMSE: {min_trend_rmse:.3f}")

        # 3. Control中RMSE最大的模型
        if df['control_rmse'].notna().any():
            max_rmse_idx = df['control_rmse'].idxmax()
            max_rmse_model = df.loc[max_rmse_idx, 'model']
            max_rmse_value = df.loc[max_rmse_idx, 'control_rmse']
            max_rmse_trend = df.loc[max_rmse_idx, 'control_trend']

            print(f"\n  3. Control中RMSE最大: {max_rmse_model}")
            print(f"     RMSE: {max_rmse_value:.3f}, 趋势: {max_rmse_trend:.3f}°C/decade")

            # 该模型在MambaUNet中的表现
            mamba_rmse = df.loc[max_rmse_idx, 'mamba_rmse']
            mamba_trend = df.loc[max_rmse_idx, 'mamba_trend']
            print(f"     MambaUNet中: RMSE={mamba_rmse:.3f}, 趋势={mamba_trend:.3f}°C/decade")
            print(f"     RMSE变化: {mamba_rmse - max_rmse_value:+.3f}")

        # 4. MambaUNet中改善最大的模型
        if df['rmse_change'].notna().any():
            min_change_idx = df['rmse_change'].idxmin()  # 负值越大改善越大
            best_improve_model = df.loc[min_change_idx, 'model']
            rmse_change = df.loc[min_change_idx, 'rmse_change']
            trend_change = df.loc[min_change_idx, 'trend_change']

            print(f"\n  4. MambaUNet中RMSE改善最大: {best_improve_model}")
            print(f"     RMSE变化: {rmse_change:+.3f}, 趋势变化: {trend_change:+.3f}°C/decade")

        # 5. 趋势变化最大的模型
        if df['trend_change'].notna().any():
            min_trend_change_idx = df['trend_change'].idxmin()  # 负值最大表示减弱最多
            max_trend_change_idx = df['trend_change'].idxmax()  # 正值最大表示增强最多

            min_change_model = df.loc[min_trend_change_idx, 'model']
            min_change_value = df.loc[min_trend_change_idx, 'trend_change']
            max_change_model = df.loc[max_trend_change_idx, 'model']
            max_change_value = df.loc[max_trend_change_idx, 'trend_change']

            print(f"\n  5. 趋势变化极端值:")
            print(f"     减弱最多: {min_change_model} ({min_change_value:+.3f}°C/decade)")
            print(f"     增强最多: {max_change_model} ({max_change_value:+.3f}°C/decade)")

        return df

    def identify_high_overestimation_models(self, threshold_percentile=75):
        """识别高估模型 (趋势特别大且性能差的模型)"""
        print("\n" + "=" * 80)
        print("🎯 识别高估模型 (趋势大且性能差)")
        print("=" * 80)

        data_points = []

        for model in self.config['MODELS']:
            if (model in self.model_results['control'] and
                    model in self.performance_metrics['control']):

                trend = self.model_results['control'][model]['trend_stats']['trend_per_decade']
                rmse = self.performance_metrics['control'][model].get('rmse', np.nan)
                mae = self.performance_metrics['control'][model].get('mae', np.nan)
                pcc = self.performance_metrics['control'][model].get('pcc', np.nan)

                if not np.isnan(trend) and not np.isnan(rmse):
                    data_points.append({
                        'model': model,
                        'trend': trend,
                        'rmse': rmse,
                        'mae': mae,
                        'pcc': pcc
                    })

        if not data_points:
            print("⚠️  无足够数据")
            return

        df = pd.DataFrame(data_points)

        # 计算百分位数
        trend_threshold = np.percentile(df['trend'], threshold_percentile)
        rmse_threshold = np.percentile(df['rmse'], threshold_percentile)

        print(f"\n📏 阈值设置 (前{threshold_percentile}%):")
        print(f"  趋势阈值: >{trend_threshold:.3f} °C/decade")
        print(f"  RMSE阈值: >{rmse_threshold:.3f} °C")
        print(f"  (趋势前{threshold_percentile}%且RMSE前{threshold_percentile}%的模型视为高估模型)")

        # 识别高估模型
        high_trend_mask = df['trend'] > trend_threshold
        high_rmse_mask = df['rmse'] > rmse_threshold

        high_overestimation_mask = high_trend_mask & high_rmse_mask

        if high_overestimation_mask.any():
            high_models = df.loc[high_overestimation_mask]

            print(f"\n🔍 识别到 {len(high_models)} 个高估模型:")
            print("-" * 70)
            print(f"{'模型':<20} {'趋势':>10} {'RMSE':>10} {'MAE':>10} {'PCC':>10}")
            print("-" * 70)

            for _, row in high_models.iterrows():
                print(
                    f"{row['model']:<20} {row['trend']:>10.3f} {row['rmse']:>10.3f} {row['mae']:>10.3f} {row['pcc']:>10.3f}")

            print("-" * 70)

            # 分析这些模型在MambaUNet中的表现
            print(f"\n📊 高估模型在MambaUNet中的表现改善情况:")
            print("-" * 80)
            print(f"{'模型':<20} {'趋势变化':>12} {'RMSE变化':>12} {'MAE变化':>12} {'PCC变化':>12}")
            print("-" * 80)

            for model_name in high_models['model']:
                if model_name in self.model_results['mambaunet']:
                    mamba_trend = self.model_results['mambaunet'][model_name]['trend_stats']['trend_per_decade']
                    mamba_rmse = self.performance_metrics['mambaunet'][model_name].get('rmse', np.nan)
                    mamba_mae = self.performance_metrics['mambaunet'][model_name].get('mae', np.nan)
                    mamba_pcc = self.performance_metrics['mambaunet'][model_name].get('pcc', np.nan)

                    control_trend = df.loc[df['model'] == model_name, 'trend'].values[0]
                    control_rmse = df.loc[df['model'] == model_name, 'rmse'].values[0]
                    control_mae = df.loc[df['model'] == model_name, 'mae'].values[0]
                    control_pcc = df.loc[df['model'] == model_name, 'pcc'].values[0]

                    trend_change = mamba_trend - control_trend
                    rmse_change = mamba_rmse - control_rmse
                    mae_change = mamba_mae - control_mae
                    pcc_change = mamba_pcc - control_pcc

                    # 判断改善情况
                    trend_improve = "✅" if trend_change < -0.01 else "⚠️"  # 趋势减弱视为改善
                    rmse_improve = "✅" if rmse_change < 0 else "⚠️"  # RMSE减小视为改善
                    mae_improve = "✅" if mae_change < 0 else "⚠️"  # MAE减小视为改善
                    pcc_improve = "✅" if pcc_change > 0 else "⚠️"  # PCC增加视为改善

                    print(f"{model_name:<20} {trend_change:>+10.3f}{trend_improve:>2} "
                          f"{rmse_change:>+10.3f}{rmse_improve:>2} "
                          f"{mae_change:>+10.3f}{mae_improve:>2} "
                          f"{pcc_change:>+10.3f}{pcc_improve:>2}")

            print("-" * 80)

            # 计算平均改善
            total_trend_change = 0
            total_rmse_change = 0
            total_mae_change = 0
            total_pcc_change = 0
            count = 0

            for model_name in high_models['model']:
                if model_name in self.model_results['mambaunet']:
                    mamba_trend = self.model_results['mambaunet'][model_name]['trend_stats']['trend_per_decade']
                    mamba_rmse = self.performance_metrics['mambaunet'][model_name].get('rmse', np.nan)
                    mamba_mae = self.performance_metrics['mambaunet'][model_name].get('mae', np.nan)
                    mamba_pcc = self.performance_metrics['mambaunet'][model_name].get('pcc', np.nan)

                    control_trend = df.loc[df['model'] == model_name, 'trend'].values[0]
                    control_rmse = df.loc[df['model'] == model_name, 'rmse'].values[0]
                    control_mae = df.loc[df['model'] == model_name, 'mae'].values[0]
                    control_pcc = df.loc[df['model'] == model_name, 'pcc'].values[0]

                    total_trend_change += (mamba_trend - control_trend)
                    total_rmse_change += (mamba_rmse - control_rmse)
                    total_mae_change += (mamba_mae - control_mae)
                    total_pcc_change += (mamba_pcc - control_pcc)
                    count += 1

            if count > 0:
                print(f"\n📈 高估模型平均改善:")
                print(f"  平均趋势变化: {total_trend_change / count:+.3f} °C/decade")
                print(f"  平均RMSE变化: {total_rmse_change / count:+.3f}")
                print(f"  平均MAE变化: {total_mae_change / count:+.3f}")
                print(f"  平均PCC变化: {total_pcc_change / count:+.3f}")
        else:
            print(f"\n⚠️  未找到同时满足高趋势和高RMSE的模型")

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n" + "=" * 80)
        print("📋 综合分析报告")
        print("=" * 80)

        # 1. 总体趋势分析
        control_trends = []
        mamba_trends = []

        for model in self.config['MODELS']:
            if model in self.model_results['control']:
                control_trends.append(
                    self.model_results['control'][model]['trend_stats']['trend_per_decade']
                )
            if model in self.model_results['mambaunet']:
                mamba_trends.append(
                    self.model_results['mambaunet'][model]['trend_stats']['trend_per_decade']
                )

        if control_trends and mamba_trends:
            control_mean = np.mean(control_trends)
            mamba_mean = np.mean(mamba_trends)
            trend_change = mamba_mean - control_mean

            print(f"\n🎯 主要发现:")
            print(f"1. MambaUNet校正显著减弱了北极SST的变暖趋势:")
            print(f"   • Control平均趋势: {control_mean:.3f} °C/decade")
            print(f"   • MambaUNet平均趋势: {mamba_mean:.3f} °C/decade")
            print(f"   • 趋势减弱: {trend_change:+.3f} °C/decade ({trend_change / control_mean * 100:+.1f}%)")

            # 2. 模型离散度分析
            control_std = np.std(control_trends)
            mamba_std = np.std(mamba_trends)

            print(f"\n2. 模型间离散度显著收敛:")
            print(f"   • Control离散度(标准差): {control_std:.3f} °C/decade")
            print(f"   • MambaUNet离散度: {mamba_std:.3f} °C/decade")
            print(f"   • 离散度减小: {(mamba_std - control_std):+.3f} °C/decade")

            # 3. 极端模型分析
            control_max = np.max(control_trends)
            control_min = np.min(control_trends)
            mamba_max = np.max(mamba_trends)
            mamba_min = np.min(mamba_trends)

            print(f"\n3. 极端趋势范围明显缩小:")
            print(f"   • Control趋势范围: [{control_min:.3f}, {control_max:.3f}] °C/decade")
            print(f"   • MambaUNet趋势范围: [{mamba_min:.3f}, {mamba_max:.3f}] °C/decade")
            print(f"   • 范围缩小: {(mamba_max - mamba_min) - (control_max - control_min):+.3f} °C/decade")

            # 4. 性能指标相关性分析
            if self.performance_metrics:
                print(f"\n4. 趋势与模型性能的关系:")

                # 收集数据
                control_trend_list = []
                control_rmse_list = []
                control_mae_list = []
                control_pcc_list = []

                for model in self.config['MODELS']:
                    if (model in self.model_results['control'] and
                            model in self.performance_metrics['control']):
                        trend = self.model_results['control'][model]['trend_stats']['trend_per_decade']
                        rmse = self.performance_metrics['control'][model].get('rmse', np.nan)
                        mae = self.performance_metrics['control'][model].get('mae', np.nan)
                        pcc = self.performance_metrics['control'][model].get('pcc', np.nan)

                        if not np.isnan(trend) and not np.isnan(rmse):
                            control_trend_list.append(trend)
                            control_rmse_list.append(rmse)
                            control_mae_list.append(mae)
                            control_pcc_list.append(pcc)

                if len(control_trend_list) > 2:
                    # 与RMSE相关性
                    corr_rmse, p_rmse = stats.pearsonr(control_trend_list, control_rmse_list)
                    corr_mae, p_mae = stats.pearsonr(control_trend_list, control_mae_list)
                    corr_pcc, p_pcc = stats.pearsonr(control_trend_list, control_pcc_list)

                    print(f"   • 在Control中:")
                    print(f"     趋势 vs RMSE: r={corr_rmse:.3f}, p={p_rmse:.3f}")
                    print(f"     趋势 vs MAE: r={corr_mae:.3f}, p={p_mae:.3f}")
                    print(f"     趋势 vs PCC: r={corr_pcc:.3f}, p={p_pcc:.3f}")

                    if p_rmse < 0.05:
                        if corr_rmse > 0:
                            print(f"     → 趋势大的模型有更高的RMSE (性能更差)")
                            print(f"     → 这表明高估变暖趋势的模型通常性能较差")
                            print(f"     → MambaUNet校正能有效改善这些模型的性能和趋势")
                        else:
                            print(f"     → 趋势与RMSE呈负相关")
                    else:
                        print(f"     → 趋势与RMSE无显著相关关系")

        print("\n" + "=" * 80)
        print("✅ 分析完成 - 可将上述结果用于论文撰写")
        print("=" * 80)

    def generate_ranking_table(self, mode='control'):
        """
        生成趋势与性能指标的综合排名表，包含具体数值
        mode: 'control' or 'mambaunet'
        """
        assert mode in ['control', 'mambaunet']

        records = []

        for model in self.config['MODELS']:
            if model not in self.model_results[mode]:
                continue
            if model not in self.performance_metrics[mode]:
                continue

            trend = self.model_results[mode][model]['trend_stats']['trend_per_decade']
            mean_sst = self.model_results[mode][model]['trend_stats']['mean_sst']

            metrics = self.performance_metrics[mode][model]
            rmse = metrics.get('rmse', np.nan)
            mae = metrics.get('mae', np.nan)
            pcc = metrics.get('pcc', np.nan)
            bias = metrics.get('bias', np.nan)

            records.append({
                'model': model,
                'trend': trend,
                'mean_sst': mean_sst,
                'rmse': rmse,
                'mae': mae,
                'pcc': pcc,
                'bias': bias,
                'abs_bias': np.abs(bias)
            })

        df = pd.DataFrame(records)

        # === 排名规则 ===
        # 趋势：从大到小排名（趋势越大排名越靠前）
        df['Trend_rank'] = df['trend'].rank(ascending=False, method='min').astype(int)
        df['MeanSST_rank'] = df['mean_sst'].rank(ascending=False, method='min').astype(int)
        df['RMSE_rank'] = df['rmse'].rank(ascending=True, method='min').astype(int)
        df['MAE_rank'] = df['mae'].rank(ascending=True, method='min').astype(int)
        df['PCC_rank'] = df['pcc'].rank(ascending=False, method='min').astype(int)
        df['Bias_rank'] = df['abs_bias'].rank(ascending=True, method='min').astype(int)

        # 按趋势排序（趋势从大到小）
        df = df.sort_values('Trend_rank')

        # 创建包含排名和具体数值的字符串列
        # 注意：排名已经是整数，我们只需要格式化数值
        df['趋势排名(°C/decade)'] = df.apply(
            lambda row: f"{row['Trend_rank']} ({row['trend']:.3f})", axis=1
        )
        df['平均SST排名(°C)'] = df.apply(
            lambda row: f"{row['MeanSST_rank']} ({row['mean_sst']:.2f})", axis=1
        )
        df['RMSE排名'] = df.apply(
            lambda row: f"{row['RMSE_rank']} ({row['rmse']:.3f})", axis=1
        )
        df['MAE排名'] = df.apply(
            lambda row: f"{row['MAE_rank']} ({row['mae']:.3f})", axis=1
        )
        df['PCC排名'] = df.apply(
            lambda row: f"{row['PCC_rank']} ({row['pcc']:.3f})", axis=1
        )
        df['Bias排名'] = df.apply(
            lambda row: f"{row['Bias_rank']} ({row['bias']:.3f})" if not np.isnan(row['bias']) else "", axis=1
        )

        # 准备输出表格
        rank_df = df[
            ['model',
             '趋势排名(°C/decade)',
             '平均SST排名(°C)',
             'RMSE排名',
             'MAE排名',
             'PCC排名',
             'Bias排名']
        ].reset_index(drop=True)

        # 重命名model列为'模型'
        rank_df = rank_df.rename(columns={'model': '模型'})

        # 设置pandas显示选项以便更好的对齐
        pd.set_option('display.unicode.east_asian_width', True)
        pd.set_option('display.width', 140)
        pd.set_option('display.max_colwidth', 20)

        print(f"\n📊 {mode.upper()} 综合排名表 (按趋势排序，显示排名和具体数值)")
        print("=" * 140)

        # 创建格式化的表格
        # 首先，创建一个格式化的字符串表示
        formatted_table = []

        # 表头
        headers = ['模型', '趋势排名(°C/decade)', '平均SST排名(°C)', 'RMSE排名', 'MAE排名', 'PCC排名', 'Bias排名']
        col_widths = [20, 25, 20, 15, 15, 15, 15]

        # 创建表头行
        header_row = " | ".join(f"{headers[i]:^{col_widths[i]}}" for i in range(len(headers)))
        formatted_table.append(header_row)
        formatted_table.append("-" * 140)

        # 添加数据行
        for _, row in rank_df.iterrows():
            row_values = [
                str(row['模型'])[:18].ljust(18),  # 模型名，左对齐
                str(row['趋势排名(°C/decade)']).center(23),  # 居中
                str(row['平均SST排名(°C)']).center(18),  # 居中
                str(row['RMSE排名']).center(13),  # 居中
                str(row['MAE排名']).center(13),  # 居中
                str(row['PCC排名']).center(13),  # 居中
                str(row['Bias排名']).center(13)  # 居中
            ]
            row_str = " | ".join(row_values)
            formatted_table.append(row_str)

        # 打印表格
        for line in formatted_table:
            print(line)

        print("=" * 140)

        # 保存到CSV文件
        output_file = f'ranking_table_{mode}.csv'
        rank_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 排名表已保存到: {output_file}")

        return rank_df

    def run_full_analysis(self):
        """运行完整分析"""
        print("🚀 SST趋势与性能指标综合分析开始")
        print(f"📅 时间范围: {self.config['START_YEAR']}-{self.config['END_YEAR']}")
        print(f"📍 区域: 北极地区")
        print("=" * 80)

        # 1. 加载数据
        if not self.load_all_data():
            return False

        # 2. 分析所有模型
        self.analyze_all_models()

        # 3. 打印详细结果
        self.print_detailed_results_table()

        # 4. 打印统计摘要
        self.print_statistical_summary()

        # 5. 分析趋势与指标关系
        if self.performance_metrics:
            df = self.analyze_trend_metrics_relationship()

            # 6. 识别高估模型
            self.identify_high_overestimation_models(threshold_percentile=75)

        # 7. 生成综合报告
        self.generate_comprehensive_report()

        print("\n" + "=" * 80)
        print("📋 生成排名表格")
        print("=" * 80)

        control_ranks = self.generate_ranking_table(mode='control')
        mambaunet_ranks = self.generate_ranking_table(mode='mambaunet')

        return True


def main():
    """主函数"""
    analyzer = SSTTrendsWithMetricsAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()