import os
import sys
import csv
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 添加项目路径
base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(base_dir)

# 导入必要的模块
from utils.loader import get_estimator_by_name
from utils.utils import display_experiment_configuration
from utils.parse import get_df, preprocess, save_result

# 参数设置
param_settings = {
    'estimator': None,
    'regularization': True,
    'sample_splits': False,
    'reg_lambda': 1e-2,
    'reg_lambda_tilde': 1e-2,
    'kernel': 'gauss',
    'density': 'gaussian',
    'bandwidth': 1,
    'bandwidth_mode': 'auto',
    'epsilon': 0.3,
    'normalized': True
}

data_settings = {
    'expname': 'custom_data_mediation',
    'data': 'custom',
    'n_samples': None,
    'd': None,
    'd_prime': None,
    'random_state': None
}

# 实验名称
EXPNAME = 'custom_data_mediation'


def load_custom_data(data_path, t_col, m_col, y_col, categorical_cols=[]):
    """
    加载自定义CSV数据，处理分类变量
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"文件不存在: {data_path}")

        data = pd.read_csv(data_path)
        print(f"数据加载成功: {data.shape[0]}行, {data.shape[1]}列")

        required_cols = [t_col, m_col, y_col] + categorical_cols
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少必要的列: {missing_cols}")

        if categorical_cols:
            print(f"处理分类变量: {categorical_cols}")
            for col in categorical_cols:
                if col in data.columns:
                    data[col] = data[col].astype(str)

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ],
                remainder='passthrough'
            )
            transformed_data = preprocessor.fit_transform(data)

            cat_encoder = preprocessor.named_transformers_['cat']
            cat_cols = list(cat_encoder.get_feature_names_out(categorical_cols))
            num_cols = [col for col in data.columns if col not in categorical_cols]

            new_columns = cat_cols + num_cols
            data = pd.DataFrame(transformed_data, columns=new_columns)

            t_col_new = [col for col in new_columns if col.endswith(f"_{t_col}") or col == t_col]
            m_col_new = [col for col in new_columns if col.endswith(f"_{m_col}") or col == m_col]
            y_col_new = [col for col in new_columns if col.endswith(f"_{y_col}") or col == y_col]

            if t_col_new: t_col = t_col_new[0]
            if m_col_new: m_col = m_col_new[0]
            if y_col_new: y_col = y_col_new[0]

        t = data[t_col].values.reshape(-1, 1)
        m = data[m_col].values.reshape(-1, 1)
        y = data[y_col].values.reshape(-1, 1)

        x_cols = [col for col in data.columns if col not in [t_col, m_col, y_col]]
        x = data[x_cols].values

        if not all(np.issubdtype(arr.dtype, np.number) for arr in [t, m, y, x]):
            print("警告: 检测到非数值型数据，尝试转换为数值型...")
            t = t.astype(float)
            m = m.astype(float)
            y = y.astype(float)
            x = x.astype(float)

        print("处理缺失值...")
        valid_idx = ~np.isnan(t).any(axis=1) & ~np.isnan(m).any(axis=1) & ~np.isnan(y).any(axis=1) & ~np.isnan(x).any(
            axis=1)
        n_valid = np.sum(valid_idx)
        print(f"有效样本: {n_valid}/{len(y)} ({(n_valid / len(y)) * 100:.1f}%)")

        return x[valid_idx], t[valid_idx], m[valid_idx], y[valid_idx]

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        raise


def run_single_experiment(args, d_value, d_prime_value, x, t, m, y):
    """
    执行单个处置水平的中介效应分析
    """
    param_settings = {
        'estimator': args.estimator,
        'regularization': True,
        'sample_splits': 5,
        'reg_lambda': 1e-3,
        'reg_lambda_tilde': 1e-3,
        'kernel': 'gauss',
        'density': 'gaussian',
        'bandwidth': 1,
        'bandwidth_mode': args.bandwidth_mode,
        'epsilon': args.epsilon,
        'normalized': True
    }

    data_settings = {
        'expname': EXPNAME,
        'data': 'custom',
        'n_samples': args.n_samples,
        'd': d_value,
        'd_prime': d_prime_value,
        'random_state': args.random_seed
    }

    # display_experiment_configuration(data_settings, param_settings)
    estimator = get_estimator_by_name(param_settings)(param_settings)
    estimator.fit(t, m, x, y)
    causal_results = estimator.estimate(d_value, d_prime_value, t, m, x, y)
    return causal_results


def plot_multi_d_results(results_df, d_prime):
    """
    可视化：绘制总效应、直接效应和中介效应及其置信区间
    """
    print("\n" + "=" * 50)
    print("可视化多重效应分析结果")
    print("=" * 50)

    # 创建绘图目录
    figures_dir = 'plots/multi_d_analysis/'
    os.makedirs(figures_dir, exist_ok=True)

    # 设置中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 适配Windows/Mac
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    plt.figure(figsize=(12, 8))

    # 提取数据
    d_values = results_df['d']

    # 定义绘图数据配置
    # 格式: (效应值列名, 误差列名, 标签, 颜色, 线型)
    plot_configs = [
        ('total_effect', 'margin_error_total', '总效应 (Total Effect)', '#1f77b4', '-'),  # 蓝色实线
        ('direct_effect', 'margin_error_direct', '直接效应 (Direct Effect)', '#2ca02c', '--'),  # 绿色虚线
        ('indirect_effect', 'margin_error_indirect', '中介效应 (Indirect Effect)', '#d62728', '-.')  # 红色点划线
    ]

    for effect_col, error_col, label, color, linestyle in plot_configs:
        if effect_col in results_df.columns:
            effect_values = results_df[effect_col]

            # 绘制线条
            plt.plot(d_values, effect_values, color=color, linestyle=linestyle, linewidth=2.5, label=label)

            # 处理误差区间
            margin = 0
            if error_col in results_df.columns:
                margin = results_df[error_col]
            elif 'margin_error' in results_df.columns and effect_col == 'indirect_effect':
                # 如果没有特定的误差列，间接效应回退到通用误差
                margin = results_df['margin_error']

            if isinstance(margin, (pd.Series, np.ndarray, float, int)) and np.any(margin > 0):
                lower_bound = effect_values - margin
                upper_bound = effect_values + margin
                plt.fill_between(d_values, lower_bound, upper_bound, color=color, alpha=0.15)
        else:
            print(f"警告: 数据中缺少列 {effect_col}，跳过绘制")

    # 添加辅助线 (y=0)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    # 添加标注
    plt.title(f'各效应随处置水平变化 (基准 d_prime={d_prime:.2f})', fontsize=16, pad=20)
    plt.xlabel('处置水平 (d)', fontsize=14)
    plt.ylabel('效应大小 (Effect Size)', fontsize=14)
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)

    # 保存图像
    save_path = f'{figures_dir}all_effects_comparison.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"已保存对比图: {save_path}")

    # 显示
    plt.show()


def run_multiple_d_values(args):
    print("=" * 50)
    print("开始多处置水平中介效应分析 (无抽样模式)")
    print("=" * 50)

    categorical_cols = args.categorical_cols.split(',') if args.categorical_cols else []
    categorical_cols = [col.strip() for col in categorical_cols if col.strip()]

    x, t, m, y = load_custom_data(
        data_path=args.data_path,
        t_col=args.t_col,
        m_col=args.m_col,
        y_col=args.y_col,
        categorical_cols=categorical_cols
    )

    # 设置 d_prime 固定值
    d_prime = 3.0

    # 创建 d 值范围
    d_min, d_max = 0, 3.5
    d_step = 0.25
    d_values = np.arange(d_min, d_max + d_step, d_step)
    d_values = d_values[d_values <= d_max]

    print(f"\n分析{d_values.size}个处置水平: d从{d_min:.2f}到{d_max:.2f}, 步长{d_step:.2f}")
    print(f"固定基准值 d_prime={d_prime:.2f}")

    results = []

    for d in d_values:
        try:
            # 运行实验
            print(f"正在计算 d={d:.2f} ...", end='\r')
            causal_results = run_single_experiment(args, d, d_prime, x, t, m, y)

            # 获取各效应的误差
            me_indirect = causal_results.get('margin_error_indirect', causal_results.get('margin_error', 0))
            me_direct = causal_results.get('margin_error_direct', 0)
            me_total = causal_results.get('margin_error_total', 0)

            result_row = {
                'd': d,
                'd_prime': d_prime,
                'total_effect': causal_results.get('total_effect', 0),
                'direct_effect': causal_results.get('direct_effect', 0),
                'indirect_effect': causal_results.get('indirect_effect', 0),

                'margin_error_indirect': me_indirect,
                'margin_error_direct': me_direct,  # 已修复空格问题
                'margin_error_total': me_total,  # 已修复空格问题

                'mediated_response': causal_results.get('mediated_response', 0),
                'margin_error': causal_results.get('margin_error', 0),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result_row)

        except Exception as e:
            print(f"\n处置水平 d={d:.2f} 分析失败: {str(e)}")
            continue

    print("\n计算完成。")

    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    results_file = f"results/multi_d_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"结果已保存至: {results_file}")

    # 可视化结果
    plot_multi_d_results(results_df, d_prime)

    return results_df


def get_saved_results():
    results_dir = 'results/'
    if not os.path.exists(results_dir):
        print("结果目录不存在")
        return None

    result_files = [f for f in os.listdir(results_dir) if f.startswith('multi_d_results') and f.endswith('.csv')]
    if not result_files:
        print("未找到结果文件")
        return None

    result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    latest_file = os.path.join(results_dir, result_files[0])

    print(f"加载结果文件: {latest_file}")
    results_df = pd.read_csv(latest_file)
    d_prime = results_df['d_prime'].iloc[0]
    plot_multi_d_results(results_df, d_prime)
    return results_df


def experiment(args):
    # 单次运行模式
    param_settings = {
        'estimator': args.estimator,
        'regularization': True,
        'sample_splits': 5,
        'reg_lambda': 1e-3,
        'reg_lambda_tilde': 1e-3,
        'kernel': 'gauss',
        'density': 'gaussian',
        'bandwidth': 1,
        'bandwidth_mode': args.bandwidth_mode,
        'epsilon': args.epsilon,
        'normalized': True
    }

    categorical_cols = args.categorical_cols.split(',') if args.categorical_cols else []
    categorical_cols = [col.strip() for col in categorical_cols if col.strip()]

    x, t, m, y = load_custom_data(args.data_path, args.t_col, args.m_col, args.y_col, categorical_cols)
    causal_results = run_single_experiment(args, args.d, args.d_prime, x, t, m, y)

    print("\n" + "=" * 50)
    print(f"单次分析结果 (d={args.d} vs d_prime={args.d_prime}):")
    print(f"中介效应: {causal_results.get('indirect_effect', 0):.4f}")

    save_result(param_settings, causal_results, {'d': args.d, 'd_prime': args.d_prime})
    return causal_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='中介效应分析')
    parser.add_argument('--data_path', required=True, help='CSV数据文件路径')
    parser.add_argument('--t_col', required=True, help='处置变量列名')
    parser.add_argument('--m_col', required=True, help='中介变量列名')
    parser.add_argument('--y_col', required=True, help='结果变量列名')
    parser.add_argument('--run', action='store_true', help='执行单次分析')
    parser.add_argument('--run_multi_d', action='store_true', help='执行多处置水平分析')
    parser.add_argument('--get_results', action='store_true', help='可视化保存的结果')
    parser.add_argument('--categorical_cols', type=str, default='', help='分类变量')
    parser.add_argument('--estimator', default='linear', help='估计器选择')
    parser.add_argument('--d', type=float, help='单次分析d')
    parser.add_argument('--d_prime', type=float, help='单次分析d_prime')
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--bandwidth_mode', default='amse')
    parser.add_argument('--epsilon', type=float, default=0.6)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    if not (args.run or args.run_multi_d or args.get_results):
        print("错误: 请指定 --run, --run_multi_d 或 --get_results")
        sys.exit(1)

    if args.run_multi_d:
        run_multiple_d_values(args)
    elif args.run:
        if args.d is None or args.d_prime is None:
            print("需要指定 --d 和 --d_prime")
            sys.exit(1)
        experiment(args)
    elif args.get_results:
        get_saved_results()