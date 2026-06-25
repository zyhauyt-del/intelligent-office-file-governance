# AHP-EWM + TOPSIS + 0-1整数规划 决策模块

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations


# Saaty标度随机一致性指标
_RI_TABLE: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59,
}

# 论文AHP判断矩阵（6个二级指标两两比较）
_AHP_MATRIX = np.array([
    [1,    2,    1,    2,    1,    3   ],
    [1/2,  1,    1/2,  1,    1/2,  2   ],
    [1,    2,    1,    2,    2,    3   ],
    [1/2,  1,    1/2,  1,    1/2,  1   ],
    [1,    2,    1/2,  2,    1,    2   ],
    [1/3,  1/2,  1/3,  1,    1/2,  1   ],
], dtype=np.float64)


# ---------- AHP ----------

def ahp_normalize(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 列归一化 → 行均值 = 权重向量
    n = matrix.shape[0]
    if n < 1:
        raise ValueError('判断矩阵维度不能为空')
    if np.any(matrix <= 0):
        raise ValueError('判断矩阵所有元素必须为正数')
    col_sums = matrix.sum(axis=0)
    if np.any(col_sums == 0):
        raise ValueError('判断矩阵列和不能为0')
    norm_matrix = matrix / col_sums
    weight = norm_matrix.mean(axis=1)
    return weight, norm_matrix


def ahp_consistency_check(matrix: np.ndarray) -> Dict[str, float]:
    # 一致性检验 CR < 0.1 则认为通过
    n = matrix.shape[0]
    weight, _ = ahp_normalize(matrix)
    aw = matrix @ weight
    lambda_max = float(np.mean(aw / weight))
    if n == 1:
        ci = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
    ri = _RI_TABLE.get(n, 1.59)
    cr = ci / ri if ri != 0 else 0.0
    is_consistent = cr < 0.1
    return {
        'lambda_max': round(lambda_max, 4),
        'CI': round(ci, 4),
        'RI': ri,
        'CR': round(cr, 4),
        'is_consistent': is_consistent,
    }


# ---------- EWM 熵权法 ----------

def entropy_weight_method(data: np.ndarray) -> np.ndarray:
    # 熵值越小 → 信息量越大 → 权重越高
    n_samples, n_indicators = data.shape
    if n_samples < 2:
        raise ValueError(f'样本数 ({n_samples}) 不足，EWM 至少需要 2 个样本')
    if n_indicators < 1:
        raise ValueError('指标数不能为 0')

    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    denom = data_max - data_min
    denom[denom == 0] = 1e-8
    norm_data = (data - data_min) / denom
    norm_data = np.clip(norm_data, 1e-8, 1.0)

    col_sums = norm_data.sum(axis=0)
    p_matrix = norm_data / col_sums
    k = 1.0 / np.log(n_samples)
    e_values = -k * np.sum(p_matrix * np.log(p_matrix + 1e-12), axis=0)
    d_values = 1.0 - e_values
    w_ewm = d_values / d_values.sum()
    return w_ewm


def combine_weights(
    w_ahp: np.ndarray,
    w_ewm: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    # AHP-EWM线性组合 w = α*主观 + (1-α)*客观
    if len(w_ahp) != len(w_ewm):
        raise ValueError(f'权重向量长度不一致: {len(w_ahp)} vs {len(w_ewm)}')
    if not 0 <= alpha <= 1:
        raise ValueError(f'alpha 必须在 [0, 1] 区间，当前值: {alpha}')
    combined = alpha * w_ahp + (1 - alpha) * w_ewm
    return combined / combined.sum()


# ---------- TOPSIS ----------

def topsis_evaluate(
    data: np.ndarray,
    weights: np.ndarray,
    benefit: Optional[np.ndarray] = None
) -> np.ndarray:
    # 向量归一化 → 加权 → 正负理想解距离 → 贴近度C_i
    n_samples, n_indicators = data.shape
    if len(weights) != n_indicators:
        raise ValueError(f'权重维度 ({len(weights)}) 与指标数 ({n_indicators}) 不匹配')
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError('数据包含 NaN 或 Inf，请先清洗')
    if benefit is None:
        benefit = np.ones(n_indicators, dtype=bool)

    norms = np.sqrt(np.sum(data ** 2, axis=0))
    norms[norms == 0] = 1e-8
    norm_data = data / norms
    weighted = norm_data * weights

    a_plus = np.zeros(n_indicators)
    a_minus = np.zeros(n_indicators)
    for j in range(n_indicators):
        col = weighted[:, j]
        if benefit[j]:
            a_plus[j] = col.max()
            a_minus[j] = col.min()
        else:
            a_plus[j] = col.min()
            a_minus[j] = col.max()

    d_plus = np.sqrt(np.sum((weighted - a_plus) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - a_minus) ** 2, axis=1))
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-8
    c_values = d_minus / denom
    return c_values


def classify_by_score(scores: np.ndarray,
                      low_threshold: float = 0.33,
                      high_threshold: float = 0.67) -> Dict[str, np.ndarray]:
    # 高/中/低三分级
    high_idx = np.where(scores >= high_threshold)[0]
    medium_idx = np.where((scores >= low_threshold) & (scores < high_threshold))[0]
    low_idx = np.where(scores < low_threshold)[0]
    return {'high': high_idx, 'medium': medium_idx, 'low': low_idx}


# ---------- 0-1 整数规划 ----------

def zero_one_knapsack_optimization(
    scores: np.ndarray,
    time_costs: np.ndarray,
    accuracy_rates: np.ndarray,
    total_time_budget: float,
    min_accuracy: float = 0.85,
) -> Dict:
    # 贪心求解：高分文件强制复核，其余按性价比(得分/耗时)排序选取
    n = len(scores)
    if n == 0:
        return {}
    if len(time_costs) != n or len(accuracy_rates) != n:
        raise ValueError('输入数组长度不一致')

    high_idx = np.where(scores >= 0.67)[0]
    x = np.zeros(n, dtype=np.int32)
    x[high_idx] = 1
    remaining = total_time_budget - time_costs[high_idx].sum()

    remaining_idx = np.where(x == 0)[0]
    if len(remaining_idx) > 0 and remaining > 0:
        efficiency = scores[remaining_idx] / (time_costs[remaining_idx] + 1e-6)
        sorted_order = remaining_idx[np.argsort(efficiency)[::-1]]
        for i in sorted_order:
            if time_costs[i] <= remaining:
                x[i] = 1
                remaining -= time_costs[i]
            else:
                break

    selected = np.where(x == 1)[0]
    selected_count = len(selected)
    total_time = float(time_costs[selected].sum())
    avg_accuracy = (
        float(accuracy_rates[selected].mean()) if selected_count > 0 else 0.0
    )
    total_score = float(scores[selected].sum())
    high_covered = np.intersect1d(selected, high_idx)
    is_high_covered = len(high_covered) / len(high_idx) if len(high_idx) > 0 else 1.0

    return {
        'x': x,
        'selected_count': selected_count,
        'total_time': round(total_time, 2),
        'avg_accuracy': round(avg_accuracy, 4),
        'total_score': round(total_score, 4),
        'is_high_covered': round(is_high_covered, 4),
    }


# ---------- 雷达图 ----------

def plot_ahp_radar(
    labels: Optional[List[str]] = None,
    weights: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if labels is None:
        labels = [
            '紧急程度\n(时效/加急)',
            '紧急程度\n(资金/政策)',
            '错分风险\n(低置信度)',
            '错分风险\n(多类别/模糊)',
            '复核必要性\n(核心业务)',
            '复核必要性\n(合规要求)',
        ]
    if weights is None:
        weights, _ = ahp_normalize(_AHP_MATRIX)

    n = len(weights)
    if len(labels) != n:
        raise ValueError(f'标签数 ({len(labels)}) 与权重数 ({n}) 不一致')

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values = np.concatenate((weights, [weights[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    labels_closed = np.concatenate((labels, [labels[0]]))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='#3498db', alpha=0.25, label='AHP 权重')
    ax.plot(angles, values, color='#2980b9', linewidth=2.5, marker='o',
            markersize=8, markerfacecolor='#e74c3c')

    for a, v in zip(angles[:-1], weights):
        ax.annotate(f'{v:.2f}', xy=(a, v), xytext=(10, 10),
                     textcoords='offset points', fontsize=11,
                     color='#2c3e50', fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_ylim(0, max(weights) * 1.3)
    ax.set_title('AHP 专家打分权重多维度雷达图', size=16, pad=25,
                  fontweight='bold', color='#2c3e50')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] 雷达图已保存至: {save_path}')
    else:
        plt.show()


# ---------- 全流程演示 ----------

def demo_full_pipeline() -> Dict:
    print('=' * 65)
    print('  问题三：AHP-EWM + TOPSIS + 0-1 规划 全流程演示')
    print('=' * 65)

    dim_names = [
        '紧急程度(时效/加急)', '紧急程度(资金/政策)',
        '错分风险(低置信度)', '错分风险(多类别/模糊)',
        '复核必要性(核心业务)', '复核必要性(合规要求)',
    ]

    print('\n[Step 1] AHP 判断矩阵权重计算')
    w_ahp, norm_mat = ahp_normalize(_AHP_MATRIX)
    consistency = ahp_consistency_check(_AHP_MATRIX)
    for name, w in zip(dim_names, w_ahp):
        print(f'  {name}: {w:.4f}')
    status = '[PASS]' if consistency['is_consistent'] else '[FAIL]'
    print(f'\n  一致性检验: lambda_max={consistency["lambda_max"]}, '
          f'CI={consistency["CI"]}, CR={consistency["CR"]} {status}')

    print('\n[Step 2] EWM 熵权法（模拟文件指标数据）')
    rng = np.random.RandomState(42)
    n_files = 100
    sim_data = np.zeros((n_files, 6))
    for i in range(n_files):
        sim_data[i] = [
            rng.beta(2, 5), rng.beta(1.5, 5), rng.beta(3, 3),
            rng.beta(2, 4), rng.beta(2.5, 3), rng.beta(1.5, 5),
        ]
    w_ewm = entropy_weight_method(sim_data)
    for name, w in zip(dim_names, w_ewm):
        print(f'  {name}: {w:.4f}')

    print('\n[Step 3] AHP-EWM 组合赋权 (α=0.5)')
    w_combined = combine_weights(w_ahp, w_ewm, alpha=0.5)
    for name, w in zip(dim_names, w_combined):
        print(f'  {name}: {w:.4f}')

    print('\n[Step 4] TOPSIS 综合评价')
    scores = topsis_evaluate(sim_data, w_combined)
    classification = classify_by_score(scores)
    print(f'  高分文件 (强制复核): {len(classification["high"])} 个')
    print(f'  中分文件:             {len(classification["medium"])} 个')
    print(f'  低分文件 (自动归档): {len(classification["low"])} 个')

    print('\n[Step 5] 0-1 整数规划优化')
    time_costs = rng.uniform(0.5, 3.0, n_files)
    accuracy = rng.uniform(0.80, 0.99, n_files)
    result = zero_one_knapsack_optimization(
        scores=scores, time_costs=time_costs, accuracy_rates=accuracy,
        total_time_budget=80.0,
    )
    print(f'  选中文件数:     {result["selected_count"]} / {n_files}')
    print(f'  总耗时:         {result["total_time"]} 小时')
    print(f'  平均准确率:     {result["avg_accuracy"]:.2%}')
    print(f'  高分覆盖率:     {result["is_high_covered"]:.2%}')

    print('\n[Step 6] 生成 AHP 权重雷达图...')
    try:
        plot_ahp_radar()
        print('  雷达图已显示（关闭图片窗口继续程序）')
    except Exception as e:
        print(f'  [警告] 雷达图显示失败: {e}')

    print('\n' + '=' * 65)
    print('  全流程结束')
    print('=' * 65)

    return {
        'w_ahp': w_ahp, 'w_ewm': w_ewm, 'w_combined': w_combined,
        'consistency': consistency, 'scores': scores,
        'classification': classification, 'optimization': result,
    }


if __name__ == '__main__':
    demo_full_pipeline()
