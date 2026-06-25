# 可视化：雷达图、聚类散点图、TOPSIS分布图、权重对比柱状图

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple


# ---------- 字体 & 配色 ----------

def setup_chinese_font() -> str:
    import matplotlib.font_manager as fm
    candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'Source Han Sans SC', 'DejaVu Sans']
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font_name
    print('[警告] 未找到中文字体，图表中文标签可能显示为方块。')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


_USED_FONT = setup_chinese_font()

_COLORS = {
    'blue': '#3498db', 'dark_blue': '#2980b9', 'red': '#e74c3c',
    'green': '#2ecc71', 'orange': '#f39c12', 'purple': '#9b59b6',
    'grey': '#95a5a6', 'dark': '#2c3e50',
}


# ---------- 雷达图 ----------

def plot_ahp_radar(
    labels: Optional[List[str]] = None,
    values: Optional[np.ndarray] = None,
    title: str = 'AHP 专家打分权重多维度雷达图',
    save_path: Optional[str] = None,
) -> None:
    if labels is None:
        labels = np.array([
            '紧急程度\n(时效/加急)', '紧急程度\n(资金/政策)',
            '错分风险\n(低置信度)', '错分风险\n(多类别/模糊)',
            '复核必要性\n(核心业务)', '复核必要性\n(合规要求)',
        ])
    if values is None:
        values = np.array([0.20, 0.15, 0.20, 0.15, 0.18, 0.12])

    n = len(values)
    if len(labels) != n:
        raise ValueError(f'标签数({len(labels)})与数据数({n})不匹配')

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values_closed = np.concatenate((values, [values[0]]))
    angles_closed = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, values_closed, color=_COLORS['blue'], alpha=0.25)
    ax.plot(angles_closed, values_closed, color=_COLORS['dark_blue'],
            linewidth=2.5, marker='o', markersize=8,
            markerfacecolor=_COLORS['red'])

    for a, v in zip(angles, values):
        ax.annotate(f'{v:.2f}', xy=(a, v), xytext=(12, 10),
                     textcoords='offset points', fontsize=11,
                     color=_COLORS['dark'], fontweight='bold')

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=11, color=_COLORS['dark'])
    ax.set_yticklabels([])
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_title(title, size=16, pad=25, fontweight='bold', color=_COLORS['dark'])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] 雷达图已保存: {save_path}')
    else:
        plt.show()


# ---------- 聚类散点图 ----------

def plot_clustering_2d(
    matrix: np.ndarray,
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    title: str = 'K-Means 聚类结果 (PCA 降维投影)',
    save_path: Optional[str] = None,
) -> None:
    # PCA降维到2D
    if matrix.shape[1] > 2:
        X_centered = matrix - matrix.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        reduced = (U[:, :2] * S[:2])
    else:
        reduced = matrix.copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(n_clusters)]

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                    c=[colors[i]], label=f'簇 {lbl + 1}',
                    alpha=0.7, s=60, edgecolors='white', linewidth=0.5)

    if centers is not None:
        if centers.shape[1] > 2:
            centers_centered = centers - matrix.mean(axis=0)
            centers_reduced = centers_centered @ Vt[:2, :].T
        else:
            centers_reduced = centers
        ax.scatter(centers_reduced[:, 0], centers_reduced[:, 1],
                    c='red', marker='X', s=200, edgecolors='black',
                    linewidth=1.5, zorder=10, label='聚类中心')

    ax.set_xlabel('主成分 1', fontsize=12)
    ax.set_ylabel('主成分 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


# ---------- TOPSIS 分布直方图 ----------

def plot_topsis_distribution(
    scores: np.ndarray,
    classification: Optional[Dict[str, np.ndarray]] = None,
    title: str = 'TOPSIS 综合评分分布',
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 21)
    if classification:
        for level, idx, color, label in [
            ('high', classification['high'], _COLORS['red'], '高 (强制复核)'),
            ('medium', classification['medium'], _COLORS['orange'], '中'),
            ('low', classification['low'], _COLORS['green'], '低 (自动归档)'),
        ]:
            if len(idx) > 0:
                ax.hist(scores[idx], bins=bins, color=color, alpha=0.6,
                         label=f'{label}: {len(idx)} 个')
    else:
        ax.hist(scores, bins=bins, color=_COLORS['blue'], alpha=0.6)

    ax.axvline(x=0.33, color=_COLORS['grey'], linestyle='--',
                linewidth=1.5, label='低/中 阈值 (0.33)')
    ax.axvline(x=0.67, color=_COLORS['dark'], linestyle='--',
                linewidth=1.5, label='中/高 阈值 (0.67)')
    ax.set_xlabel('TOPSIS 综合评分 Cᵢ', fontsize=12)
    ax.set_ylabel('文件数量', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


# ---------- 权重对比柱状图 ----------

def plot_weight_comparison(
    w_ahp: np.ndarray,
    w_ewm: np.ndarray,
    w_combined: np.ndarray,
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    n = len(w_ahp)
    if labels is None:
        labels = [f'指标 {i+1}' for i in range(n)]

    x = np.arange(n)
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, w_ahp, width, color=_COLORS['blue'],
                    alpha=0.8, label='AHP (主观)')
    bars2 = ax.bar(x, w_ewm, width, color=_COLORS['orange'],
                    alpha=0.8, label='EWM (客观)')
    bars3 = ax.bar(x + width, w_combined, width, color=_COLORS['green'],
                    alpha=0.8, label='AHP-EWM 组合')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=8, rotation=45,
                         color=_COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('权重值', fontsize=12)
    ax.set_title('AHP-EWM 主客观组合赋权对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(w_ahp), max(w_ewm), max(w_combined)) * 1.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    print(f'使用字体: {_USED_FONT}')
    print('\n[测试] 绘制 AHP 雷达图...')
    plot_ahp_radar()

    print('\n[测试] 绘制聚类散点图...')
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.randn(50, 5) + [0, 0, 0, 0, 0],
        rng.randn(50, 5) + [3, 3, 3, 3, 3],
    ])
    lbls = np.array([0]*50 + [1]*50)
    plot_clustering_2d(X, lbls)
