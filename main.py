# 论文代码主入口：多源异构文件识别与分类优化
#
# 用法:
#   python main.py              # 跑全部
#   python main.py --problem 3  # 只跑问题三
#   python main.py --plot       # 带图

import sys
import os
import argparse
import numpy as np

# Windows控制台utf-8兼容
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def run_problem_1(show_plot: bool = False) -> dict:
    # 问题一：文本预处理 + TF-IDF + K-Means无监督聚类
    from preprocessing import (
        simulate_multiformat_files, preprocess_document, tfidf_vectorize,
    )
    from models import kmeans_cluster, silhouette_score, auto_name_clusters

    print('\n' + '█' * 60)
    print('  问题一：多源异构文件无监督聚类')
    print('█' * 60)

    print('\n>>> 1. 加载多格式文件...')
    files = simulate_multiformat_files()
    print(f'   共加载 {len(files)} 个文件，覆盖格式: '
          f'{set(f["format"] for f in files)}')

    print('\n>>> 2. 文本预处理 (清洗 → Jieba 分词 → 停用词过滤)...')
    documents = [preprocess_document(f['content']) for f in files]
    print(f'   完成 {len(documents)} 篇文档的预处理')

    print('\n>>> 3. TF-IDF 向量化...')
    matrix, vocab = tfidf_vectorize(documents)
    print(f'   特征矩阵: {matrix.shape[0]} 文档 × {matrix.shape[1]} 特征词')

    print('\n>>> 4. K-Means 聚类 (K=5 演示, 实际 K=27)...')
    n_clusters = min(5, matrix.shape[0] - 1)
    labels, centers = kmeans_cluster(matrix, n_clusters=n_clusters)
    print(f'   聚类数 K = {n_clusters}')
    for k in range(n_clusters):
        count = int(np.sum(labels == k))
        print(f'   簇 {k+1}: {count} 个文件')

    print('\n>>> 5. 聚类质量评估...')
    sc = silhouette_score(matrix, labels)
    print(f'   轮廓系数 SC = {sc:.4f} (论文值: 0.42)')

    print('\n>>> 6. 聚类标签自动生成...')
    keywords = auto_name_clusters(matrix, labels, vocab)
    for k, words in keywords.items():
        print(f'   簇 {k+1}: {" | ".join(words[:5])}')

    if show_plot:
        print('\n>>> 7. 生成聚类可视化图...')
        from visualization import plot_clustering_2d
        try:
            plot_clustering_2d(matrix, labels, centers)
        except Exception as e:
            print(f'   [跳过] 图形显示失败: {e}')

    return {
        'n_documents': len(files),
        'n_features': matrix.shape[1],
        'n_clusters': n_clusters,
        'sc': sc,
        'cluster_sizes': [int(np.sum(labels == k)) for k in range(n_clusters)],
    }


def run_problem_2(show_plot: bool = False) -> dict:
    # 问题二：伪标签 + XGBoost半监督迁移分类 + PSI/熵评估
    from preprocessing import (
        simulate_multiformat_files, preprocess_document, tfidf_vectorize,
    )
    from models import (
        kmeans_cluster, SimpleXGBoost, classification_report,
        population_stability_index,
    )

    print('\n' + '█' * 60)
    print('  问题二：XGBoost 半监督迁移分类与评估')
    print('█' * 60)

    print('\n>>> 1. 准备训练数据（伪标签来自问题一聚类）...')
    files = simulate_multiformat_files()
    documents = [preprocess_document(f['content']) for f in files]
    matrix, vocab = tfidf_vectorize(documents)
    labels_all, _ = kmeans_cluster(matrix, n_clusters=min(3, matrix.shape[0] - 1))
    y_train = (labels_all == 0).astype(np.int32)
    print(f'   训练样本: {len(y_train)}, 正类占比: {y_train.mean():.2%}')

    print('\n>>> 2. 训练 XGBoost 分类器...')
    model = SimpleXGBoost(n_estimators=80, learning_rate=0.1)
    model.fit(matrix, y_train)
    print(f'   完成，共训练 {len(model._trees)} 棵树')

    print('\n>>> 3. 迁移预测 — 数据集2（模拟轻微分布偏移）...')
    rng = np.random.RandomState(42)
    noise_2 = rng.randn(*matrix.shape) * 0.3
    X2 = matrix + noise_2
    proba_2 = model.predict_proba(X2)
    y_pred_2 = model.predict(X2)
    metrics_2 = classification_report(y_train, y_pred_2, proba_2)
    psi_2 = population_stability_index(model.predict_proba(matrix), proba_2)
    print(f'   准确率: {metrics_2["准确率 (Accuracy)"]:.2%}')
    print(f'   精确率: {metrics_2["精确率 (Precision)"]:.2%}  (论文: 82.50%)')
    print(f'   PSI:    {psi_2:.4f}  (论文: 0.085)')
    print(f'   预测熵: {metrics_2["预测熵 (Entropy/bit)"]:.4f} bit')

    print('\n>>> 4. 迁移预测 — 数据集3（模拟较大分布偏移）...')
    noise_3 = rng.randn(*matrix.shape) * 0.5
    X3 = matrix + noise_3
    proba_3 = model.predict_proba(X3)
    y_pred_3 = model.predict(X3)
    metrics_3 = classification_report(y_train, y_pred_3, proba_3)
    psi_3 = population_stability_index(model.predict_proba(matrix), proba_3)
    print(f'   准确率: {metrics_3["准确率 (Accuracy)"]:.2%}')
    print(f'   精确率: {metrics_3["精确率 (Precision)"]:.2%}  (论文: 86.15%)')
    print(f'   PSI:    {psi_3:.4f}  (论文: 0.062)')
    print(f'   预测熵: {metrics_3["预测熵 (Entropy/bit)"]:.4f} bit')

    print('\n>>> 5. 高风险模糊样本识别（高熵 + 中概率区间）...')
    high_entropy_mask = (
        (-proba_3 * np.log2(np.clip(proba_3, 1e-12, 1))
         - (1 - proba_3) * np.log2(np.clip(1 - proba_3, 1e-12, 1)))
        > 0.8
    )
    mid_proba_mask = (proba_3 > 0.3) & (proba_3 < 0.7)
    risky = high_entropy_mask & mid_proba_mask
    print(f'   高风险样本数: {risky.sum()} / {len(proba_3)} '
          f'({risky.sum()/len(proba_3):.1%}, 论文: ~10%~12%)')

    return {
        'train_samples': len(y_train),
        'accuracy_2': metrics_2['准确率 (Accuracy)'],
        'precision_2': metrics_2['精确率 (Precision)'],
        'psi_2': psi_2,
        'accuracy_3': metrics_3['准确率 (Accuracy)'],
        'precision_3': metrics_3['精确率 (Precision)'],
        'psi_3': psi_3,
        'risky_pct': float(risky.sum() / len(proba_3)),
    }


def run_problem_3(show_plot: bool = False) -> dict:
    # 问题三：AHP-EWM组合赋权 + TOPSIS评分 + 0-1整数规划
    from decision import (
        ahp_normalize, ahp_consistency_check, _AHP_MATRIX,
        entropy_weight_method, combine_weights,
        topsis_evaluate, classify_by_score,
        zero_one_knapsack_optimization, plot_ahp_radar,
    )

    print('\n' + '█' * 60)
    print('  问题三：AHP-EWM + TOPSIS + 0-1 整数规划')
    print('█' * 60)

    dim_names = [
        '紧急程度(时效/加急)', '紧急程度(资金/政策)',
        '错分风险(低置信度)', '错分风险(多类别/模糊)',
        '复核必要性(核心业务)', '复核必要性(合规要求)',
    ]

    print('\n>>> 1. AHP 判断矩阵与权重计算...')
    w_ahp, _ = ahp_normalize(_AHP_MATRIX)
    consistency = ahp_consistency_check(_AHP_MATRIX)
    for name, w in zip(dim_names, w_ahp):
        print(f'   {name}: {w:.4f}')
    status = '[PASS] CR<0.1' if consistency['is_consistent'] else '[FAIL]'
    print(f'   一致性检验: lambda_max={consistency["lambda_max"]}, '
          f'CR={consistency["CR"]} {status}')

    print('\n>>> 2. EWM 熵权法（模拟100个文件的6维指标数据）...')
    rng = np.random.RandomState(42)
    n_files = 100
    sim_data = np.column_stack([
        rng.beta(2, 5, n_files), rng.beta(1.5, 5, n_files),
        rng.beta(3, 3, n_files), rng.beta(2, 4, n_files),
        rng.beta(2.5, 3, n_files), rng.beta(1.5, 5, n_files),
    ])
    w_ewm = entropy_weight_method(sim_data)
    for name, w in zip(dim_names, w_ewm):
        print(f'   {name}: {w:.4f}')

    print('\n>>> 3. AHP-EWM 组合赋权 (α=0.5)...')
    w_combined = combine_weights(w_ahp, w_ewm, alpha=0.5)
    for name, w in zip(dim_names, w_combined):
        print(f'   {name}: {w:.4f}')

    print('\n>>> 4. TOPSIS 综合评价与分级...')
    scores = topsis_evaluate(sim_data, w_combined)
    classification = classify_by_score(scores)
    print(f'   高分 (强制复核): {len(classification["high"])} 个')
    print(f'   中分:             {len(classification["medium"])} 个')
    print(f'   低分 (自动归档): {len(classification["low"])} 个')

    print('\n>>> 5. 0-1 整数规划优化（总时间预算 80 小时）...')
    time_costs = rng.uniform(0.5, 3.0, n_files)
    accuracy_rates = rng.uniform(0.80, 0.99, n_files)
    opt_result = zero_one_knapsack_optimization(
        scores=scores, time_costs=time_costs, accuracy_rates=accuracy_rates,
        total_time_budget=80.0,
    )
    print(f'   选中文件:   {opt_result["selected_count"]} / {n_files}')
    print(f'   总耗时:     {opt_result["total_time"]} h')
    print(f'   平均准确率: {opt_result["avg_accuracy"]:.2%}')
    print(f'   高分覆盖率: {opt_result["is_high_covered"]:.1%} (论文: 100%)')

    if show_plot:
        print('\n>>> 6. 生成可视化图表...')
        from visualization import (
            plot_ahp_radar, plot_topsis_distribution, plot_weight_comparison,
        )
        try:
            print('   生成 AHP 权重雷达图...')
            plot_ahp_radar()
            print('   生成 TOPSIS 评分分布图...')
            plot_topsis_distribution(scores, classification)
            print('   生成权重对比柱状图...')
            plot_weight_comparison(w_ahp, w_ewm, w_combined, dim_names)
        except Exception as e:
            print(f'   [跳过] 图形显示失败: {e}')

    return {
        'w_ahp': w_ahp.tolist(), 'w_ewm': w_ewm.tolist(),
        'w_combined': w_combined.tolist(), 'cr': consistency['CR'],
        'high_count': len(classification['high']),
        'medium_count': len(classification['medium']),
        'low_count': len(classification['low']),
        'selected_count': opt_result['selected_count'],
        'high_coverage': opt_result['is_high_covered'],
    }


def run_all(show_plot: bool = False) -> dict:
    print('=' * 65)
    print('  论文: 智能办公场景下多源异构文件识别与分类优化研究')
    print('  完整算法流程演示')
    print('=' * 65)
    results = {}
    results['problem_1'] = run_problem_1(show_plot=False)
    results['problem_2'] = run_problem_2(show_plot=False)
    results['problem_3'] = run_problem_3(show_plot=show_plot)

    print('\n' + '=' * 65)
    print('  全部流程结束 — 汇总')
    print('=' * 65)
    print(f'  问题一: {results["problem_1"]["n_documents"]} 文件 → '
          f'{results["problem_1"]["n_clusters"]} 类, '
          f'SC={results["problem_1"]["sc"]:.4f}')
    print(f'  问题二: 精确率(数据集2)={results["problem_2"]["precision_2"]:.2%}, '
          f'PSI(数据集2)={results["problem_2"]["psi_2"]:.4f}')
    print(f'  问题三: 高分{results["problem_3"]["high_count"]}个强制复核, '
          f'CR={results["problem_3"]["cr"]:.4f}')
    return results


def main():
    parser = argparse.ArgumentParser(
        description='论文代码 — 智能办公场景下多源异构文件识别与分类优化研究',
    )
    parser.add_argument('--problem', '-p', type=int, choices=[1, 2, 3],
                        help='仅运行指定问题 (1/2/3)')
    parser.add_argument('--plot', action='store_true',
                        help='显示matplotlib图形')
    args = parser.parse_args()

    if args.problem == 1:
        run_problem_1(show_plot=args.plot)
    elif args.problem == 2:
        run_problem_2(show_plot=args.plot)
    elif args.problem == 3:
        run_problem_3(show_plot=args.plot)
    else:
        run_all(show_plot=args.plot)


if __name__ == '__main__':
    main()
