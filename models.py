# K-Means聚类、XGBoost分类器、评估指标

import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------- K-Means 聚类 ----------

def _validate_input(matrix: np.ndarray, n_clusters: int) -> None:
    # 参数校验
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f'特征矩阵必须是 numpy.ndarray，当前类型: {type(matrix)}')
    if matrix.ndim != 2:
        raise ValueError(f'特征矩阵必须是二维数组，当前维度: {matrix.ndim}')
    n_samples, n_features = matrix.shape
    if n_samples < 2:
        raise ValueError(f'样本数量 ({n_samples}) 不足，至少需要 2 个样本')
    if n_features < 1:
        raise ValueError('特征数量不能为 0')
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError(f'聚类数 K 必须是 >= 2 的整数，当前值: {n_clusters}')
    if n_clusters > n_samples:
        raise ValueError(f'聚类数 K ({n_clusters}) 不能超过样本数 ({n_samples})')
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ValueError('特征矩阵包含 NaN 或 Inf 值，请先清洗数据')


def kmeans_cluster(
    matrix: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    # 标准K-Means，E步分配 + M步更新中心，收敛或达到最大迭代停止
    _validate_input(matrix, n_clusters)
    n_samples, n_features = matrix.shape
    rng = np.random.RandomState(random_state)

    indices = rng.choice(n_samples, n_clusters, replace=False)
    centers = matrix[indices].copy().astype(np.float64)
    labels = np.zeros(n_samples, dtype=np.int32)

    for iteration in range(max_iter):
        # E步：分配最近中心
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            diff = matrix - centers[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)
        new_labels = np.argmin(distances, axis=1)

        # M步：更新中心
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            members = matrix[new_labels == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = matrix[rng.choice(n_samples)]  # 空簇处理

        center_shift = np.sum((new_centers - centers) ** 2)
        centers = new_centers
        labels = new_labels
        if center_shift < tol:
            break

    return labels, centers


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    # 轮廓系数 SC ∈ [-1,1]，越大聚类质量越好
    n_samples = matrix.shape[0]
    if n_samples < 2:
        return 0.0
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0

    silhouette_vals = []
    for i in range(n_samples):
        # a(i)：到同簇样本的平均距离
        same_cluster = labels == labels[i]
        n_same = same_cluster.sum() - 1
        if n_same == 0:
            silhouette_vals.append(0.0)
            continue
        diff_same = matrix[i] - matrix[same_cluster]
        dist_same = np.sqrt(np.sum(diff_same ** 2, axis=1))
        a_i = dist_same.sum() / n_same

        # b(i)：到最近异簇的平均距离
        b_i = float('inf')
        for lbl in unique_labels:
            if lbl == labels[i]:
                continue
            other_cluster = labels == lbl
            diff_other = matrix[i] - matrix[other_cluster]
            dist_other = np.sqrt(np.sum(diff_other ** 2, axis=1))
            b_other = dist_other.mean()
            if b_other < b_i:
                b_i = b_other

        if a_i == 0 and b_i == 0:
            s_i = 0.0
        elif a_i <= b_i:
            s_i = 1.0 - a_i / b_i if b_i > 0 else 0.0
        else:
            s_i = b_i / a_i - 1.0 if a_i > 0 else 0.0
        silhouette_vals.append(s_i)

    return float(np.mean(silhouette_vals))


def auto_name_clusters(
    matrix: np.ndarray,
    labels: np.ndarray,
    vocab: List[str],
    top_n: int = 5
) -> Dict[int, List[str]]:
    # 取每簇TF-IDF均值最高的词作为类别标签
    cluster_keywords = {}
    for k in np.unique(labels):
        members = matrix[labels == k]
        if len(members) == 0:
            cluster_keywords[int(k)] = []
            continue
        centroid = members.mean(axis=0)
        top_indices = np.argsort(centroid)[::-1][:top_n]
        keywords = [vocab[i] for i in top_indices if i < len(vocab)]
        cluster_keywords[int(k)] = keywords
    return cluster_keywords


# ---------- XGBoost 简化实现 ----------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x_clipped))


class SimpleXGBoost:
    # 简化版梯度提升树，每轮用单层树桩拟合残差

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42
    ):
        if n_estimators < 1:
            raise ValueError(f'n_estimators 必须 >= 1，当前值: {n_estimators}')
        if not 0 < learning_rate <= 1:
            raise ValueError(f'learning_rate 必须在 (0, 1] 区间，当前值: {learning_rate}')
        if max_depth < 1:
            raise ValueError(f'max_depth 必须 >= 1，当前值: {max_depth}')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._trees: List[Dict] = []
        self._base_score: float = 0.0
        self._is_fitted: bool = False

    def _build_stump(self, X: np.ndarray, residuals: np.ndarray) -> Dict:
        # 遍历特征选最优分割点（中位数），最小化平方损失
        n_samples, n_features = X.shape
        best_feature = 0
        best_threshold = 0.0
        best_left_val = 0.0
        best_right_val = 0.0
        best_loss = float('inf')

        for feat in range(min(n_features, 50)):
            values = X[:, feat]
            if np.std(values) < 1e-8:
                continue
            threshold = np.median(values)
            left_mask = values <= threshold
            right_mask = ~left_mask
            if left_mask.sum() < 2 or right_mask.sum() < 2:
                continue
            left_residual = residuals[left_mask]
            right_residual = residuals[right_mask]
            left_val = left_residual.mean()
            right_val = right_residual.mean()
            loss = (np.sum((left_residual - left_val) ** 2)
                    + np.sum((right_residual - right_val) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_feature = feat
                best_threshold = threshold
                best_left_val = left_val
                best_right_val = right_val
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_val': best_left_val,
            'right_val': best_right_val,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleXGBoost':
        if X.shape[0] != len(y):
            raise ValueError(f'X 样本数 ({X.shape[0]}) 与 y 标签数 ({len(y)}) 不一致')
        if X.shape[0] < 2:
            raise ValueError(f'训练样本数不足: {X.shape[0]}')
        self._base_score = np.mean(y)
        pred = np.full(len(y), self._base_score, dtype=np.float64)
        self._trees = []
        for _ in range(self.n_estimators):
            residuals = y - _sigmoid(pred)
            stump = self._build_stump(X, residuals)
            self._trees.append(stump)
            for i in range(len(y)):
                if X[i, stump['feature']] <= stump['threshold']:
                    pred[i] += self.learning_rate * stump['left_val']
                else:
                    pred[i] += self.learning_rate * stump['right_val']
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError('模型尚未训练，请先调用 fit()')
        pred = np.full(X.shape[0], self._base_score, dtype=np.float64)
        for stump in self._trees:
            feat = stump['feature']
            threshold = stump['threshold']
            for i in range(X.shape[0]):
                if X[i, feat] <= threshold:
                    pred[i] += self.learning_rate * stump['left_val']
                else:
                    pred[i] += self.learning_rate * stump['right_val']
        return _sigmoid(pred)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int32)


# ---------- 评估指标 ----------

def prediction_entropy(proba: np.ndarray) -> float:
    # 预测熵 H = -[p*log2(p) + (1-p)*log2(1-p)]，越高越不确定
    proba = np.clip(proba, 1e-12, 1 - 1e-12)
    entropy = -(proba * np.log2(proba) + (1 - proba) * np.log2(1 - proba))
    return float(np.mean(entropy))


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10
) -> float:
    # PSI群体稳定性指标，<0.1稳定，>0.25显著偏移
    if len(expected) < bins or len(actual) < bins:
        return 0.0
    bin_edges = np.linspace(0, 1, bins + 1)
    e_hist, _ = np.histogram(expected, bins=bin_edges)
    a_hist, _ = np.histogram(actual, bins=bin_edges)
    e_pct = (e_hist / len(expected)) + 1e-6
    a_pct = (a_hist / len(actual)) + 1e-6
    psi = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi)


def coverage_rate(proba: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean(proba >= threshold))


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray
) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {}
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        '准确率 (Accuracy)': round(accuracy, 4),
        '精确率 (Precision)': round(precision, 4),
        '召回率 (Recall)': round(recall, 4),
        'F1-Score': round(f1, 4),
        '预测熵 (Entropy/bit)': round(prediction_entropy(proba), 4),
        '覆盖率 (Coverage)': round(coverage_rate(proba), 4),
    }


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    X = np.vstack([
        rng.randn(30, 5) + np.array([0, 0, 0, 0, 0]),
        rng.randn(30, 5) + np.array([3, 3, 3, 3, 3]),
        rng.randn(30, 5) + np.array([0, 3, 0, 3, 0]),
    ])
    print('=== K-Means 聚类测试 ===')
    labels, centers = kmeans_cluster(X, n_clusters=3)
    print(f'聚类标签分布: {np.bincount(labels)}')
    sc = silhouette_score(X, labels)
    print(f'轮廓系数 (SC): {sc:.4f}')

    print('\n=== XGBoost 分类器测试 ===')
    y = (labels == 0).astype(np.int32)
    model = SimpleXGBoost(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    proba = model.predict_proba(X)
    y_pred = model.predict(X)
    metrics = classification_report(y, y_pred, proba)
    for k, v in metrics.items():
        print(f'  {k}: {v}')
