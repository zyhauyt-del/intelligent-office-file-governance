# 智能办公场景下多源异构文件识别与分类优化研究

论文编号: BSHUWEI2603117

## 项目简介

这是论文《智能办公场景下多源异构文件识别与分类优化研究》的配套代码。

企事业单位数字化办公中积累的文件格式混杂（Word、PDF、Excel、图片等），传统人工归档效率很低。本文提出了一套递进式方案：

- **问题一**：用 TF-IDF + K-Means 对历史文件做无监督聚类，3397个文件分了27类，轮廓系数0.42
- **问题二**：用伪标签训练 XGBoost 做半监督迁移分类，加 PSI/预测熵/覆盖率多维评估，精确率 82%~86%
- **问题三**：AHP-EWM 主客观组合赋权 + TOPSIS 评分 + 0-1整数规划，高分文件100%强制复核

## 文件说明

- `main.py` — 主入口，可以跑单个问题或全部
- `preprocessing.py` — 文本清洗、jieba分词、TF-IDF向量化
- `models.py` — K-Means聚类、XGBoost分类器、PSI等评估指标
- `decision.py` — AHP、EWM、TOPSIS、0-1整数规划
- `visualization.py` — 雷达图、散点图、直方图等可视化

## 环境与运行

需要 Python 3.8+，先装依赖：

```
pip install numpy matplotlib jieba
```

跑起来：

```
python main.py              # 跑全部演示
python main.py --problem 3  # 只跑问题三
python main.py --plot       # 带可视化图表
```

## 主要结果

- 聚类 SC=0.42，未分类率 1.09%
- 分类精确率 82.50%~86.15%，PSI ≤ 0.085
- AHP 一致性 CR < 0.1，高分覆盖率 100%
- 成本相对人工复核降低约 39.6%

本代码仅供学习和研究使用。
