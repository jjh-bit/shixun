# 糖尿病预测系统

## 项目概述

基于患者健康指标数据，利用机器学习方法构建糖尿病早期预测模型，判断患者是否存在糖尿病风险。

## 项目结构

```
糖尿病预测/
├── data/                    # 数据文件
│   ├── 糖尿病预测.csv           # 原始数据集
│   ├── 数据_编码后.csv          # 预处理编码后数据
│   └── 特征工程输出_给C成员.csv  # 特征工程后最终数据
├── src/                     # 源代码
│   ├── A_数据预处理.py         # [成员A] 数据清洗与标准化
│   ├── B_特征工程.py          # [成员B] 特征构建与选择
│   ├── C_使用B特征示例.py      # [协作示例] C成员读取B特征并跑基线
│   ├── C_模型训练.py          # [成员C] 模型训练
│   └── D_测试评估.py          # [成员D] 评估与报告
├── notebooks/               # Jupyter Notebook
├── models/                  # 训练好的模型文件
├── results/                 # 结果图表与报告
├── README.md                # 本文件
└── requirements.txt         # 依赖库
```

## 算法流程

```
原始数据 → 数据清洗 → 特征编码 → 特征构建 → 特征选择
    → 模型训练(Logistic Regression / Random Forest / XGBoost)
    → 评估(AUC-ROC / Recall / F1-Score) → 预测输出
```

## 小组成员与分工

| 成员 | 角色 | 职责 |
|------|------|------|
| A | 数据研发工程师 | 数据清洗、EDA、量纲统一 |
| B | 特征研发工程师 | 特征构建、特征选择、相关性分析 |
| C | 算法工程师 | 模型设计、训练与调优 |
| D | 测试与统筹管理 | 模型评估、超参数调优、报告撰写 |

## 环境要求

- Python 3.7+
- PyTorch 2.x（可选）
- numpy, pandas, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm

## 复现步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 数据预处理（成员A）
python src/A_数据预处理.py

# 3. 特征工程（成员B）
python src/B_特征工程.py

# 3.1 C成员快速验证B的特征输出（可选）
python src/C_使用B特征示例.py

# 4. 模型训练（成员C）
python src/C_模型训练.py

# 5. 评估预测（成员D）
python src/D_测试评估.py
```

## 参数设置

### 成员 B：特征工程交付

成员 B 的脚本会从 `data/糖尿病预测.csv` 读取原始数据，完成编码、医学组合特征构建、相关性分析、VIF 共线性检查、SelectKBest 卡方检验、互信息、Lasso 特征选择和 PCA 降维可视化。

主要输出：

| 文件 | 用途 |
|------|------|
| `data/特征工程输出_给C成员.csv` | 推荐给 C 成员优先建模的数据，最后一列 `class` 为标签 |
| `data/特征工程全量特征_给C成员.csv` | 包含全部构建特征，供 C 成员做对照实验 |
| `results/B成员交付说明_给C成员.md` | 字段含义、推荐建模方式、TOP 特征说明 |
| `results/特征重要性排名.csv` | 综合卡方、互信息、目标相关性、Lasso 的特征排名 |
| `results/特征选择明细_给C成员.csv` | C 成员可查看的筛选依据 |
| `results/特征字段说明.csv` | 字段解释和是否推荐建模 |
| `results/VIF_多重共线性检查.csv` | 多重共线性检查 |
| `results/目标相关性排序.csv` | 特征与目标变量的相关性排序 |
| `results/01_相关性热力图.png` | 相关性热力图 |
| `results/02_PCA降维分析.png` | PCA 降维分析图 |

标签约定：

- `class=1`：Positive，存在糖尿病风险
- `class=0`：Negative，未标记为糖尿病风险

C 成员接入建议：

1. 优先读取 `data/特征工程输出_给C成员.csv`。
2. 将 `class` 作为 `y`，其余列作为 `X`。
3. 使用 `train_test_split(..., stratify=y, random_state=42)` 保持类别比例稳定。
4. 树模型可直接使用特征；逻辑回归、SVM、MLP 建议先做 `StandardScaler`。
5. 医疗场景优先关注 `Recall`、`AUC-ROC`、`F1-Score`，必要时调整分类阈值以减少漏诊。
