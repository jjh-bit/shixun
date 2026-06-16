# 糖尿病风险预测系统

基于患者健康指标数据，利用机器学习构建糖尿病早期风险预测模型。项目采用 **A → B → C → D 四阶段流水线**架构，每位成员独立负责一个环节，通过 CSV 文件交接，最终由 `main.py` 一键串联。

## 项目结构

```
shixun/
├── main.py                          # 一键自动化流水线入口
├── requirements.txt                 # Python 依赖
├── data/                            # 数据目录（原始数据 + 中间产出）
│   └── 糖尿病预测.csv                # 原始数据集（520 条 × 17 列）
├── src/                             # 源代码
│   ├── __init__.py                  # 包初始化
│   ├── A_data_preprocessing.py      # 阶段 A：数据清洗与标准化
│   ├── B_特征工程.py                 # 阶段 B：特征构建、选择、降维
│   ├── C_数据集划分.py               # 阶段 C-1：训练/验证/测试集划分
│   ├── C_模型训练.py                 # 阶段 C-2：5 类模型训练
│   └── D_evaluation.py              # 阶段 D：独立测试集评估
├── models/                          # 训练产出的模型文件（自动生成）
├── results/                         # 评估图表与报告（自动生成）
└── README.md                        # 本文件
```

## 流水线架构

```
原始数据 (data/糖尿病预测.csv)
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 A：数据预处理                   │
│  清洗 → 二值化编码 → Age 标准化        │
│  输出: data/cleaned_糖尿病预测.csv    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 B：特征工程                     │
│  EDA → 相关性热力图 → VIF 共线性检查    │
│  医学组合特征 → 卡方/互信息/Lasso 筛选  │
│  PCA 降维可视化                       │
│  输出: data/特征工程输出_给C成员.csv   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 C-1：数据集划分                 │
│  分层抽样 → train(70%) / val(15%)    │
│           / test(15%)               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 C-2：模型训练                   │
│  train + val 合并（85%）训练 5 个模型  │
│  Logistic Regression / Random Forest │
│  XGBoost / LightGBM / MLP            │
│  输出: models/{5个模型}.pkl          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  阶段 D：模型评估                     │
│  test(15%) 独立评估全部 5 个模型       │
│  混淆矩阵 + ROC 对比 + 指标排名        │
│  输出: results/{图表与报告}           │
└─────────────────────────────────────┘
```

## 小组成员与分工

| 成员 | 角色 | 职责范围 |
|------|------|----------|
| **A** | 数据研发工程师 | 数据清洗、缺失值处理、二值化编码、数值标准化、日志记录 |
| **B** | 特征研发工程师 | EDA 探索、相关性热力图、VIF 共线性检查、医学组合特征构建、卡方/互信息/Lasso 特征选择、PCA 降维可视化 |
| **C** | 算法工程师 | 数据集划分、5 类模型训练（LR / RF / XGBoost / LightGBM / MLP）、模型持久化 |
| **D** | 测试与统筹 | 独立测试集评估、混淆矩阵、ROC 对比、模型排名、最终报告 |

## 环境要求

- Python 3.8+
- 依赖库见 [requirements.txt](requirements.txt)

核心依赖：
```
numpy, pandas, matplotlib, seaborn
scikit-learn, xgboost, lightgbm
joblib
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行全流水线

```bash
cd shixun
python main.py
```

> `main.py` 自动定位项目根目录（基于 `__file__`），不依赖当前工作目录。可在任意位置运行。

### 3. 逐步运行（调试用）

```bash
# 阶段 A — 数据预处理
python src/A_data_preprocessing.py

# 阶段 B — 特征工程
python src/B_特征工程.py

# 阶段 C-1 — 数据集划分
python src/C_数据集划分.py

# 阶段 C-2 — 模型训练
python src/C_模型训练.py

# 阶段 D — 模型评估
python src/D_evaluation.py
```

## 数据流与文件约定

| 阶段 | 输入 | 输出 |
|------|------|------|
| A | `data/糖尿病预测.csv` | `data/cleaned_糖尿病预测.csv` |
| B | `data/cleaned_糖尿病预测.csv` | `data/特征工程输出_给C成员.csv`、`data/特征工程全量特征_给C成员.csv`、`data/数据_编码后.csv` |
| C-1 | `data/特征工程输出_给C成员.csv` | `data/特征工程输出_train_70.csv`、`_val_15.csv`、`_test_15.csv` |
| C-2 | `data/特征工程输出_train_70.csv` + `_val_15.csv`（合并 85%） | `models/logistic_regression.pkl`、`random_forest.pkl`、`xgboost.pkl`、`lightgbm.pkl`、`mlp.pkl`、`scaler.pkl` |
| D | `data/特征工程输出_test_15.csv` + `models/*` | `results/` 下全部评估图表 |

> **标签约定**：`class=1` 为患病风险（Positive），`class=0` 为未患病（Negative）。

## 模型说明

全部 5 个模型仅使用训练集（train + val 合并，占 85% 数据）训练，测试集（15%）完全独立，留待 D 评估。

| 模型 | 类型 | 关键参数 |
|------|------|----------|
| Logistic Regression | 基线模型 | `max_iter=2000`, `class_weight='balanced'` |
| Random Forest | 树模型 | `n_estimators=150`, `max_depth=10`, `class_weight='balanced'` |
| XGBoost | 树模型 | `n_estimators=150`, `max_depth=8`, `learning_rate=0.05` |
| LightGBM | 树模型 | `n_estimators=100`, `max_depth=5`, `learning_rate=0.03` |
| MLP | 深度学习 | `hidden_layer_sizes=(256,128,64)`, `max_iter=1000` |

> Logistic Regression 和 MLP 在训练前使用 `StandardScaler` 标准化；树模型使用原始特征。

## 评估指标

D 阶段在独立测试集上评估全部模型，输出以下指标：

- **Accuracy**（准确率）
- **Precision**（精确率）
- **Recall**（召回率 — 医疗场景重点关注，对应漏诊率）
- **F1 Score**
- **AUC-ROC**

## B 成员产出物清单

| 文件 | 说明 |
|------|------|
| `results/01_相关性热力图.png` | 全部特征相关性热力图 |
| `results/02_PCA降维分析.png` | PCA 二维投影 + 累计解释方差 |
| `results/特征重要性排名.csv` | 综合卡方/互信息/相关系数/Lasso 的特征排名 |
| `results/VIF_多重共线性检查.csv` | 多重共线性检查结果 |
| `results/特征字段说明.csv` | 所有特征的字段解释与推荐标记 |
| `results/B成员交付说明_给C成员.md` | 面向 C 成员的接入说明文档 |

## 接入指南（跨成员协作）

### C 成员接入 B 成员输出

1. 读取 `data/特征工程输出_给C成员.csv`（推荐）或 `data/特征工程全量特征_给C成员.csv`（对照实验）
2. 将 `class` 列作为标签 `y`，其余列作为特征 `X`
3. 使用 `train_test_split(..., stratify=y, random_state=42)` 保持类别均衡
4. 树模型可直接使用特征；LR/MLP 建议先做 `StandardScaler`

### D 成员接入 C 成员输出

1. 从 `data/特征工程输出_test_15.csv` 加载独立测试集
2. 从 `models/` 加载全部 `.pkl` 模型文件
3. LR/MLP 需使用 `models/scaler.pkl` 对测试集标准化
4. 优先关注 Recall、AUC-ROC、F1-Score

## License

见 [LICENSE](LICENSE) 文件。
