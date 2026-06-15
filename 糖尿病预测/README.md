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

# 4. 模型训练（成员C）
python src/C_模型训练.py

# 5. 评估预测（成员D）
python src/D_测试评估.py
```

## 参数设置

（待各成员完成模块后补充）
