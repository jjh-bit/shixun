# ===================== 适配你本地糖尿病数据集的完整代码 =====================
# 依赖安装：pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据处理
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取本地数据 =====================
# 把这里改成你的 CSV 文件路径
file_path = "糖尿病预测.csv"  # 例如：C:/data/diabetes_data.csv 或 ./diabetes_data.csv

df = pd.read_csv(file_path)
print("数据集形状：", df.shape)
print("\n前5行数据：")
print(df.head())
print("\n列名：")
print(df.columns.tolist())
print("\n标签分布：")
print(df['class'].value_counts())

# ===================== 2. 数据预处理 =====================
# 1）处理二分类特征（Yes/No）
binary_cols = [
    'Polyuria', 'Polydipsia', 'sudden weweakness', 'Polyphagia',
    'Genital thr', 'visual blurr', 'Itching', 'Irritability',
    'delayed he', 'partial pare', 'muscle stif', 'Alopecia', 'Obesity'
]
# 把 Yes/No 转为 1/0
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# 2）处理 Gender 特征（Male/Female）
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# 3）处理标签 class（Positive/Negative）
df['class'] = df['class'].map({'Positive': 1, 'Negative': 0})

# 4）划分特征和标签
X = df.drop('class', axis=1)
y = df['class']

print("\n预处理后的数据前5行：")
print(X.head())

# 5）划分训练集/测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 6）特征标准化（逻辑回归、MLP使用）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# ===================== 3. 通用评估函数 =====================
def evaluate_model(y_true, y_pred, y_pred_proba=None):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba[:, 1]) if y_pred_proba is not None else np.nan
    return [acc, pre, rec, f1, auc]

results = []
model_names = []

# ===================== 4. 基线模型：逻辑回归 =====================
print("\n" + "="*50)
print("【基线模型】逻辑回归")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_lr_pred = lr.predict(X_test_scaled)
y_lr_proba = lr.predict_proba(X_test_scaled)
lr_res = evaluate_model(y_test, y_lr_pred, y_lr_proba)

print(f"准确率: {lr_res[0]:.4f}")
print(f"精确率: {lr_res[1]:.4f}")
print(f"召回率: {lr_res[2]:.4f}")
print(f"F1分数: {lr_res[3]:.4f}")
print(f"AUC: {lr_res[4]:.4f}")

results.append(lr_res)
model_names.append("逻辑回归(基线)")

# ===================== 5. 随机森林 =====================
print("\n" + "="*50)
print("【集成模型】随机森林(基础版)")
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)

y_rf_pred = rf_base.predict(X_test)
y_rf_proba = rf_base.predict_proba(X_test)
rf_base_res = evaluate_model(y_test, y_rf_pred, y_rf_proba)

print(f"准确率: {rf_base_res[0]:.4f}")
print(f"精确率: {rf_base_res[1]:.4f}")
print(f"召回率: {rf_base_res[2]:.4f}")
print(f"F1分数: {rf_base_res[3]:.4f}")
print(f"AUC: {rf_base_res[4]:.4f}")

results.append(rf_base_res)
model_names.append("随机森林(基础)")

# 随机森林调参
print("\n随机森林开始网格搜索调参...")
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5, scoring="roc_auc", n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print("随机森林最优参数：", rf_grid.best_params_)

rf_best = rf_grid.best_estimator_
y_rf_best_pred = rf_best.predict(X_test)
y_rf_best_proba = rf_best.predict_proba(X_test)
rf_best_res = evaluate_model(y_test, y_rf_best_pred, y_rf_best_proba)

print("\n【集成模型】随机森林(调优版)")
print(f"准确率: {rf_best_res[0]:.4f}")
print(f"精确率: {rf_best_res[1]:.4f}")
print(f"召回率: {rf_best_res[2]:.4f}")
print(f"F1分数: {rf_best_res[3]:.4f}")
print(f"AUC: {rf_best_res[4]:.4f}")

results.append(rf_best_res)
model_names.append("随机森林(调优)")

# 特征重要性
feat_imp = pd.Series(rf_best.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 6))
feat_imp.sort_values(ascending=False).plot(kind="bar")
plt.title("随机森林 - 特征重要性")
plt.tight_layout()
plt.show()

# ===================== 6. XGBoost =====================
print("\n" + "="*50)
print("【集成模型】XGBoost(基础版)")
xgb_base = xgb.XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric="logloss"
)
xgb_base.fit(X_train, y_train)

y_xgb_pred = xgb_base.predict(X_test)
y_xgb_proba = xgb_base.predict_proba(X_test)
xgb_base_res = evaluate_model(y_test, y_xgb_pred, y_xgb_proba)

print(f"准确率: {xgb_base_res[0]:.4f}")
print(f"精确率: {xgb_base_res[1]:.4f}")
print(f"召回率: {xgb_base_res[2]:.4f}")
print(f"F1分数: {xgb_base_res[3]:.4f}")
print(f"AUC: {xgb_base_res[4]:.4f}")

results.append(xgb_base_res)
model_names.append("XGBoost(基础)")

# XGBoost调参
print("\nXGBoost开始随机搜索调参...")
from sklearn.model_selection import RandomizedSearchCV
xgb_param = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9]
}
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
    param_distributions=xgb_param,
    n_iter=10, cv=5, scoring="roc_auc", n_jobs=-1, random_state=42
)
xgb_search.fit(X_train, y_train)
print("XGBoost最优参数：", xgb_search.best_params_)

xgb_best = xgb_search.best_estimator_
y_xgb_best_pred = xgb_best.predict(X_test)
y_xgb_best_proba = xgb_best.predict_proba(X_test)
xgb_best_res = evaluate_model(y_test, y_xgb_best_pred, y_xgb_best_proba)

print("\n【集成模型】XGBoost(调优版)")
print(f"准确率: {xgb_best_res[0]:.4f}")
print(f"精确率: {xgb_best_res[1]:.4f}")
print(f"召回率: {xgb_best_res[2]:.4f}")
print(f"F1分数: {xgb_best_res[3]:.4f}")
print(f"AUC: {xgb_best_res[4]:.4f}")

results.append(xgb_best_res)
model_names.append("XGBoost(调优)")

# ===================== 7. LightGBM =====================
print("\n" + "="*50)
print("【集成模型】LightGBM(基础版)")
lgb_base = lgb.LGBMClassifier(random_state=42)
lgb_base.fit(X_train, y_train)

y_lgb_pred = lgb_base.predict(X_test)
y_lgb_proba = lgb_base.predict_proba(X_test)
lgb_base_res = evaluate_model(y_test, y_lgb_pred, y_lgb_proba)

print(f"准确率: {lgb_base_res[0]:.4f}")
print(f"精确率: {lgb_base_res[1]:.4f}")
print(f"召回率: {lgb_base_res[2]:.4f}")
print(f"F1分数: {lgb_base_res[3]:.4f}")
print(f"AUC: {lgb_base_res[4]:.4f}")

results.append(lgb_base_res)
model_names.append("LightGBM(基础)")

# LightGBM调参
print("\nLightGBM开始网格搜索调参...")
lgb_param = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1]
}
lgb_grid = GridSearchCV(
    lgb.LGBMClassifier(random_state=42),
    param_grid=lgb_param,
    cv=5, scoring="roc_auc", n_jobs=-1
)
lgb_grid.fit(X_train, y_train)
print("LightGBM最优参数：", lgb_grid.best_params_)

lgb_best = lgb_grid.best_estimator_
y_lgb_best_pred = lgb_best.predict(X_test)
y_lgb_best_proba = lgb_best.predict_proba(X_test)
lgb_best_res = evaluate_model(y_test, y_lgb_best_pred, y_lgb_best_proba)

print("\n【集成模型】LightGBM(调优版)")
print(f"准确率: {lgb_best_res[0]:.4f}")
print(f"精确率: {lgb_best_res[1]:.4f}")
print(f"召回率: {lgb_best_res[2]:.4f}")
print(f"F1分数: {lgb_best_res[3]:.4f}")
print(f"AUC: {lgb_best_res[4]:.4f}")

results.append(lgb_best_res)
model_names.append("LightGBM(调优)")

# ===================== 8. 深度学习 MLP（可选） =====================
print("\n" + "="*50)
print("【深度学习】多层感知机 MLP")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 转为张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class DiabetesMLP(nn.Module):
    def __init__(self, in_dim=X_train.shape[1]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

mlp = DiabetesMLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)
epochs = 50

# 训练
for epoch in range(epochs):
    mlp.train()
    total_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = mlp(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 测试
mlp.eval()
with torch.no_grad():
    mlp_proba = mlp(X_test_tensor).numpy()
mlp_pred = (mlp_proba > 0.5).astype(int).flatten()
mlp_proba_full = np.hstack([1 - mlp_proba, mlp_proba])

mlp_res = evaluate_model(y_test, mlp_pred, mlp_proba_full)
print(f"\n准确率: {mlp_res[0]:.4f}")
print(f"精确率: {mlp_res[1]:.4f}")
print(f"召回率: {mlp_res[2]:.4f}")
print(f"F1分数: {mlp_res[3]:.4f}")
print(f"AUC: {mlp_res[4]:.4f}")

results.append(mlp_res)
model_names.append("MLP深度学习")

# ===================== 9. 所有模型结果汇总 =====================
print("\n" + "="*60)
print("全部模型性能汇总表")
res_df = pd.DataFrame(
    results,
    columns=["准确率", "精确率", "召回率", "F1分数", "AUC"],
    index=model_names
)
print(res_df.round(4))