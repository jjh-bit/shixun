import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==========================
# 1. 读取数据
# ==========================

df = pd.read_csv("糖尿病预测.csv")

print("数据集形状：")
print(df.shape)

print("\n前5行数据：")
print(df.head())

# ==========================
# 2. 数据编码
# ==========================

encoder = LabelEncoder()

for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================
# 3. 划分特征和标签
# ==========================

X = df.drop("class", axis=1)
y = df["class"]

# ==========================
# 4. 划分训练集测试集
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练集大小:", X_train.shape)
print("测试集大小:", X_test.shape)

# ==========================
# 5. 随机森林基线模型
# ==========================

rf = RandomForestClassifier(
    random_state=42
)

rf.fit(X_train, y_train)

# ==========================
# 6. 预测
# ==========================

y_pred = rf.predict(X_test)

y_prob = rf.predict_proba(X_test)[:, 1]

# ==========================
# 7. 评估指标
# ==========================

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n====================")
print("模型评估结果")
print("====================")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ==========================
# 8. 混淆矩阵
# ==========================

cm = confusion_matrix(y_test, y_pred)

print("\n混淆矩阵：")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ==========================
# 9. ROC曲线
# ==========================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC={auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

# ==========================
# 10. GridSearch调参
# ==========================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n====================")
print("最优参数")
print("====================")

print(grid.best_params_)

best_model = grid.best_estimator_

# ==========================
# 11. 最优模型评估
# ==========================

best_pred = best_model.predict(X_test)

best_prob = best_model.predict_proba(X_test)[:, 1]

print("\n====================")
print("调参后模型")
print("====================")

print(
    "Accuracy:",
    accuracy_score(y_test, best_pred)
)

print(
    "Recall:",
    recall_score(y_test, best_pred)
)

print(
    "F1:",
    f1_score(y_test, best_pred)
)

print(
    "AUC:",
    roc_auc_score(y_test, best_prob)
)

# ==========================
# 12. 风险预测结果
# ==========================

result = pd.DataFrame()

result["真实标签"] = y_test.values
result["预测标签"] = best_pred
result["患病概率"] = best_prob

print("\n前10个预测结果：")

print(result.head(10))

result.to_csv(
    "diabetes_prediction_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n预测结果已保存：")
print("diabetes_prediction_result.csv")