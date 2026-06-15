import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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
# 1. 读取成员B的特征工程数据
# ==========================

df = pd.read_csv("特征工程输出_给C成员.csv")

print("数据集形状:")
print(df.shape)

# ==========================
# 2. 特征和标签
# ==========================

X = df.drop("class", axis=1)
y = df["class"]

# ==========================
# 3. 划分训练集和测试集
# （必须和成员C训练时保持一致）
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# 4. 加载成员C训练好的模型
# ==========================

model = joblib.load("lightgbm_medical_best.pkl")

print("\nLightGBM模型加载成功")

# ==========================
# 5. 预测
# ==========================

# LightGBM Booster 返回概率

y_prob = model.predict(X_test)

# 概率转分类标签

y_pred = (y_prob >= 0.5).astype(int)

print("预测概率前10个:")
print(y_prob[:10])

print("预测标签前10个:")
print(y_pred[:10])



# ==========================
# 6. 模型评估
# ==========================

acc = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("\n======================")
print("模型评估结果")
print("======================")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred))

# ==========================
# 7. 混淆矩阵
# ==========================

cm = confusion_matrix(y_test, y_pred)

print("\n混淆矩阵:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# 8. ROC曲线
# ==========================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.4f}"
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

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# 9. 风险预测结果导出
# ==========================

result = pd.DataFrame()

result["真实标签"] = y_test.values
result["预测标签"] = y_pred
result["患病概率"] = y_prob

result.to_csv(
    "diabetes_prediction_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n预测结果已保存：")
print("diabetes_prediction_result.csv")

print("\n前10条预测结果：")
print(result.head(10))