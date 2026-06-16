import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ======================================
# 路径设置
# ======================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# ======================================
# 加载测试集
# ======================================

test_path = os.path.join(
    DATA_DIR,
    "特征工程输出_test_15.csv"
)

df = pd.read_csv(test_path)

print("测试集大小：", df.shape)

X_test = df.drop("class", axis=1)
y_test = df["class"]

# ======================================
# 模型列表
# ======================================

models = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "LightGBM": "lightgbm.pkl",
    "MLP": "mlp.pkl"
}

# ======================================
# 加载Scaler
# ======================================

scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

scaler = None

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler加载成功")

# ======================================
# 结果记录
# ======================================

results = []

plt.figure(figsize=(10, 8))

best_auc = -1
best_model_name = None

# ======================================
# 遍历所有模型
# ======================================

for model_name, model_file in models.items():

    model_path = os.path.join(MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        print(f"{model_file} 不存在，跳过")
        continue

    print("\n==============================")
    print(f"评估模型: {model_name}")
    print("==============================")

    model = joblib.load(model_path)

    X_input = X_test.copy()

    # Logistic和MLP一般需要标准化
    if scaler is not None and (
            "Logistic" in model_name
            or "MLP" in model_name
    ):
        X_input = scaler.transform(X_input)

    # -------------------------
    # 获取预测概率
    # -------------------------

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_input)[:, 1]

    else:

        y_prob = model.predict(X_input)

        if isinstance(y_prob, np.ndarray):
            y_prob = y_prob.astype(float)

    y_pred = (y_prob >= 0.5).astype(int)

    # -------------------------
    # 指标
    # -------------------------

    acc = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    results.append([
        model_name,
        acc,
        precision,
        recall,
        f1,
        auc
    ])

    # -------------------------
    # ROC曲线
    # -------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f"{model_name} (AUC={auc:.4f})"
    )

    # -------------------------
    # 混淆矩阵
    # -------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"{model_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    if auc > best_auc:
        best_auc = auc
        best_model_name = model_name

# ======================================
# ROC总图
# ======================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "ROC_Comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ======================================
# 输出结果表
# ======================================

result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]
)

result_df = result_df.sort_values(
    by="AUC",
    ascending=False
)

result_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "Model_Comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n==============================")
print("模型综合排名")
print("==============================")

print(result_df)

print("\n最佳模型：")
print(best_model_name)

print(f"AUC = {best_auc:.4f}")

print("\n结果保存位置：")
print(RESULT_DIR)