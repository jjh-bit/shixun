import os
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

# ==========================================
# 0. 项目根目录定位（基于本文件位置，不依赖 CWD）
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")


def main():
    # 1. 加载 B 成员输出的特征工程数据
    data_path = os.path.join(DATA_DIR, "特征工程输出_给C成员.csv")
    df = pd.read_csv(data_path)

    print("\n>>> 正在执行阶段 4: 模型评估与输出诊断报告 (成员 D) ...")
    print("数据集形状:")
    print(df.shape)

    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # 2. 加载 C 成员训练好的模型
    model_path = os.path.join(MODEL_DIR, "lightgbm_medical_best.pkl")
    model = joblib.load(model_path)

    print("\nLightGBM模型加载成功")

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    print("预测概率前10个:")
    print(y_prob[:10])
    print("预测标签前10个:")
    print(y_pred[:10])

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

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")

    # 输出到 results/ 目录而非 CWD
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    result = pd.DataFrame()
    result["真实标签"] = y_test.values
    result["预测标签"] = y_pred
    result["患病概率"] = y_prob

    result.to_csv(os.path.join(RESULT_DIR, "diabetes_prediction_result.csv"),
                  index=False, encoding="utf-8-sig")

    print(f"\n预测结果已保存：{os.path.join(RESULT_DIR, 'diabetes_prediction_result.csv')}")

if __name__ == "__main__":
    main()