import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix)
import lightgbm as lgb

# ===================== 1. 路径配置（适配真实数据） =====================
# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据路径优先级：特征工程输出 > 全量特征 > 清洗后原始数据
DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "特征工程输出_给C成员.csv"),
    os.path.join(BASE_DIR, "data", "特征工程全量特征_给C成员.csv"),
    os.path.join(BASE_DIR, "data", "cleaned_糖尿病预测.csv")
]
# 输出路径
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 绘图配置（支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


# ===================== 2. 数据加载与校验（适配真实数据格式） =====================
def load_best_data(data_paths):
    """自动选择可用的最优数据集"""
    for path in data_paths:
        if os.path.exists(path):
            print(f"加载最优数据集：{os.path.basename(path)}")
            df = pd.read_csv(path)
            # 数据格式校验（确保标签列存在）
            if "class" not in df.columns:
                # 适配可能的标签列名（如cleaned数据可能用"是否患病"）
                if "是否患病" in df.columns:
                    df.rename(columns={"是否患病": "class"}, inplace=True)
                    print("ℹ标签列名已从'是否患病'改为'class'")
                else:
                    raise ValueError(f"数据集 {os.path.basename(path)} 无标签列（需'class'或'是否患病'）")
            # 查看数据基本信息
            print(f"数据集形状：{df.shape} | 标签分布：")
            print(df["class"].value_counts())
            return df
    raise FileNotFoundError("未找到任何数据文件，请检查data目录下是否有3份数据之一")


def split_data(df, test_size=0.3, random_state=42):
    """分层划分训练集/测试集（医疗场景保证类别均衡）"""
    X = df.drop("class", axis=1)
    y = df["class"]
    # 处理可能的标签值（如1/0或Positive/Negative）
    if y.dtype == "object":
        y = y.map({"Positive": 1, "Negative": 0})
        print("ℹ标签已从文字(Positive/Negative)转为数值(1/0)")
    # 分层划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"训练集：{X_train.shape} | 测试集：{X_test.shape}")
    return X_train, X_test, y_train, y_test


# ===================== 3. 模型训练与优化（医疗场景参数） =====================
def train_lgbm_medical(X_train, X_test, y_train, y_test):
    """训练LightGBM（适配新版API，无参数错误）"""
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.08,
        "max_depth": 4,
        "min_split_gain": 0.02,
        "min_child_samples": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
        "class_weight": "balanced"
    }

    print("\n开始训练LightGBM（启用早停）")
    # 修正：早停仅通过 callbacks 传入，不写 early_stopping_rounds 参数
    model = lgb.train(
        params,
        train_set=lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_eval],
        callbacks=[
            lgb.log_evaluation(period=10),           # 每10轮打印一次日志
            lgb.early_stopping(stopping_rounds=15)    # 早停（新版API正确写法）
        ]
    )

    model_path = os.path.join(MODEL_DIR, "lightgbm_medical_best.pkl")
    joblib.dump(model, model_path)
    print(f"\n模型已保存至：{model_path}")
    return model


# ===================== 4. 医疗场景评估与可视化 =====================
def evaluate_medical_model(model, X_test, y_test):
    """医疗场景评估：重点输出召回率（漏诊率）、AUC"""
    # 预测概率与标签
    y_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.round(y_proba).astype(int)  # 默认阈值0.5

    # 核心指标
    metrics = {
        "准确率": accuracy_score(y_test, y_pred),
        "精确率": precision_score(y_test, y_pred),
        "召回率(防漏诊)": recall_score(y_test, y_pred),
        "F1分数": f1_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }

    # 打印评估结果（突出医疗重点）
    print("\n" + "=" * 60)
    print("医疗场景模型评估结果")
    print("=" * 60)
    for name, val in metrics.items():
        print(f"{name:12s}: {val:.4f}")
    print("=" * 60)

    # 保存指标到CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULT_DIR, "医疗场景评估指标.csv"),
                      index=False, encoding="utf-8-sig")
    print(f"评估指标已保存至 results/医疗场景评估指标.csv")
    return metrics, y_pred, y_proba


def plot_medical_visualizations(model, X_test, y_test, y_pred):
    """绘制医疗场景关键图：特征重要性、混淆矩阵"""
    # 1. 特征重要性（辅助医生判断风险因子）
    feat_importance = pd.Series(
        model.feature_importance(importance_type="gain"),  # 按增益计算重要性
        index=X_test.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    feat_importance.head(15).plot(kind="bar", color="#1f77b4")  # 显示Top15特征
    plt.title("LightGBM 特征重要性（按增益排序）", fontsize=14)
    plt.xlabel("特征名称", fontsize=12)
    plt.ylabel("特征增益（Gain）", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "01_特征重要性.png"), dpi=300)
    plt.close()
    print("特征重要性图已保存")

    # 2. 混淆矩阵（医疗场景：直观看漏诊/误诊）
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("混淆矩阵", fontsize=14)
    plt.colorbar()
    tick_labels = ["未患病(0)", "患病(1)"]
    plt.xticks([0, 1], tick_labels, fontsize=12)
    plt.yticks([0, 1], tick_labels, fontsize=12)

    # 标注数值（漏诊/误诊重点标注）
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=14)
            # 标注漏诊（真实患病但预测未患病）
            if i == 1 and j == 0:
                plt.text(j, i + 0.25, "漏诊", ha="center", va="center", color="red", fontweight="bold")
            # 标注误诊（真实未患病但预测患病）
            if i == 0 and j == 1:
                plt.text(j, i + 0.25, "误诊", ha="center", va="center", color="orange", fontweight="bold")

    plt.ylabel("真实标签", fontsize=12)
    plt.xlabel("预测标签", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "02_混淆矩阵.png"), dpi=300)
    plt.close()
    print("混淆矩阵图已保存")


# ===================== 5. 主流程（一键运行） =====================
def main():
    try:
        # 1. 加载最优数据
        df = load_best_data(DATA_PATHS)
        # 2. 划分数据
        X_train, X_test, y_train, y_test = split_data(df)
        # 3. 训练优化模型
        model = train_lgbm_medical(X_train, X_test, y_train, y_test)
        # 4. 医疗场景评估
        metrics, y_pred, y_proba = evaluate_medical_model(model, X_test, y_test)
        # 5. 可视化输出
        plot_medical_visualizations(model, X_test, y_test, y_pred)

        print("\n" + "=" * 60)
        print("全部流程完成！输出文件清单：")
        print(f"1. 模型文件：{os.path.join(MODEL_DIR, 'lightgbm_medical_best.pkl')}")
        print(f"2. 评估指标：{os.path.join(RESULT_DIR, '医疗场景评估指标.csv')}")
        print(f"3. 特征重要性：{os.path.join(RESULT_DIR, '01_特征重要性.png')}")
        print(f"4. 混淆矩阵：{os.path.join(RESULT_DIR, '02_混淆矩阵.png')}")
        print("=" * 60)

    except Exception as e:
        print(f"\n运行出错：{str(e)}")
        raise


if __name__ == "__main__":
    main()