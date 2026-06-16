"""
============================================================
 糖尿病风险预测脚本
 支持：单条数据预测 / 批量 CSV 预测 / 测试集评估
 用法：python predict.py                          （默认测试集评估）
       python predict.py --input data/new_patients.csv
       python predict.py --model lightgbm           （指定模型）
       python predict.py --list                     （列出可用模型）
============================================================
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

# ==========================================
# 0. 路径定位（基于本文件位置，不依赖 CWD）
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================================
# 1. 可用模型注册表
# ==========================================
AVAILABLE_MODELS = {
    "logistic": {
        "file": "logistic_regression.pkl",
        "name": "Logistic Regression",
        "need_scaler": True,
    },
    "random_forest": {
        "file": "random_forest.pkl",
        "name": "Random Forest",
        "need_scaler": False,
    },
    "xgboost": {
        "file": "xgboost.pkl",
        "name": "XGBoost",
        "need_scaler": False,
    },
    "lightgbm": {
        "file": "lightgbm.pkl",
        "name": "LightGBM",
        "need_scaler": False,
    },
    "mlp": {
        "file": "mlp.pkl",
        "name": "MLP (深度学习)",
        "need_scaler": True,
    },
}

# 模型中文描述
RISK_DESCRIPTION = {
    (0.0, 0.2): "🟢 低风险 — 目前指标未显示明显糖尿病风险，建议保持健康生活方式。",
    (0.2, 0.5): "🟡 中等风险 — 部分指标存在异常趋势，建议定期复查血糖。",
    (0.5, 0.7): "🟠 较高风险 — 多项指标指向糖尿病风险，建议尽快就医检查。",
    (0.7, 1.0): "🔴 高风险 — 指标高度提示糖尿病可能，请立即就医进行专业诊断。",
}


def list_models():
    """列出所有可用模型及其状态"""
    print("\n可用的预测模型：")
    print("-" * 60)
    for key, info in AVAILABLE_MODELS.items():
        path = os.path.join(MODEL_DIR, info["file"])
        status = "✅ 已训练" if os.path.exists(path) else "❌ 未找到"
        print(f"  {key:15s} → {info['name']:25s} {status}")
    print("-" * 60)
    print("提示：如模型未找到，请先运行 C_模型训练.py 或 main.py\n")


def load_model(model_key="lightgbm"):
    """加载指定模型和标准化器，返回 (model, scaler, model_info)"""
    if model_key not in AVAILABLE_MODELS:
        raise KeyError(f"未知模型 '{model_key}'，可选: {list(AVAILABLE_MODELS.keys())}")

    info = AVAILABLE_MODELS[model_key]
    model_path = os.path.join(MODEL_DIR, info["file"])

    if not os.path.exists(model_path):
        # 尝试找第一个可用的模型
        for alt_key, alt_info in AVAILABLE_MODELS.items():
            alt_path = os.path.join(MODEL_DIR, alt_info["file"])
            if os.path.exists(alt_path):
                print(f"⚠️  模型 '{info['file']}' 不存在，自动切换到 '{alt_info['name']}'")
                model_key = alt_key
                info = alt_info
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"未找到任何模型文件！请先运行 main.py 训练模型。\n"
                f"期望路径: {MODEL_DIR}/"
            )

    model = joblib.load(model_path)
    print(f"✅ 已加载模型: {info['name']}  ({os.path.basename(model_path)})")

    # 加载标准化器（仅需要标准化的模型使用）
    scaler = None
    if info["need_scaler"]:
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✅ 已加载标准化器: scaler.pkl")
        else:
            print(f"⚠️  该模型需要标准化但未找到 scaler.pkl，将使用原始数值预测")

    return model, scaler, info


def predict_single(model, scaler, info, data_dict):
    """
    单条预测：传入特征字典，返回预测结果

    参数:
        data_dict: dict, 如 {'Age': 45, 'Polyuria': 1, 'Gender': 1, ...}
    返回:
        dict: 含概率、标签、风险描述
    """
    df = pd.DataFrame([data_dict])
    X = df.values

    if scaler is not None:
        X = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probability = proba[1] if proba.shape[0] > 1 else proba[0]
    else:
        probability = float(model.predict(X)[0])

    pred_label = 1 if probability >= 0.5 else 0

    # 匹配风险描述
    description = "⚪ 无法判断"
    for (lo, hi), desc in RISK_DESCRIPTION.items():
        if lo <= probability < hi:
            description = desc
            break

    return {
        "患病概率": round(probability, 4),
        "预测结果": "阳性 (有糖尿病风险)" if pred_label == 1 else "阴性 (暂无糖尿病风险)",
        "风险等级": description,
    }


def predict_dataframe(model, scaler, info, df, has_label=False):
    """
    批量预测 DataFrame

    参数:
        df: pd.DataFrame, 特征列（如有 class 列会自动排除）
        has_label: bool, 是否包含真实标签
    返回:
        pd.DataFrame: 原始数据 + 预测结果
    """
    # 排除标签列（如存在）
    X = df.drop(columns=["class"], errors="ignore")

    # 保存列顺序供后续使用
    feature_names = X.columns.tolist()

    X_arr = X.values
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    # 获取概率
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_arr)
        probabilities = proba[:, 1] if proba.ndim > 1 and proba.shape[1] > 1 else proba.flatten()
    else:
        probabilities = model.predict(X_arr).flatten().astype(float)

    pred_labels = (probabilities >= 0.5).astype(int)

    # 构建结果 DataFrame
    result = X.copy()
    result["患病概率"] = np.round(probabilities, 4)
    result["预测标签"] = pred_labels
    result["预测结论"] = [
        "阳性 (有糖尿病风险)" if p == 1 else "阴性 (暂无糖尿病风险)"
        for p in pred_labels
    ]

    # 风险等级
    risk_levels = []
    for prob in probabilities:
        assigned = False
        for (lo, hi), desc in RISK_DESCRIPTION.items():
            if lo <= prob < hi:
                risk_levels.append(desc)
                assigned = True
                break
        if not assigned:
            risk_levels.append("⚪ 无法判断")
    result["风险描述"] = risk_levels

    # 如有真实标签，追加对比列
    if has_label and "class" in df.columns:
        result["真实标签"] = df["class"].values.astype(int)
        result["预测正确"] = result["预测标签"] == result["真实标签"]
        result["预测正确"] = result["预测正确"].map({True: "✅", False: "❌"})

    return result


def main():
    parser = argparse.ArgumentParser(
        description="糖尿病风险预测 — 判断患者是否存在糖尿病风险"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="待预测的 CSV 文件路径（默认使用测试集 data/特征工程输出_test_15.csv）",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="lightgbm",
        choices=list(AVAILABLE_MODELS.keys()),
        help="选择预测模型（默认: lightgbm）",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用模型及状态",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="预测结果输出路径（默认保存到 results/预测结果.csv）",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=None,
        help="随机抽取 N 条数据展示（默认展示全部）",
    )

    args = parser.parse_args()

    # --list 列出模型
    if args.list:
        list_models()
        return

    print("\n" + "=" * 60)
    print("  🏥  糖尿病风险预测系统")
    print("=" * 60)

    # 加载模型
    list_models()
    model, scaler, info = load_model(args.model)

    # 确定输入数据
    if args.input is not None:
        input_path = args.input
    else:
        input_path = os.path.join(DATA_DIR, "特征工程输出_test_15.csv")
        if not os.path.exists(input_path):
            # 回退到给 C 成员的完整特征文件
            input_path = os.path.join(DATA_DIR, "特征工程输出_给C成员.csv")
            if not os.path.exists(input_path):
                print(f"❌ 找不到测试数据文件！请通过 --input 指定 CSV 路径。")
                return

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return

    print(f"\n📂 输入数据: {os.path.basename(input_path)}")
    df = pd.read_csv(input_path)
    print(f"   数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")

    has_label = "class" in df.columns
    if args.input is None and has_label:
        print(f"   包含真实标签列 (class)，将对比预测准确度\n")

    # 执行预测
    print(f"\n🔮 正在使用 {info['name']} 进行预测...")
    result = predict_dataframe(model, scaler, info, df, has_label=has_label)

    # 统计摘要
    print("\n" + "-" * 60)
    print("📊 预测结果统计")
    print("-" * 60)
    pred_counts = result["预测标签"].value_counts()
    print(f"   预测为阳性 (有风险): {pred_counts.get(1, 0)} 人")
    print(f"   预测为阴性 (无风险): {pred_counts.get(0, 0)} 人")
    print(f"   整体阳性率: {pred_counts.get(1, 0) / len(result) * 100:.1f}%")

    if has_label:
        correct = result["预测正确"].eq("✅").sum() if "预测正确" in result.columns else 0
        print(f"   预测正确数: {correct} / {len(result)}  ({correct / len(result) * 100:.1f}%)")

    # 风险分层统计
    print(f"\n📈 风险分层分布:")
    for (lo, hi), desc in RISK_DESCRIPTION.items():
        count = ((result["患病概率"] >= lo) & (result["患病概率"] < hi)).sum()
        bar = "█" * (count // max(1, len(result) // 30))
        print(f"   {desc.split('—')[0].strip():30s} {count:>4d} 人  {bar}")

    # 保存结果
    output_path = args.output or os.path.join(RESULT_DIR, "预测结果.csv")
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 预测结果已保存: {output_path}")

    # 展示样本
    display_cols = ["患病概率", "预测结论"]
    if has_label:
        display_cols = ["患病概率", "预测结论", "预测正确"]
    if "风险描述" in result.columns:
        display_cols.append("风险描述")

    sample_n = args.sample or min(10, len(result))
    if sample_n > 0:
        print(f"\n📋 前 {sample_n} 条预测结果预览:")
        print("-" * 80)
        for i, (_, row) in enumerate(result.head(sample_n).iterrows()):
            prob = row["患病概率"]
            label = "阳性" if row["预测标签"] == 1 else "阴性"
            verdict = row.get("预测正确", "")
            risk_brief = str(row.get("风险描述", "")).split("—")[0].strip()
            print(f"  [{i+1:>3d}] 概率={prob:.4f} | {label:4s} | {risk_brief:20s} {verdict}")

    print("\n" + "=" * 60)
    print("  ✅ 预测完成")
    print("=" * 60)
    print("\n⚠️  免责声明：本预测仅供辅助参考，不能替代专业医疗诊断。如有疑虑请及时就医。")


if __name__ == "__main__":
    main()
