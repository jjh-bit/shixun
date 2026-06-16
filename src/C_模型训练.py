# src/C_模型训练.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
# 传统机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# 树模型
import xgboost as xgb
import lightgbm as lgb
# 深度学习MLP
from sklearn.neural_network import MLPClassifier


def get_project_paths():
    """自动获取项目各目录路径（适配任意环境）"""
    # 当前脚本路径（src/C_模型训练.py）
    current_script_path = os.path.abspath(__file__)
    # 项目根目录（shixun）
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    # 数据目录（shixun/data）
    data_dir = os.path.join(project_root, "data")
    # 模型保存目录（shixun/models）
    model_save_dir = os.path.join(project_root, "models")

    # 确保目录存在（首次运行自动创建）
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    return data_dir, model_save_dir


def load_train_val_data(data_dir, data_prefix="selected_features"):
    """加载训练集 + 验证集，合并后用于模型训练（测试集保持独立，留给 D 评估）"""
    train_path = os.path.join(data_dir, f"{data_prefix}_train_70.csv")
    val_path = os.path.join(data_dir, f"{data_prefix}_val_15.csv")

    # 检查文件是否存在
    for p, label in [(train_path, "训练集"), (val_path, "验证集")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{label}文件缺失：{p}\n请先运行 C_数据集划分.py 生成文件")

    # 加载训练集和验证集
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 合并训练集 + 验证集 → 模型用 85% 的数据训练
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    # 分离特征 X 和标签 y（标签列在最后一列）
    X_train = combined_df.iloc[:, :-1]
    y_train = combined_df.iloc[:, -1]

    # 输出合并后的数据集信息
    print("=" * 60)
    print(f"✅ 成功加载 {data_prefix} 训练集 + 验证集（合并用于训练）")
    print(f"  训练集: {train_df.shape[0]} 行  +  验证集: {val_df.shape[0]} 行")
    print(f"  合并后: {combined_df.shape[0]} 行（85% 数据用于训练）")
    print(f"  特征数={X_train.shape[1]}, 患病样本数={int(sum(y_train == 1))}")
    print(f"  注：测试集（15%）保持独立，由成员 D 用于最终评估")
    print("=" * 60)

    return X_train, y_train


def train(X_train, y_train, model_save_dir):
    """仅用训练集训练5类模型并保存"""
    # 1. 数据标准化（仅逻辑回归、MLP需要，树模型无需标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 仅用训练集拟合标准化器
    # 保存标准化器（后续用测试集预测时需复用）
    scaler_path = os.path.join(model_save_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\n📊 标准化器已保存（基于训练集拟合）：{scaler_path}")

    # 2. 初始化模型字典（统一管理）
    model_dict = {}

    # 3. 训练基线模型：逻辑回归（仅用训练集）
    print("\n1. 🚀 训练基线模型：Logistic Regression（仅训练集）")
    lr = LogisticRegression(
        max_iter=2000,  # 确保在训练集上收敛
        class_weight="balanced",  # 处理医疗数据类别失衡
        random_state=42  # 固定随机种子，结果可复现
    )
    lr.fit(X_train_scaled, y_train)  # 仅传入训练集数据
    model_dict["logistic_regression"] = lr

    # 4. 训练核心树模型1：随机森林（仅用训练集）
    print("2. 🚀 训练核心模型：Random Forest（仅训练集）")
    rf = RandomForestClassifier(
        n_estimators=150,  # 平衡训练集拟合效果与效率
        max_depth=10,  # 限制树深，避免训练集过拟合
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)  # 树模型用原始训练集特征，无需标准化
    model_dict["random_forest"] = rf

    # 5. 训练核心树模型2：XGBoost（仅用训练集，无废弃参数）
    print("3. 🚀 训练核心模型：XGBoost（仅训练集）")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,  # 低学习率适配训练集单数据来源
        scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),  # 基于训练集计算类别权重
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)  # 仅用训练集训练
    model_dict["xgboost"] = xgb_model

    # 6. 训练核心树模型3：LightGBM（仅用训练集，无警告）
    print("4. 🚀 训练核心模型：LightGBM（仅训练集）")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,  # 降低复杂度，适配训练集单数据训练
        learning_rate=0.03,
        class_weight="balanced",
        random_state=42,
        verbose=-1,  # 关闭冗余日志
        reg_alpha=0.1,  # 正则化避免训练集过拟合
        reg_lambda=0.1
    )
    lgb_model.fit(X_train, y_train)  # 仅用训练集训练
    model_dict["lightgbm"] = lgb_model

    # 7. 训练深度学习模型：MLP（仅用训练集）
    print("5. 🚀 训练深度学习模型：MLP（仅训练集）")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=1000,  # 足够迭代次数确保在训练集上收敛
        batch_size=32,  # 小批量适配训练集样本量
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)  # 仅用标准化后的训练集
    model_dict["mlp"] = mlp

    # 8. 保存所有仅用训练集训练的模型
    for model_name, model in model_dict.items():
        model_path = os.path.join(model_save_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ 仅用训练集训练的模型已保存：{model_path}")

    # 9. 输出训练完成总结
    print("\n" + "=" * 60)
    print("🎉 所有模型训练完成（仅基于训练集）！")
    print("训练模型列表：", list(model_dict.keys()))
    print(f"📁 模型及标准化器已保存至：{model_save_dir}")
    print("注：所有模型未使用验证集，仅通过训练集完成参数学习")
    print("=" * 60)

    return model_dict

def main():
    # 1. 获取项目路径（自动适配）
    data_dir, model_save_dir = get_project_paths()

    # 2. 加载训练集 + 验证集，合并后用于训练（85% 数据）
    X_train, y_train = load_train_val_data(
        data_dir=data_dir,
        data_prefix="特征工程输出"  # 可切换为 "full_features" 使用全量特征
    )

    # 3. 仅用训练集训练模型
    trained_models = train(
        X_train=X_train,
        y_train=y_train,
        model_save_dir=model_save_dir
    )

# 主执行逻辑（直接点击运行时触发）
if __name__ == "__main__":
    main()