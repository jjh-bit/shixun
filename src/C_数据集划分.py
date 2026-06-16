# src/C_数据集划分.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_diabetes_data(raw_data_path, save_dir, save_prefix, label_col=-1):
    """
    糖尿病特征数据集划分函数
    :param raw_data_path: 输入特征工程数据集路径
    :param save_dir: 输出文件保存目录
    :param save_prefix: 输出文件前缀（区分全量特征/筛选后特征）
    :param label_col: 标签列索引（默认最后一列，若标签列不是最后一列需手动指定）
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载数据
    print(f"正在加载数据集：{raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"数据集形状：{df.shape} | 列名：{list(df.columns)}")

    # 2. 基础预处理：缺失值填充（医疗数据常用策略）
    # 替换原来的 if/else 里的 fillna 代码
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    print("缺失值处理完成")

    # 3. 分离特征X和标签y
    X = df.iloc[:, df.columns != df.columns[label_col]]
    y = df.iloc[:, label_col]
    print(f"特征矩阵形状：{X.shape} | 标签向量形状：{y.shape}")

    # 4. 分层抽样划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 5. 拼接特征与标签，保存为CSV
    train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([X_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # 输出路径
    train_path = os.path.join(save_dir, f"{save_prefix}_train_70.csv")
    val_path = os.path.join(save_dir, f"{save_prefix}_val_15.csv")
    test_path = os.path.join(save_dir, f"{save_prefix}_test_15.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"\n{save_prefix}数据集划分完成！")
    print(f"训练集（70%）：{train_df.shape} | 保存路径：{train_path}")
    print(f"验证集（15%）：{val_df.shape} | 保存路径：{val_path}")
    print(f"测试集（15%）：{test_df.shape} | 保存路径：{test_path}")
    print(f"标签分布（训练集）：0={sum(y_train==0)}, 1={sum(y_train==1)}")
    print("-" * 50)

def main():
    # 路径配置（自动适配项目结构）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    # 划分筛选后特征数据集
    split_diabetes_data(
        raw_data_path=os.path.join(DATA_DIR, "特征工程输出_给C成员.csv"),
        save_dir=DATA_DIR,
        save_prefix="特征工程输出"
    )

if __name__ == "__main__":
    main()