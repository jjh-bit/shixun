"""
===============================================================================
 C 成员使用 B 成员特征工程结果的示例脚本
-------------------------------------------------------------------------------
 用途：
   1. 直接读取 data/特征工程输出_给C成员.csv
   2. 完成训练集/测试集分层划分
   3. 跑通 Logistic Regression 和 Random Forest 两个基线模型

 说明：
   本文件是协作示例，不替代 C 成员正式的模型训练与调参脚本。
===============================================================================
"""

import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RANDOM_STATE = 42


def evaluate_model(name, model, X_test, y_test):
    """输出医疗分类任务常用指标，重点关注 Recall 和 AUC。"""
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall:   {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_score):.4f}")


def main():
    data_path = os.path.join(DATA_DIR, '特征工程输出_给C成员.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')

    X = df.drop(columns=['class'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("=" * 60)
    print("  C 成员读取 B 特征示例")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"特征数量: {X.shape[1]}")
    print(f"训练集: {X_train.shape[0]} 行, 测试集: {X_test.shape[0]} 行")

    log_reg = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    log_reg.fit(X_train, y_train)
    evaluate_model('Logistic Regression 基线', log_reg, X_test, y_test)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    evaluate_model('Random Forest 基线', rf, X_test, y_test)


if __name__ == '__main__':
    main()
