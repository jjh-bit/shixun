import os
import sys
import shutil
import importlib

# ==========================================
# 0. 项目根目录定位（基于本文件位置，不依赖 CWD）
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 将项目根目录加入 sys.path，确保模块导入不受 CWD 影响
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ==========================================
# 1. 导入各模块（统一使用 src.xxx 形式）
# ==========================================
import src.A_data_preprocessing as dp_module
import src.B_特征工程 as B_特征工程
import src.C_数据集划分 as C_数据集划分
import src.C_模型训练 as C_模型训练
import src.D_evaluation as D_evaluation

# 强制重载，确保每次运行都读取磁盘上的最新代码
importlib.reload(dp_module)
importlib.reload(B_特征工程)
importlib.reload(C_数据集划分)
importlib.reload(C_模型训练)
importlib.reload(D_evaluation)

# reload 后重新绑定函数引用
preprocess_data = dp_module.preprocess_data


def main():
    print("🚀 === 糖尿病预测项目自动化流水线启动 === 🚀\n")

    # 基础目录初始化
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ==========================================
    # 阶段 A：数据预处理 (成员 A)
    # ==========================================
    print(">>> 正在执行阶段 A: 数据清洗与标准化 ...")
    raw_data_path = os.path.join(DATA_DIR, "糖尿病预测.csv")
    if not os.path.exists(raw_data_path):
        print(f"❌ 找不到原始数据文件 {raw_data_path}，请确保它在 data 文件夹中！")
        return

    cleaned_df = preprocess_data(raw_data_path, export_csv=True)

    # A 成员默认输出在 CWD，将其规范化移动到 data/
    legacy_output = os.path.join(BASE_DIR, "cleaned_糖尿病预测.csv")
    target_output = os.path.join(DATA_DIR, "cleaned_糖尿病预测.csv")
    if os.path.exists(legacy_output):
        shutil.move(legacy_output, target_output)
    print(f"✅ 阶段 A 完成！数据已清洗并存入 {target_output}，"
          f"维度: {cleaned_df.shape}\n")

    # ==========================================
    # 阶段 B：特征工程 (成员 B)
    # ==========================================
    print(">>> 正在执行阶段 B: 特征工程 ...")
    B_特征工程.main()
    print("✅ 阶段 B 完成！\n")

    # ==========================================
    # 阶段 C：数据集划分 + 模型训练 (成员 C)
    # ==========================================
    print(">>> 正在执行阶段 C-1: 数据集划分 ...")
    try:
        C_数据集划分.main()
    except Exception as e:
        print(f"❌ 数据集划分失败: {e}")
        return
    print("✅ 数据集划分完成！\n")

    print(">>> 正在执行阶段 C-2: 模型训练 ...")
    try:
        C_模型训练.main()
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    print("✅ 阶段 C 完成！\n")

    # ==========================================
    # 阶段 D：模型评估 (成员 D)
    # ==========================================
    print(">>> 正在执行阶段 D: 模型评估 ...")
    try:
        D_evaluation.main()
    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        return
    print("✅ 阶段 D 完成！所有产出物已输出至 results/ 目录。\n")

    print("🎉 === 全自动流水线全部成功执行完毕！ === 🎉")


if __name__ == "__main__":
    main()
