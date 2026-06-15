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

# 导入各模块
from src.data_preprocessing import preprocess_data
from src import B_特征工程, C_模型训练, evaluation
import src.data_preprocessing as dp_module

# 强制重载，确保每次运行都读取磁盘上的最新代码，避免缓存旧逻辑
importlib.reload(dp_module)
importlib.reload(B_特征工程)
importlib.reload(C_模型训练)
importlib.reload(evaluation)

# reload 后重新绑定函数引用
preprocess_data = dp_module.preprocess_data


def main():
    print("🚀 === 糖尿病预测项目自动化流水线启动 === 🚀\n")

    # 1. 基础目录初始化
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 2. 阶段 1：数据预处理 (成员 A)
    print(">>> 正在执行阶段 1: 数据清洗与标准化 (成员 A) ...")
    raw_data_path = os.path.join(DATA_DIR, "糖尿病预测.csv")
    if not os.path.exists(raw_data_path):
        print(f"❌ 找不到原始数据文件 {raw_data_path}，请确保它在 data 文件夹中！")
        return

    # 显式执行预处理，并接收返回的清洗后 DataFrame
    cleaned_df = preprocess_data(raw_data_path, export_csv=True)

    # A 成员默认输出在根目录，将其规范化移动到 data/
    legacy_output = os.path.join(BASE_DIR, "cleaned_糖尿病预测.csv")
    target_output = os.path.join(DATA_DIR, "cleaned_糖尿病预测.csv")
    if os.path.exists(legacy_output):
        shutil.move(legacy_output, target_output)
    print(f"✅ 阶段 1 完成！数据已清洗并存入 {target_output}，"
          f"维度: {cleaned_df.shape}\n")

    # 3. 阶段 2：特征工程 (成员 B)
    print(">>> 正在执行阶段 2: 特征工程 (成员 B) ...")
    B_特征工程.main()
    print("✅ 阶段 2 完成！\n")

    # 4. 阶段 3：模型训练 (成员 C)
    print(">>> 正在执行阶段 3: 模型训练 (成员 C) ...")
    try:
        C_模型训练.main()
    except Exception as e:
        print(f"❌ 阶段 3 失败: {e}")
        print("⚠️ 流水线中断，跳过后续阶段。")
        return
    print("✅ 阶段 3 完成！\n")

    # 5. 阶段 4：模型评估 (成员 D)
    print(">>> 正在执行阶段 4: 模型评估 (成员 D) ...")
    evaluation.main()

    print("✅ 阶段 4 完成！所有产出物已输出至 results/ 目录。\n")
    print(" === 全部成功执行完毕！ === ")

if __name__ == "__main__":
    main()