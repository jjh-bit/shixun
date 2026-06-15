import pandas as pd
import numpy as np
import logging
import sys
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 工业级日志配置 (Logger Setup)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_pipeline.log', mode='a')
    ]
)
logger = logging.getLogger("DataPreprocessor")

# ==========================================
# 2. 核心处理模块 (保持原有逻辑不变)
# ==========================================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        logger.info(f"开始加载数据文件: {file_path}")
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("加载的数据集为空")
        logger.info(f"数据加载成功。当前数据维度: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"致命错误：未找到文件 '{file_path}'，请检查文件路径是否正确。")
        raise
    except Exception as e:
        logger.error(f"读取数据时发生未知错误: {str(e)}", exc_info=True)
        raise

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("开始进行特征二值化编码...")
    df_encoded = df.copy()
    try:
        gender_map = {'Male': 1, 'Female': 0}
        yes_no_map = {'Yes': 1, 'No': 0}
        class_map = {'Positive': 1, 'Negative': 0}
        
        if 'Gender' in df_encoded.columns:
            df_encoded['Gender'] = df_encoded['Gender'].map(gender_map)
        else:
            logger.warning("未检测到 'Gender' 列，跳过性别编码。")

        if 'class' in df_encoded.columns:
            df_encoded['class'] = df_encoded['class'].map(class_map)
        else:
            logger.warning("未检测到 'class' 目标列，可能当前为测试集数据。")

        yes_no_columns = [
            'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
            'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
            'Irritability', 'delayed healing', 'partial paresis', 
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        
        processed_count = 0
        for col in yes_no_columns:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map(yes_no_map)
                processed_count += 1
            else:
                logger.debug(f"特征 '{col}' 不存在，已跳过。")
                
        logger.info(f"特征编码完成，共处理了 {processed_count} 个二分类症状特征。")
        
        if df_encoded.isnull().any().sum() > 0:
            logger.warning("编码过程中产生缺失值(NaN)，可能存在未预期的脏数据分类标签！")
            
        return df_encoded
    except Exception as e:
        logger.error(f"特征编码过程中发生异常: {str(e)}", exc_info=True)
        raise

def scale_numeric_features(df: pd.DataFrame, num_cols: list = ['Age']) -> pd.DataFrame:
    logger.info(f"开始对数值特征 {num_cols} 进行量纲统一(StandardScaler)...")
    df_scaled = df.copy()
    try:
        scaler = StandardScaler()
        missing_cols = [col for col in num_cols if col not in df_scaled.columns]
        
        if missing_cols:
            raise KeyError(f"数据中缺失需要标准化的列: {missing_cols}")
            
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
        logger.info("标准化处理完成。")
        return df_scaled
    except Exception as e:
        logger.error(f"特征标准化过程中发生异常: {str(e)}", exc_info=True)
        raise

def export_data(df: pd.DataFrame, output_path: str):
    try:
        logger.info(f"准备导出处理后的数据至: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("数据导出成功！")
    except Exception as e:
        logger.error(f"数据导出失败: {str(e)}", exc_info=True)
        raise

# ==========================================
# 3. 流水线接口封装 (为 main.py 提供调用端点)
# ==========================================
def preprocess_data(input_file: str, export_csv: bool = False) -> pd.DataFrame:
    """
    数据预处理主接口
    
    参数:
        input_file: str, 原始CSV文件路径
        export_csv: bool, 是否在本地保存中间结果文件
        
    返回:
        final_df: pd.DataFrame, 清洗并编码完成的数据矩阵
    """
    logger.info("========== 数据预处理模块被调用 ==========")
    
    try:
        # Step 1: 加载数据
        raw_df = load_data(input_file)
        
        # Step 2: 文本特征二值化
        encoded_df = encode_categorical_features(raw_df)
        
        # Step 3: 数值特征标准化
        final_df = scale_numeric_features(encoded_df, num_cols=['Age'])
        
        # Step 4: 根据需求决定是否落盘保存中间结果
        if export_csv:
            output_file = 'cleaned_' + input_file.split('/')[-1]
            export_data(final_df, output_file)
            
        logger.info("========== 数据预处理模块执行完毕 ==========")
        
        # 核心修改：返回处理后的 DataFrame 给全局主程序
        return final_df
        
    except Exception as e:
        logger.critical("数据预处理流水线发生致命错误", exc_info=True)
        # 修改为向上层抛出异常，由 main.py 决定是否中断
        raise

# 本地测试入口
if __name__ == "__main__":
    # 仅当直接运行此脚本时执行，验证逻辑是否正确
    test_df = preprocess_data('糖尿病预测.csv', export_csv=True)
    print("返回的数据维度:", test_df.shape)