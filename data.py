import pandas as pd
import numpy as np
import logging
import sys
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 工业级日志配置 (Logger Setup)
# ==========================================
# 配置日志输出格式，同时输出到控制台和日志文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),                     # 控制台输出
        logging.FileHandler('data_pipeline.log', mode='a')     # 追加模式写入日志文件
    ]
)
logger = logging.getLogger("DataPreprocessor")

# ==========================================
# 2. 核心处理模块
# ==========================================
def load_data(file_path: str) -> pd.DataFrame:
    """加载原始数据并进行基础校验"""
    try:
        logger.info(f"开始加载数据文件: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("加载的数据集为空")
            
        logger.info(f"数据加载成功。当前数据维度: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"致命错误：未找到文件 '{file_path}'，请检查文件路径是否正确。")
        raise # 抛出异常，中断流水线
    except Exception as e:
        logger.error(f"读取数据时发生未知错误: {str(e)}", exc_info=True)
        raise

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """对二分类文本特征进行数值化编码"""
    logger.info("开始进行特征二值化编码...")
    df_encoded = df.copy()
    
    try:
        # 1. 核心特征映射字典
        gender_map = {'Male': 1, 'Female': 0}
        yes_no_map = {'Yes': 1, 'No': 0}
        class_map = {'Positive': 1, 'Negative': 0}
        
        # 2. 检查关键列是否存在
        if 'Gender' in df_encoded.columns:
            df_encoded['Gender'] = df_encoded['Gender'].map(gender_map)
        else:
            logger.warning("未检测到 'Gender' 列，跳过性别编码。")

        if 'class' in df_encoded.columns:
            df_encoded['class'] = df_encoded['class'].map(class_map)
        else:
            logger.warning("未检测到 'class' 目标列，可能当前为测试集数据。")

        # 3. 批量处理症状特征
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
        
        # 4. 数据一致性校验：检查是否存在映射失败产生的 NaN
        if df_encoded.isnull().any().sum() > 0:
            logger.warning("编码过程中产生缺失值(NaN)，可能存在未预期的脏数据分类标签！")
            
        return df_encoded
        
    except Exception as e:
        logger.error(f"特征编码过程中发生异常: {str(e)}", exc_info=True)
        raise

def scale_numeric_features(df: pd.DataFrame, num_cols: list = ['Age']) -> pd.DataFrame:
    """对连续型数值特征进行标准化处理"""
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
        
    except KeyError as ke:
        logger.error(f"列名错误: {str(ke)}")
        raise
    except ValueError as ve:
        logger.error(f"数据格式错误，无法进行标准化: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"特征标准化过程中发生异常: {str(e)}", exc_info=True)
        raise

def export_data(df: pd.DataFrame, output_path: str):
    """导出清洗完毕的数据"""
    try:
        logger.info(f"准备导出处理后的数据至: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("数据导出成功！数据预处理流水线正常结束。")
    except PermissionError:
        logger.error(f"权限被拒绝：无法写入文件 {output_path}。文件可能正被其他程序占用。")
        raise
    except Exception as e:
        logger.error(f"数据导出失败: {str(e)}", exc_info=True)
        raise

# ==========================================
# 3. 流水线执行入口 (Pipeline Execution)
# ==========================================
def main():
    logger.info("========== 糖尿病预测: 数据预处理任务启动 ==========")
    input_file = '糖尿病预测.csv'
    output_file = 'cleaned_糖尿病预测.csv'
    
    try:
        # Step 1: 加载数据
        raw_df = load_data(input_file)
        
        # Step 2: 文本特征二值化
        encoded_df = encode_categorical_features(raw_df)
        
        # Step 3: 数值特征标准化
        final_df = scale_numeric_features(encoded_df, num_cols=['Age'])
        
        # Step 4: 导出可用数据
        export_data(final_df, output_file)
        
    except Exception as e:
        logger.critical("流水线致命错误，程序非正常退出！请检查日志文件。", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()