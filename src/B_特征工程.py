"""
===============================================================================
 成员 B：特征研发工程师
 职责：消除冗余特征与噪声，提取对糖尿病预测最具区分度的核心指标
===============================================================================
 技术要求 2 对应内容：
   (1) 特征构建 -- 根据医学常识交叉组合新特征
   (2) 特征降维与选择 -- 过滤法/嵌入法剔除噪声特征
   (3) 相关性分析 -- 检查多重共线性
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 弹窗
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
SRC_DIR = os.path.join(BASE_DIR, 'src')
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

FEATURE_DESCRIPTIONS = {
    'Age': '年龄，原始数值特征',
    'Gender': '性别编码，Male=1，Female=0',
    'Polyuria': '多尿症状编码，Yes=1，No=0',
    'Polydipsia': '烦渴症状编码，Yes=1，No=0',
    'sudden weight loss': '突然体重下降编码，Yes=1，No=0',
    'weakness': '乏力编码，Yes=1，No=0',
    'Polyphagia': '多食症状编码，Yes=1，No=0',
    'Genital thrush': '生殖器念珠菌感染编码，Yes=1，No=0',
    'visual blurring': '视力模糊编码，Yes=1，No=0',
    'Itching': '瘙痒编码，Yes=1，No=0',
    'Irritability': '易怒编码，Yes=1，No=0',
    'delayed healing': '伤口愈合延迟编码，Yes=1，No=0',
    'partial paresis': '局部麻痹编码，Yes=1，No=0',
    'muscle stiffness': '肌肉僵硬编码，Yes=1，No=0',
    'Alopecia': '脱发编码，Yes=1，No=0',
    'Obesity': '肥胖编码，Yes=1，No=0',
    'Age_<30': '年龄小于 30 岁分组',
    'Age_30-40': '年龄 30-40 岁分组',
    'Age_40-50': '年龄 40-50 岁分组',
    'Age_50-60': '年龄 50-60 岁分组',
    'Age_60+': '年龄 60 岁以上分组',
    'Triad_Score': '典型三多症状评分，多尿+烦渴+多食，取值 0-3',
    'Metabolic_Score': '代谢相关症状评分，肥胖+乏力+视力模糊+愈合延迟，取值 0-4',
    'Neural_Score': '神经相关症状评分，局部麻痹+肌肉僵硬+脱发，取值 0-3',
    'Age_obesity_interaction': '年龄与肥胖交互项，用于表达年龄增长叠加肥胖风险',
    'Age_polyuria_interaction': '年龄与多尿交互项，用于表达年龄增长叠加典型症状风险',
    'Total_Symptoms': '14 个症状的总数，取值 0-14',
    'class': '目标标签，Positive=1，Negative=0'
}

def save_csv(df, path):
    """统一保存 CSV，避免中文表头在 Excel 中显示乱码。"""
    df.to_csv(path, index=False, encoding='utf-8-sig')

def normalize_input_columns(df):
    """兼容 A 成员清洗后可能出现的列名细微差异。"""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    rename_map = {
        'delayedhealing': 'delayed healing',
        'DelayedHealing': 'delayed healing',
        'Delayed healing': 'delayed healing',
        'Sudden weight loss': 'sudden weight loss',
        'Visual blurring': 'visual blurring',
        'Partial paresis': 'partial paresis',
        'Muscle stiffness': 'muscle stiffness',
    }
    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    return df

def encode_binary_column(series, mapping, col_name):
    """兼容原始字符型数据和 A 清洗后的 0/1 数值型数据。"""
    if pd.api.types.is_numeric_dtype(series):
        encoded = pd.to_numeric(series, errors='coerce')
    else:
        normalized = series.astype(str).str.strip()
        encoded = normalized.map(mapping)
        numeric_mask = encoded.isna()
        encoded.loc[numeric_mask] = pd.to_numeric(
            normalized.loc[numeric_mask], errors='coerce')

    allowed_values = {0, 1, 0.0, 1.0}
    invalid_values = sorted(set(encoded.dropna().unique()) - allowed_values)
    if invalid_values:
        raise ValueError(f"{col_name} 存在非 0/1 取值: {invalid_values}")
    return encoded.astype('Int64')

# ============================================================================
# 主控函数 (必须要有这个，否则一导入就执行)
# ============================================================================
def main():
    # ============================================================================
    # 1. 加载数据
    # ============================================================================
    print("=" * 60)
    print("  成员 B：特征工程模块")
    print("=" * 60)

    input_file = os.path.join(DATA_DIR, 'cleaned_糖尿病预测.csv')
    if not os.path.exists(input_file):
        input_file = os.path.join(DATA_DIR, '糖尿病预测.csv')

    df_raw = normalize_input_columns(pd.read_csv(input_file))
    print(f"  输入文件: {input_file}")

    print(f"\n[OK] 数据加载成功: {df_raw.shape[0]} 行 x {df_raw.shape[1]} 列\n")

    # ============================================================================
    # 2. EDA 与 相关性分析（技术要求 2）
    # ============================================================================
    print("-" * 60)
    print("  阶段 1/4：EDA 与相关性分析")
    print("-" * 60)

    target = 'class'
    numerical_features = ['Age']
    binary_features = [col for col in df_raw.columns
                       if col not in numerical_features + [target]]

    print(f"  数值特征: {numerical_features}")
    print(f"  二分类特征: {binary_features}")
    print(f"  目标变量: {target}")

    df_encoded = df_raw.copy()
    gender_map = {'Male': 1, 'Female': 0, 'male': 1, 'female': 0,
                  'M': 1, 'F': 0, '男': 1, '女': 0}
    yes_no_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
                  'Y': 1, 'N': 0, '是': 1, '否': 0, '有': 1, '无': 0}
    target_map = {'Positive': 1, 'Negative': 0, 'positive': 1, 'negative': 0,
                  '患病': 1, '未患病': 0}

    df_encoded['Gender'] = encode_binary_column(df_encoded['Gender'], gender_map, 'Gender')
    for col in binary_features:
        if col != 'Gender':
            df_encoded[col] = encode_binary_column(df_encoded[col], yes_no_map, col)

    df_encoded['class'] = encode_binary_column(df_encoded['class'], target_map, 'class')

    if df_encoded.isnull().any().any():
        missing_cols = df_encoded.columns[df_encoded.isnull().any()].tolist()
        raise ValueError(f"编码后存在缺失值，请检查字段取值: {missing_cols}")

    df_encoded = df_encoded.astype(float)

    print("\n[数据] 目标变量分布:")
    print(f"   正样本 (Positive): {df_encoded['class'].sum()}  ({df_encoded['class'].mean()*100:.1f}%)")
    print(f"   负样本 (Negative): {(1-df_encoded['class']).sum()}  ({(1-df_encoded['class'].mean())*100:.1f}%)")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    corr_matrix = df_encoded.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    ax.set_title('糖尿病风险特征相关性热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '01_相关性热力图.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] 相关性热力图 -> results/01_相关性热力图.png")

    target_corr = corr_matrix['class'].drop('class').sort_values(ascending=False)
    print("\n[分析] 各特征与糖尿病风险的相关性（从高到低）:")
    for feat, corr_val in target_corr.items():
        bar = '#' * int(abs(corr_val) * 30)
        print(f"   {feat:25s}  {corr_val:+.4f}  {bar}")

    from sklearn.linear_model import LinearRegression

    def calculate_vif(df, features):
        vif_data = pd.DataFrame()
        vif_data['特征'] = features
        vif_data['VIF'] = 0.0
        for i, feature in enumerate(features):
            other_features = [f for f in features if f != feature]
            if len(other_features) == 0:
                vif_data.loc[i, 'VIF'] = 1.0
                continue
            model = LinearRegression()
            model.fit(df[other_features], df[feature])
            r2 = model.score(df[other_features], df[feature])
            vif_data.loc[i, 'VIF'] = 1 / (1 - r2) if r2 < 1 else float('inf')
        return vif_data

    feature_cols = [c for c in df_encoded.columns if c != 'class']
    vif_result = calculate_vif(df_encoded, feature_cols)
    print("\n[分析] 多重共线性检查 (VIF):")
    print(f"   {'特征':25s} {'VIF':>8s} {'风险':>6s}")
    print(f"   {'-'*41}")
    for _, row in vif_result.iterrows():
        risk = '**高**' if row['VIF'] > 10 else ('*中*' if row['VIF'] > 5 else '低')
        print(f"   {row['特征']:25s} {row['VIF']:>8.2f} {risk:>6s}")

    # ============================================================================
    # 3. 特征构建（技术要求 1）
    # ============================================================================
    print("\n" + "-" * 60)
    print("  阶段 2/4：特征构建")
    print("-" * 60)

    df_feat = df_encoded.copy()

    if df_feat['Age'].min() < 0 or df_feat['Age'].max() <= 5:
        df_feat['Age_Group'] = pd.qcut(
            df_feat['Age'],
            q=5,
            labels=['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'],
            duplicates='drop'
        )
        print("   [OK] 年龄分位数组特征 (Age_Q1_low ~ Age_Q5_high，适配标准化 Age)")
    else:
        df_feat['Age_Group'] = pd.cut(df_feat['Age'],
                                      bins=[0, 30, 40, 50, 60, 100],
                                      labels=['<30', '30-40', '40-50', '50-60', '60+'])
        print("   [OK] 年龄分组特征 (Age_<30, Age_30-40, Age_40-50, Age_50-60, Age_60+)")

    age_dummies = pd.get_dummies(df_feat['Age_Group'], prefix='Age')
    df_feat = pd.concat([df_feat, age_dummies.astype(int)], axis=1)

    df_feat['Triad_Score'] = (df_feat['Polyuria'] +
                              df_feat['Polydipsia'] +
                              df_feat['Polyphagia'])
    print("   [OK] 三多症状评分 (Triad_Score): 多尿+烦渴+多食 [0-3]")

    df_feat['Metabolic_Score'] = (df_feat['Obesity'] +
                                  df_feat['weakness'] +
                                  df_feat['visual blurring'] +
                                  df_feat['delayed healing'])
    print("   [OK] 代谢综合征评分 (Metabolic_Score): 肥胖+乏力+视力模糊+愈合延迟 [0-4]")

    df_feat['Neural_Score'] = (df_feat['partial paresis'] +
                               df_feat['muscle stiffness'] +
                               df_feat['Alopecia'])
    print("   [OK] 神经症状评分 (Neural_Score): 局部麻痹+肌肉僵硬+脱发 [0-3]")

    df_feat['Age_obesity_interaction'] = df_feat['Age'] * df_feat['Obesity']
    print("   [OK] 年龄-肥胖交互特征 (Age x Obesity)")

    df_feat['Age_polyuria_interaction'] = df_feat['Age'] * df_feat['Polyuria']
    print("   [OK] 年龄-多尿交互特征 (Age x Polyuria)")

    symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
                    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
                    'Irritability', 'delayed healing', 'partial paresis',
                    'muscle stiffness', 'Alopecia', 'Obesity']
    df_feat['Total_Symptoms'] = df_feat[symptom_cols].sum(axis=1)
    print("   [OK] 症状总数 (Total_Symptoms) [0-14]")

    print(f"\n   [数据] 特征构建完成: 原始 {len(df_encoded.columns)} 列 -> {len(df_feat.columns)} 列")

    # ============================================================================
    # 4. 特征选择（技术要求 2）
    # ============================================================================
    print("\n" + "-" * 60)
    print("  阶段 3/4：特征选择")
    print("-" * 60)

    X = df_feat.drop(columns=['class', 'Age_Group'])
    y = df_feat['class']
    feature_target_corr = pd.concat([X, y], axis=1).corr()['class'].drop('class')

    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

    X_for_chi2 = X.copy()
    min_vals = X_for_chi2.min()
    for col in X_for_chi2.columns:
        if min_vals[col] < 0:
            X_for_chi2[col] = X_for_chi2[col] - min_vals[col]

    selector_chi2 = SelectKBest(score_func=chi2, k='all')
    selector_chi2.fit(X_for_chi2, y)

    chi2_scores = pd.DataFrame({
        '特征': X.columns,
        '卡方统计量': selector_chi2.scores_,
        'p值': selector_chi2.pvalues_
    }).sort_values('卡方统计量', ascending=False)

    print("\n[分析] 卡方检验特征重要性排序:")
    for _, row in chi2_scores.iterrows():
        stars = '*' * min(5, int(row['卡方统计量'] / 20))
        print(f"   {row['特征']:30s}  统计量={row['卡方统计量']:>8.2f}  p值={row['p值']:.6f}  {stars}")

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        '特征': X.columns,
        '互信息': mi_scores
    }).sort_values('互信息', ascending=False)

    print("\n[分析] 互信息特征重要性排序:")
    for _, row in mi_df.iterrows():
        bar = '#' * int(row['互信息'] * 100)
        print(f"   {row['特征']:30s}  {row['互信息']:.4f}  {bar}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                               random_state=RANDOM_STATE, max_iter=1000)
    lasso.fit(X_scaled, y)

    lasso_coef = pd.DataFrame({
        '特征': X.columns,
        'Lasso系数': lasso.coef_[0]
    }).sort_values('Lasso系数', key=abs, ascending=False)

    kept_features = lasso_coef[lasso_coef['Lasso系数'] != 0]
    dropped_features = lasso_coef[lasso_coef['Lasso系数'] == 0]

    print("\n[分析] Lasso 回归特征选择:")
    print(f"   保留特征 ({len(kept_features)}):")
    for _, row in kept_features.iterrows():
        direction = '[正向]' if row['Lasso系数'] > 0 else '[负向]'
        print(f"   {row['特征']:30s}  系数={row['Lasso系数']:+.4f}  {direction}")

    if len(dropped_features) > 0:
        print(f"\n   剔除特征 ({len(dropped_features)}):")
        for _, row in dropped_features.iterrows():
            print(f"   {row['特征']:30s}  (系数=0)")

    combined_importance = pd.DataFrame({'特征': X.columns})
    combined_importance = combined_importance.merge(chi2_scores, on='特征', how='left')
    combined_importance = combined_importance.merge(mi_df, on='特征', how='left')
    combined_importance = combined_importance.merge(lasso_coef, on='特征', how='left')
    combined_importance['与目标相关系数'] = combined_importance['特征'].map(feature_target_corr)
    combined_importance['与目标相关系数绝对值'] = combined_importance['与目标相关系数'].abs()
    combined_importance['卡方排名'] = combined_importance['卡方统计量'].rank(
        ascending=False, method='min')
    combined_importance['互信息排名'] = combined_importance['互信息'].rank(
        ascending=False, method='min')
    combined_importance['目标相关排名'] = combined_importance['与目标相关系数绝对值'].rank(
        ascending=False, method='min')
    combined_importance['Lasso保留'] = combined_importance['Lasso系数'].abs() > 0
    combined_importance['平均排名'] = (
        combined_importance['卡方排名'] +
        combined_importance['互信息排名'] +
        combined_importance['目标相关排名']
    ) / 3
    combined_importance['推荐等级'] = np.select(
        [
            (combined_importance['平均排名'] <= 8) & combined_importance['Lasso保留'],
            combined_importance['平均排名'] <= 15,
            combined_importance['Lasso保留']
        ],
        ['强推荐', '推荐', '备选'],
        default='剔除候选'
    )
    combined_importance = combined_importance.sort_values(
        ['平均排名', 'Lasso保留', '卡方排名'], ascending=[True, False, True])

    print("\n[结果] 综合特征重要性 TOP 10:")
    print(f"   {'排名':>4s} {'特征':30s} {'平均排名':>8s} {'Lasso':>6s} {'推荐等级':>8s}")
    for i, (_, row) in enumerate(combined_importance.head(10).iterrows()):
        lasso_flag = '[Y]' if row['Lasso保留'] else '[N]'
        print(f"   {i+1:>4d} {row['特征']:30s} {row['平均排名']:>8.1f} "
              f"{lasso_flag:>6s} {row['推荐等级']:>8s}")

    # ============================================================================
    # 5. 特征降维（PCA 可视化 + 可选项）
    # ============================================================================
    print("\n" + "-" * 60)
    print("  阶段 4/4：PCA 降维分析")
    print("-" * 60)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_scaled)

    pca_full = PCA().fit(X_scaled)
    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
    print(f"\n   PCA 解释方差:")
    print(f"   第1主成分: {pca_full.explained_variance_ratio_[0]:.2%}")
    print(f"   第2主成分: {pca_full.explained_variance_ratio_[1]:.2%}")
    print(f"   前2主成分累计: {cumulative_var[1]:.2%}")
    print(f"   达到 90% 解释方差所需主成分数: {n_components_90}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    scatter = ax[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                             c=y, cmap='coolwarm', alpha=0.6, edgecolors='k', s=30)
    ax[0].set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%})')
    ax[0].set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%})')
    ax[0].set_title('PCA 降维可视化（按类别着色）', fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax[0])
    cbar.set_label('糖尿病风险 (0=健康, 1=患病)')

    ax[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-', markersize=4)
    ax[1].axhline(y=0.90, color='r', linestyle='--', label='90% 阈值')
    ax[1].axvline(x=n_components_90, color='g', linestyle='--',
                  label=f'{n_components_90} 个主成分')
    ax[1].set_xlabel('主成分数量')
    ax[1].set_ylabel('累计解释方差')
    ax[1].set_title('PCA 累计解释方差', fontsize=13)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    ax[1].set_xticks(range(1, len(cumulative_var) + 1))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, '02_PCA降维分析.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("   [OK] PCA 降维图 -> results/02_PCA降维分析.png")

    # ============================================================================
    # 6. 汇总输出
    # ============================================================================
    print("\n" + "=" * 60)
    print("  特征工程交付物汇总")
    print("=" * 60)

    final_feature_table = combined_importance[
        combined_importance['推荐等级'].isin(['强推荐', '推荐', '备选'])
    ].copy()
    final_features = final_feature_table['特征'].tolist()

    print(f"\n [OK] 推荐最终特征集 ({len(final_features)} 个特征):")
    for i, feat in enumerate(final_features):
        level = final_feature_table.loc[final_feature_table['特征'] == feat, '推荐等级'].iloc[0]
        print(f"    {i+1:>2d}. {feat} ({level})")

    X_final = X[final_features].copy()
    X_final['class'] = y.values
    X_all_features = X.copy()
    X_all_features['class'] = y.values

    output_path = os.path.join(DATA_DIR, '特征工程输出_给C成员.csv')
    all_feature_path = os.path.join(DATA_DIR, '特征工程全量特征_给C成员.csv')
    save_csv(X_final, output_path)
    save_csv(X_all_features, all_feature_path)
    print(f"\n [OK] 最终特征矩阵已保存 -> {output_path}")
    print(f"    形状: {X_final.shape[0]} 行 x {X_final.shape[1]} 列")
    print(f" [OK] 全量构建特征矩阵 -> {all_feature_path}")
    print(f"    形状: {X_all_features.shape[0]} 行 x {X_all_features.shape[1]} 列")

    importance_summary = combined_importance.copy()
    importance_summary['字段说明'] = importance_summary['特征'].map(FEATURE_DESCRIPTIONS).fillna('')
    importance_path = os.path.join(RESULT_DIR, '特征重要性排名.csv')
    selection_detail_path = os.path.join(RESULT_DIR, '特征选择明细_给C成员.csv')
    vif_path = os.path.join(RESULT_DIR, 'VIF_多重共线性检查.csv')
    corr_path = os.path.join(RESULT_DIR, '目标相关性排序.csv')
    dict_path = os.path.join(RESULT_DIR, '特征字段说明.csv')

    save_csv(importance_summary, importance_path)
    save_csv(importance_summary, selection_detail_path)
    save_csv(vif_result.sort_values('VIF', ascending=False), vif_path)
    save_csv(
        pd.DataFrame({
            '特征': feature_target_corr.abs().sort_values(ascending=False).index,
            '与目标相关系数': feature_target_corr.loc[
                feature_target_corr.abs().sort_values(ascending=False).index
            ].values,
            '相关强度': feature_target_corr.abs().sort_values(ascending=False).values
        }),
        corr_path
    )
    feature_dict = pd.DataFrame({
        '特征': X_all_features.columns,
        '字段说明': [FEATURE_DESCRIPTIONS.get(col, '') for col in X_all_features.columns],
        '是否推荐给C建模': ['是' if col in final_features or col == 'class' else '否'
                     for col in X_all_features.columns]
    })
    save_csv(feature_dict, dict_path)
    print(f" [OK] 特征重要性排名 -> results/特征重要性排名.csv")
    print(f" [OK] C 成员特征选择明细 -> results/特征选择明细_给C成员.csv")

    save_csv(df_encoded, os.path.join(DATA_DIR, '数据_编码后.csv'))
    print(f" [OK] 编码后完整数据 -> data/数据_编码后.csv")

    top_features_text = '\n'.join([
        f"{i+1}. {row['特征']}：平均排名 {row['平均排名']:.1f}，"
        f"推荐等级 {row['推荐等级']}，说明：{FEATURE_DESCRIPTIONS.get(row['特征'], '')}"
        for i, (_, row) in enumerate(combined_importance.head(10).iterrows())
    ])
    c_member_note = f"""# 成员 B 特征工程交付说明

## 推荐给 C 成员优先使用的数据

- 主推文件：`data/特征工程输出_给C成员.csv`
- 对照文件：`data/特征工程全量特征_给C成员.csv`
- 标签字段：`class`，其中 `1=Positive/患病风险`，`0=Negative/非患病风险`
- 推荐建模方式：先用主推文件训练 Logistic Regression、Random Forest、XGBoost/LightGBM；再用全量特征文件做对照实验。

## TOP 10 重要特征

{top_features_text}
"""
    note_path = os.path.join(RESULT_DIR, 'B成员交付说明_给C成员.md')
    with open(note_path, 'w', encoding='utf-8') as f:
        f.write(c_member_note)
    print(f" [OK] C 成员说明文档 -> results/B成员交付说明_给C成员.md")

    print("\n" + "=" * 60)
    print("  成员 B 特征工程任务完成！")
    print("=" * 60)

# ============================================================================
# 程序入口：告诉 Python 只有被直接运行时才执行 main()
# ============================================================================
if __name__ == '__main__':
    main()