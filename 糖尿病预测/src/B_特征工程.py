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
import sys
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

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 60)
print("  成员 B：特征工程模块")
print("=" * 60)

df_raw = pd.read_csv(os.path.join(DATA_DIR, '糖尿病预测.csv'))
print(f"\n[OK] 数据加载成功: {df_raw.shape[0]} 行 x {df_raw.shape[1]} 列\n")

# ============================================================================
# 2. EDA 与 相关性分析（技术要求 3）
# ============================================================================
print("-" * 60)
print("  阶段 1/4：EDA 与相关性分析")
print("-" * 60)

# 2.1 区分特征类型
target = 'class'
numerical_features = ['Age']
binary_features = [col for col in df_raw.columns
                   if col not in numerical_features + [target]]

print(f"  数值特征: {numerical_features}")
print(f"  二分类特征: {binary_features}")
print(f"  目标变量: {target}")

# 2.2 类别特征编码（Yes/No -> 1/0，Male/Female -> 1/0）
df_encoded = df_raw.copy()
df_encoded['Gender'] = df_encoded['Gender'].map({'Male': 1, 'Female': 0})
for col in binary_features:
    if col != 'Gender':  # Gender 已处理
        df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})

# 目标编码：Positive=1, Negative=0
df_encoded['class'] = df_encoded['class'].map({'Positive': 1, 'Negative': 0})

print("\n[数据] 目标变量分布:")
print(f"   正样本 (Positive): {df_encoded['class'].sum()}  ({df_encoded['class'].mean()*100:.1f}%)")
print(f"   负样本 (Negative): {(1-df_encoded['class']).sum()}  ({(1-df_encoded['class'].mean())*100:.1f}%)")

# 2.3 相关性热力图（核心交付物）
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

# 2.4 与目标变量的相关性排序
target_corr = corr_matrix['class'].drop('class').sort_values(ascending=False)
print("\n[分析] 各特征与糖尿病风险的相关性（从高到低）:")
for feat, corr_val in target_corr.items():
    bar = '#' * int(abs(corr_val) * 30)
    print(f"   {feat:25s}  {corr_val:+.4f}  {bar}")

# 2.5 多重共线性检查（VIF 方差膨胀因子）
from sklearn.linear_model import LinearRegression

def calculate_vif(df, features):
    """计算 VIF 方差膨胀因子"""
    vif_data = pd.DataFrame()
    vif_data['特征'] = features
    vif_data['VIF'] = 0.0
    X = df[features].values
    for i, feature in enumerate(features):
        other_features = [f for f in features if f != feature]
        if len(other_features) == 0:
            vif_data.loc[i, 'VIF'] = 1.0
            continue
        model = LinearRegression()
        model.fit(X[:, [df.columns.get_loc(f) for f in other_features]],
                  X[:, df.columns.get_loc(feature)])
        r2 = model.score(X[:, [df.columns.get_loc(f) for f in other_features]],
                         X[:, df.columns.get_loc(feature)])
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

# 3.1 年龄分组特征
df_feat['Age_Group'] = pd.cut(df_feat['Age'],
                               bins=[0, 30, 40, 50, 60, 100],
                               labels=['<30', '30-40', '40-50', '50-60', '60+'])

# One-Hot 编码年龄分组
age_dummies = pd.get_dummies(df_feat['Age_Group'], prefix='Age')
df_feat = pd.concat([df_feat, age_dummies.astype(int)], axis=1)
print("   [OK] 年龄分组特征 (Age_<30, Age_30-40, Age_40-50, Age_50-60, Age_60+)")

# 3.2 典型糖尿病症状组合评分（基于医学常识）
# 多尿(Polyuria) + 烦渴(Polydipsia) + 多食(Polyphagia) = 典型"三多"症状
df_feat['Triad_Score'] = (df_feat['Polyuria'] +
                          df_feat['Polydipsia'] +
                          df_feat['Polyphagia'])
print("   [OK] 三多症状评分 (Triad_Score): 多尿+烦渴+多食 [0-3]")

# 3.3 代谢综合征相关症状评分
df_feat['Metabolic_Score'] = (df_feat['Obesity'] +
                              df_feat['weakness'] +
                              df_feat['visual blurring'] +
                              df_feat['delayed healing'])
print("   [OK] 代谢综合征评分 (Metabolic_Score): 肥胖+乏力+视力模糊+愈合延迟 [0-4]")

# 3.4 神经症状评分
df_feat['Neural_Score'] = (df_feat['partial paresis'] +
                           df_feat['muscle stiffness'] +
                           df_feat['Alopecia'])
print("   [OK] 神经症状评分 (Neural_Score): 局部麻痹+肌肉僵硬+脱发 [0-3]")

# 3.5 年龄 x 肥胖 交互特征（年龄越大+肥胖 -> 风险更高）
df_feat['Age_obesity_interaction'] = df_feat['Age'] * df_feat['Obesity']
print("   [OK] 年龄-肥胖交互特征 (Age x Obesity)")

# 3.6 年龄 x 多尿交互特征
df_feat['Age_polyuria_interaction'] = df_feat['Age'] * df_feat['Polyuria']
print("   [OK] 年龄-多尿交互特征 (Age x Polyuria)")

# 3.7 症状总数（反映出整体健康状况）
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

# 分离 X, y
X = df_feat.drop(columns=['class', 'Age_Group'])  # Age_Group 是编码前的原始列
y = df_feat['class']

# 4.1 SelectKBest (卡方检验 + 互信息)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif

# 方法1：卡方检验（适用于非负特征）
# 找出所有特征的最小值，如果有负数就偏移
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

# 4.2 互信息（捕捉非线性关系）
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({
    '特征': X.columns,
    '互信息': mi_scores
}).sort_values('互信息', ascending=False)

print("\n[分析] 互信息特征重要性排序:")
for _, row in mi_df.iterrows():
    bar = '#' * int(row['互信息'] * 100)
    print(f"   {row['特征']:30s}  {row['互信息']:.4f}  {bar}")

# 4.3 Lasso 特征选择（嵌入法）
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                           random_state=42, max_iter=1000)
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

# 4.4 综合特征重要性排名
combined_importance = pd.DataFrame({
    '特征': X.columns,
    '卡方排名': chi2_scores['卡方统计量'].rank(ascending=False).values,
    '互信息排名': mi_df['互信息'].rank(ascending=False).values,
    'Lasso保留': lasso_coef['Lasso系数'].abs() > 0
})
combined_importance['平均排名'] = (combined_importance['卡方排名'] +
                                   combined_importance['互信息排名']) / 2
combined_importance = combined_importance.sort_values('平均排名')

print("\n[结果] 综合特征重要性 TOP 10:")
print(f"   {'排名':>4s} {'特征':30s} {'平均排名':>8s} {'Lasso':>6s}")
for i, (_, row) in enumerate(combined_importance.head(10).iterrows()):
    lasso_flag = '[Y]' if row['Lasso保留'] else '[N]'
    print(f"   {i+1:>4d} {row['特征']:30s} {row['平均排名']:>8.1f} {lasso_flag:>6s}")

# ============================================================================
# 5. 特征降维（PCA 可视化 + 可选项）
# ============================================================================
print("\n" + "-" * 60)
print("  阶段 4/4：PCA 降维分析")
print("-" * 60)

from sklearn.decomposition import PCA

# PCA 降维到 2D 用于可视化
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

# PCA 解释方差
pca_full = PCA().fit(X_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

# 找到需要多少主成分能解释 90% 方差
n_components_90 = np.argmax(cumulative_var >= 0.90) + 1
print(f"\n   PCA 解释方差:")
print(f"   第1主成分: {pca_full.explained_variance_ratio_[0]:.2%}")
print(f"   第2主成分: {pca_full.explained_variance_ratio_[1]:.2%}")
print(f"   前2主成分累计: {cumulative_var[1]:.2%}")
print(f"   达到 90% 解释方差所需主成分数: {n_components_90}")

# PCA 散点图可视化
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# 图1: PCA 散点（按类别着色）
scatter = ax[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                         c=y, cmap='coolwarm', alpha=0.6, edgecolors='k', s=30)
ax[0].set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%})')
ax[0].set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%})')
ax[0].set_title('PCA 降维可视化（按类别着色）', fontsize=13)
cbar = plt.colorbar(scatter, ax=ax[0])
cbar.set_label('糖尿病风险 (0=健康, 1=患病)')

# 图2: 累计解释方差
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

# 6.1 建议保留的特征集（综合特征选择 + Lasso）
# 选择评级高的特征 + Lasso 非零的特征
high_importance_cols = combined_importance.head(15)['特征'].tolist()
final_features = [col for col in high_importance_cols
                  if col in kept_features['特征'].values]  # Lasso 非零
# 补充一些重要特征即使 Lasso 为零
for col in high_importance_cols[:10]:
    if col not in final_features:
        final_features.append(col)

print(f"\n [OK] 推荐最终特征集 ({len(final_features)} 个特征):")
for i, feat in enumerate(final_features):
    print(f"    {i+1:>2d}. {feat}")

# 6.2 输出最终特征矩阵（给 C 成员）
X_final = X[final_features].copy()
X_final['class'] = y.values

output_path = os.path.join(DATA_DIR, '特征工程输出_给C成员.csv')
X_final.to_csv(output_path, index=False)
print(f"\n [OK] 最终特征矩阵已保存 -> {output_path}")
print(f"    形状: {X_final.shape[0]} 行 x {X_final.shape[1]} 列")

# 6.3 保存特征重要性
importance_summary = combined_importance.copy()
importance_summary.to_csv(os.path.join(RESULT_DIR, '特征重要性排名.csv'), index=False)
print(f" [OK] 特征重要性排名 -> results/特征重要性排名.csv")

# 6.4 保存编码后的完整数据（给 A 成员核对）
df_encoded.to_csv(os.path.join(DATA_DIR, '数据_编码后.csv'), index=False)
print(f" [OK] 编码后完整数据 -> data/数据_编码后.csv")

print("\n" + "=" * 60)
print("  成员 B 特征工程任务完成！")
print("=" * 60)
print(f"""
交付清单:
   |-- results/01_相关性热力图.png        <- 相关性分析报告
   |-- results/02_PCA降维分析.png         <- 降维可视化
   |-- results/特征重要性排名.csv          <- 特征重要性排序
   |-- data/特征工程输出_给C成员.csv       <- 最终特征矩阵（给 C）
   |-- data/数据_编码后.csv               <- 编码后数据（给 A 核对）

关键发现:
   - 与糖尿病最相关特征: {target_corr.index[0]}({target_corr.values[0]:.3f}),
     {target_corr.index[1]}({target_corr.values[1]:.3f})
   - 共线性风险 {"有" if any(vif_result['VIF'] > 5) else "无"}
""")
