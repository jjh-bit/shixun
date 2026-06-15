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

# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 60)
print("  成员 B：特征工程模块")
print("=" * 60)

input_file = os.path.join(DATA_DIR, 'cleaned_糖尿病预测.csv')
if not os.path.exists(input_file):
    input_file = os.path.join(DATA_DIR, '糖尿病预测.csv')

df_raw = pd.read_csv(input_file)

print(f"\n[OK] 数据加载成功: {df_raw.shape[0]} 行 x {df_raw.shape[1]} 列\n")

# ============================================================================
# 2. EDA 与 相关性分析（技术要求 2）
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

if df_encoded.isnull().any().any():
    missing_cols = df_encoded.columns[df_encoded.isnull().any()].tolist()
    raise ValueError(f"编码后存在缺失值，请检查字段取值: {missing_cols}")

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
feature_target_corr = pd.concat([X, y], axis=1).corr()['class'].drop('class')

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
mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
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

# 4.4 综合特征重要性排名
# 注意：不同方法排序后行顺序会变化，必须按“特征名”对齐，不能直接拼 values。
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

# 6.1 建议保留的特征集
# 规则：强相关/高互信息/卡方显著的特征优先，同时保留 Lasso 认为有用的稀疏特征。
final_feature_table = combined_importance[
    combined_importance['推荐等级'].isin(['强推荐', '推荐', '备选'])
].copy()
final_features = final_feature_table['特征'].tolist()

print(f"\n [OK] 推荐最终特征集 ({len(final_features)} 个特征):")
for i, feat in enumerate(final_features):
    level = final_feature_table.loc[final_feature_table['特征'] == feat, '推荐等级'].iloc[0]
    print(f"    {i+1:>2d}. {feat} ({level})")

# 6.2 输出最终特征矩阵（给 C 成员）
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

# 6.3 保存分析明细
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

# 6.4 保存编码后的完整数据（给 A 成员核对）
save_csv(df_encoded, os.path.join(DATA_DIR, '数据_编码后.csv'))
print(f" [OK] 编码后完整数据 -> data/数据_编码后.csv")

# 6.5 生成给 C 成员的说明文档
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

## 特征工程方法

1. 类别编码：Yes/No 编码为 1/0，Gender 中 Male=1、Female=0。
2. 医学组合特征：构建三多症状评分、代谢症状评分、神经症状评分、症状总数。
3. 交互特征：构建年龄与肥胖、年龄与多尿两个交互项。
4. 特征筛选：综合卡方检验、互信息、Lasso 稀疏回归、目标相关性，按平均排名输出推荐等级。
5. 共线性检查：使用 VIF，当前结论为 {'存在较高共线性风险' if any(vif_result['VIF'] > 5) else '未发现明显共线性风险'}。

## TOP 10 重要特征

{top_features_text}

## C 成员建模建议

1. 读取 `data/特征工程输出_给C成员.csv`，将 `class` 作为 y，其余列作为 X。
2. 使用分层划分训练集/测试集，建议 `test_size=0.2`、`random_state={RANDOM_STATE}`、`stratify=y`。
3. 医疗场景优先关注 Recall、AUC-ROC、F1。若 Recall 偏低，可调低分类阈值，例如从 0.5 调到 0.4。
4. 树模型可直接使用当前特征；逻辑回归、SVM、MLP 建议先做 StandardScaler。
5. 报告中建议同时展示“原始编码特征 vs B 筛选特征 vs B 全量构建特征”的模型指标对比。

## 交付文件清单

- `data/特征工程输出_给C成员.csv`：推荐筛选后的建模数据。
- `data/特征工程全量特征_给C成员.csv`：包含全部构建特征的对照数据。
- `results/特征重要性排名.csv`：综合重要性排名。
- `results/特征选择明细_给C成员.csv`：给 C 成员查看的筛选依据。
- `results/特征字段说明.csv`：字段含义和是否推荐建模。
- `results/VIF_多重共线性检查.csv`：多重共线性检查结果。
- `results/目标相关性排序.csv`：各特征与目标变量的相关性。
- `results/01_相关性热力图.png`：原始编码特征相关性热力图。
- `results/02_PCA降维分析.png`：PCA 降维可视化。
"""
note_path = os.path.join(RESULT_DIR, 'B成员交付说明_给C成员.md')
with open(note_path, 'w', encoding='utf-8') as f:
    f.write(c_member_note)
print(f" [OK] C 成员说明文档 -> results/B成员交付说明_给C成员.md")

report_text = f"""# 成员 B 特征工程报告材料

## 任务定位

本模块对应糖尿病预测项目技术要求 2：针对健康数据中可能存在的特征冗余和噪声干扰问题，完成特征构建、特征选择、相关性分析与降维可视化，为后续模型训练提供更具区分度的结构化特征。

## 数据基础

原始数据共 {df_raw.shape[0]} 条样本、{df_raw.shape[1]} 个字段。目标变量 `class` 被编码为二分类标签，其中 Positive=1、Negative=0。编码后正样本 {int(df_encoded['class'].sum())} 条，占比 {df_encoded['class'].mean()*100:.1f}%；负样本 {int((1-df_encoded['class']).sum())} 条，占比 {(1-df_encoded['class'].mean())*100:.1f}%。

## 特征构建

在原始编码特征基础上，结合糖尿病常见症状与医学常识构建了以下衍生特征：

- `Triad_Score`：多尿、烦渴、多食的“三多症状”评分。
- `Metabolic_Score`：肥胖、乏力、视力模糊、愈合延迟组成的代谢相关评分。
- `Neural_Score`：局部麻痹、肌肉僵硬、脱发组成的神经相关评分。
- `Age_obesity_interaction`：年龄与肥胖的交互项。
- `Age_polyuria_interaction`：年龄与多尿的交互项。
- `Total_Symptoms`：全部症状数量汇总。
- `Age_<30`、`Age_30-40`、`Age_40-50`、`Age_50-60`、`Age_60+`：年龄分组特征。

经过特征构建后，特征矩阵由原始 {len(df_encoded.columns)} 列扩展为 {len(df_feat.columns)} 列。

## 特征选择方法

为避免单一方法带来的偏差，本模块综合使用四类依据进行筛选：

1. 卡方检验：衡量离散特征与糖尿病标签之间的统计关联。
2. 互信息：捕捉特征与标签之间可能存在的非线性关系。
3. Lasso 逻辑回归：通过 L1 正则化筛除弱贡献特征。
4. 目标相关性：衡量特征与患病风险之间的线性相关强度。

最终根据平均排名和 Lasso 保留情况划分为“强推荐、推荐、备选、剔除候选”四类，并输出 `results/特征重要性排名.csv`。

## 关键结果

TOP 10 特征如下：

{top_features_text}

相关性分析显示，原始症状中与糖尿病风险关联最强的是 `{target_corr.index[0]}`，相关系数为 {target_corr.values[0]:.3f}；其次为 `{target_corr.index[1]}`，相关系数为 {target_corr.values[1]:.3f}。VIF 多重共线性检查结果显示，{'存在需要关注的共线性风险' if any(vif_result['VIF'] > 5) else '各特征 VIF 均处于较低水平，未发现明显多重共线性问题'}。

## 降维分析

PCA 可视化用于观察样本在低维空间中的分布情况。前两个主成分累计解释方差为 {cumulative_var[1]:.2%}，达到 90% 累计解释方差需要 {n_components_90} 个主成分，说明糖尿病风险并非由单一特征决定，而是由多个症状和交互特征共同贡献。

## 给模型训练的交付

本模块向 C 成员提供两份数据：

- `data/特征工程输出_给C成员.csv`：推荐筛选后的 {len(final_features)} 个特征加标签，作为主推建模输入。
- `data/特征工程全量特征_给C成员.csv`：全部构建特征加标签，用于与主推特征集进行对照实验。

建议 C 成员使用分层训练/测试集划分，并重点比较 Recall、AUC-ROC 和 F1-Score，以满足医疗预测中减少漏诊的需求。
"""
report_path = os.path.join(RESULT_DIR, 'B成员报告材料.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f" [OK] B 成员报告材料 -> results/B成员报告材料.md")

print("\n" + "=" * 60)
print("  成员 B 特征工程任务完成！")
print("=" * 60)
print(f"""
交付清单:
   |-- results/01_相关性热力图.png        <- 相关性分析报告
   |-- results/02_PCA降维分析.png         <- 降维可视化
   |-- results/特征重要性排名.csv          <- 特征重要性排序
   |-- results/B成员交付说明_给C成员.md     <- C 成员操作说明
   |-- results/B成员报告材料.md            <- D 成员整合报告材料
   |-- data/特征工程输出_给C成员.csv       <- 最终特征矩阵（给 C）
   |-- data/特征工程全量特征_给C成员.csv    <- 全量构建特征矩阵（给 C 对照）
   |-- data/数据_编码后.csv               <- 编码后数据（给 A 核对）

关键发现:
   - 与糖尿病最相关特征: {target_corr.index[0]}({target_corr.values[0]:.3f}),
     {target_corr.index[1]}({target_corr.values[1]:.3f})
   - 共线性风险 {"有" if any(vif_result['VIF'] > 5) else "无"}
""")
