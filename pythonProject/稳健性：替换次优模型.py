import pickle
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')
import matplotlib.cm as cm
import numpy as np
import plotly.express as px


model_path = r'Model\lightgbm_model.pkl'

df = pd.read_excel('Data.xlsx')
y = df['EV'].astype(int)  # 确保y是整数类型
# Panel A: Greed (贪婪维度)
x_greed = [
    "Chairman_sex", "Chairman_age", "Chairman_oversea", "Chairman_finance",
    "Chairman_accounting", "Chairman_academic", "Chairman_environment",
    "CEO_sex", "CEO_age", "CEO_oversea", "CEO_finance", "CEO_accounting",
    "CEO_academic", "CEO_environment", "Dual", "Incentive"
]

# Panel B: Opportunity (机会维度)
x_opportunity = [
    "Institution_ratio", "Top1_ratio", "Foreign_ratio", "Management",
    "Disclosure", "Big4", "Analyst", "Media", "Politics", "Controls"
]

# Panel C: Need (需求维度)
x_need = [
    "Lev", "FC", "Liq", "ROA", "ROE", "Growth", "Profit",
    "Competition", "Dividend", "Coverage"
]

# Panel D: Exposure (风险暴露维度)
x_exposure = [
    "Regulation", "Environmental_target", "Growth_target", "ESG",
    "Credit", "Inspection", "Interview", "Court", "Concern", "Investor"
]

#基本面包含Size、Age和从Ind1到Ind47的变量
x_basic = ["Size", "Age"] + [f"Ind{i}" for i in range(1, 48)]

# 合并所有特征
x = df[x_greed + x_opportunity + x_need + x_exposure + x_basic].astype(float)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,stratify=y, random_state=666)
#用RandomUnderSampler方法平衡数据
rus = RandomUnderSampler(random_state=666)
x_train, y_train = rus.fit_resample(x_train, y_train)

# 加载模型
model = pickle.load(open(model_path, 'rb'))
# 计算 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# 把所有变量重要性导出到一个表格中
importance_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': np.abs(shap_values[1]).mean(axis=0)
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# importance_df.to_excel(r'稳健性\lightgbm最优模型变量重要性.xlsx', index=False)

# 把四个维度的变量重要性分别导出统计到表格
greed_importance = importance_df[importance_df['Feature'].isin(x_greed)].copy()
opportunity_importance = importance_df[importance_df['Feature'].isin(x_opportunity)].copy()
need_importance = importance_df[importance_df['Feature'].isin(x_need)].copy()
exposure_importance = importance_df[importance_df['Feature'].isin(x_exposure)].copy()



# 使用 pd.concat 安全地添加总计行
greed_importance = pd.concat([greed_importance, pd.DataFrame({'Feature': ['贪婪维度总重要性'], 'Importance': [greed_importance['Importance'].sum()]})], ignore_index=True)
opportunity_importance = pd.concat([opportunity_importance, pd.DataFrame({'Feature': ['机会维度总重要性'], 'Importance': [opportunity_importance['Importance'].sum()]})], ignore_index=True)
need_importance = pd.concat([need_importance, pd.DataFrame({'Feature': ['需求维度总重要性'], 'Importance': [need_importance['Importance'].sum()]})], ignore_index=True)
exposure_importance = pd.concat([exposure_importance, pd.DataFrame({'Feature': ['风险暴露维度总重要性'], 'Importance': [exposure_importance['Importance'].sum()]})], ignore_index=True)



# 把四个维度变量重要性导出到一个表格中
with pd.ExcelWriter(r'稳健性\lightgbm各维度变量重要性.xlsx') as writer:
    greed_importance.to_excel(writer, sheet_name='贪婪维度变量重要性', index=False)
    opportunity_importance.to_excel(writer, sheet_name='机会维度变量重要性', index=False)
    need_importance.to_excel(writer, sheet_name='需求维度变量重要性', index=False)
    exposure_importance.to_excel(writer, sheet_name='风险暴露维度变量重要性', index=False)

#################################################绘图代码#########################################
# 定义每个维度要展示的TOP变量数量
TOP_N = 5  # 您可以根据需要修改这个数字

def get_top_n_features(feature_list, dimension_name, top_n):
    df_dim = importance_df[importance_df['Feature'].isin(feature_list)].copy()
    df_dim = df_dim.sort_values("Importance", ascending=False).head(top_n)
    df_dim["Dimension"] = dimension_name
    return df_dim
# 绘制结果
for TOP_N in range(1, 11):

    # ===== 1. 维度内 Top-N =====
    greed_top = get_top_n_features(x_greed, "Greed", TOP_N)
    opportunity_top = get_top_n_features(x_opportunity, "Opportunity", TOP_N)
    need_top = get_top_n_features(x_need, "Need", TOP_N)
    exposure_top = get_top_n_features(x_exposure, "Exposure", TOP_N)

    # ===== 2. 合并绘图数据 =====
    sunburst_data = pd.concat([
        greed_top,
        opportunity_top,
        need_top,
        exposure_top
    ], axis=0)[["Dimension", "Feature", "Importance"]]

    print(f"\n===== Top-{TOP_N} Sunburst Data =====")
    print(sunburst_data)

    # ===== 3. 绘制旭日图 =====
    fig = px.sunburst(
        sunburst_data,
        path=["Dimension", "Feature"],
        values="Importance",
        color="Dimension",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_traces(
        textinfo="label+percent entry",
        insidetextorientation="radial"
    )

    fig.update_layout(
        title=f"Top-{TOP_N} Feature Sunburst",
        font=dict(family="Times New Roman", size=16)
    )

    fig.show()

    #===== 保存图片 =====
    fig.write_image(
        fr"稳健性\Sunburst_Top{TOP_N}.pdf"
    )
    fig.update_layout(
        width=2400,
        height=2400,
        font=dict(family="Times New Roman", size=24)
    )

    fig.write_image(
        fr"稳健性\Sunburst_Top{TOP_N}.png",
        scale=2  # 推荐 2～3，别太大
    )

# greed_top = get_top_n_features(x_greed, "Greed")
# opportunity_top = get_top_n_features(x_opportunity, "Opportunity")
# need_top = get_top_n_features(x_need, "Need")
# exposure_top = get_top_n_features(x_exposure, "Exposure")
# # 合并四个维度的 Top-N 数据
# sunburst_data = pd.concat([
#     greed_top,
#     opportunity_top,
#     need_top,
#     exposure_top], axis=0)[["Dimension", "Feature", "Importance"]]
# print(sunburst_data)
# #开始绘制旭日图
# fig = px.sunburst(
#     sunburst_data,
#     path=["Dimension", "Feature"],
#     values="Importance",
#     color="Dimension",
#     color_discrete_sequence=px.colors.qualitative.Set3,
# )
#
# fig.update_traces(
#     textinfo="label+percent entry",
#     insidetextorientation="radial"
# )
#
# fig.update_layout(
#     title="",
#     font=dict(family="Times New Roman", size=16)
# )
#
# fig.show()










# # ======== 贪婪维度 Top-N ========
# greed_importance_plot = (
#     greed_importance[greed_importance['Feature'] != '贪婪维度总重要性']
#     .sort_values('Importance', ascending=False)
#     .head(TOP_N)
#     .copy()
# )
# greed_importance_plot["Dimension"] = "Greed"
#
# # ======== 机会维度 Top-N ========
# opportunity_importance_plot = (
#     opportunity_importance[opportunity_importance['Feature'] != '机会维度总重要性']
#     .sort_values('Importance', ascending=False)
#     .head(TOP_N)
#     .copy()
# )
# opportunity_importance_plot["Dimension"] = "Opportunity"
#
#
# # ======== 需求维度 Top-N ========
# need_importance_plot = (
#     need_importance[need_importance['Feature'] != '需求维度总重要性']
#     .sort_values('Importance', ascending=False)
#     .head(TOP_N)
#     .copy()
# )
# need_importance_plot["Dimension"] = "Need"
# # ======== 风险暴露维度 Top-N ========
# exposure_importance_plot = (
#     exposure_importance[exposure_importance['Feature'] != '风险暴露维度总重要性']
#     .sort_values('Importance', ascending=False)
#     .head(TOP_N)
#     .copy()
# )
#
# # 合并四个维度的数据
# sunburst_df = pd.concat([
#     greed_importance_plot,
#     opportunity_importance_plot,
#     need_importance_plot,
#     exposure_importance_plot
# ], axis=0)
#
# # 排除 “维度总重要性” 这几行，只保留小指标
# sunburst_df_features = sunburst_df[
#     ~sunburst_df['Feature'].isin(["Greed", "Opportunity", "Need", "Exposure"])
# ]
#
# # 构建最终旭日图数据表
# sunburst_data = pd.DataFrame({
#     "Dimension": sunburst_df_features["Dimension"],
#     "Feature": sunburst_df_features["Feature"],
#     "Importance": sunburst_df_features["Importance"]
# })
#
# # ========== 2. 绘制旭日图 ==========
# fig = px.sunburst(
#     sunburst_data,
#     path=["Dimension", "Feature"],   # 层级结构
#     values="Importance",             # 重要性数值
#     color="Dimension",               # 按维度上色
#     color_discrete_sequence=px.colors.qualitative.Set3,  # 更柔和、更适合论文展示
#     maxdepth=-1                      # 展示所有层级
# )
#
# fig.update_traces(
#     textinfo="label+percent entry",  # 显示标签 + 占比
#     insidetextorientation='radial'
# )
#
# fig.update_layout(
#     title="",
#     title_x=0.5,
#     font=dict(family="Times New Roman", size=16),
# )
#
# # ========== 3. 显示图形 ==========
# fig.show()
#
# # ========== 4. 保存高清 PNG ==========
# fig.write_image(r"最优模型分析结果\变量旭日图.png", scale=10)
