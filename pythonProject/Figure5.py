import shap
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import plotly.express as px
import pickle





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
feature_names = x.columns.tolist()
dimension_map = {
    'Greed': x_greed,
    'Opportunity': x_opportunity,
    'Need': x_need,
    'Exposure': x_exposure
}


model_path = r'Model\RF_model.pkl'
# 加载模型
model = pickle.load(open(model_path, 'rb'))
#根据模型自带的特征重要性排序，计算各维度的重要性及占比
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


#################################################绘图代码#########################################
# ===================== 构造旭日图用的全量数据 =====================
sunburst_data = []

def build_sunburst_df(feature_list, dimension_name):
    df_dim = importance_df[importance_df["Feature"].isin(feature_list)].copy()
    df_dim["Dimension"] = dimension_name
    return df_dim[["Dimension", "Feature", "Importance"]]

sunburst_data.append(build_sunburst_df(x_greed, "Greed"))
sunburst_data.append(build_sunburst_df(x_opportunity, "Opportunity"))
sunburst_data.append(build_sunburst_df(x_need, "Need"))
sunburst_data.append(build_sunburst_df(x_exposure, "Exposure"))

sunburst_data = pd.concat(sunburst_data, axis=0, ignore_index=True)

print(sunburst_data.head())
# ===================== 绘制完整旭日图（维度 + 全量变量） =====================
fig = px.sunburst(
    sunburst_data,
    path=["Dimension", "Feature"],
    values="Importance",
    color="Dimension",
    color_discrete_sequence=['#B2DFDB', '#FFEB3B', '#FF7043', '#64B5F6']
)

fig.update_traces(
    textinfo="label+percent entry",
    insidetextorientation="radial",
    insidetextfont=dict(size=48)  # 调整内圈字体大小
)

fig.update_layout(
    title="",
    font=dict(family="Times New Roman", size=18),
    width=2400,
    height=2400
)

fig.show()
# ===== PDF（投稿用）=====
fig.write_image(
    r"Figures\Figure5.pdf"
)

# ===== PNG（汇报 / PPT 用）=====
fig.write_image(
    r"Figures\Figure5.png",
    scale=3
)




























